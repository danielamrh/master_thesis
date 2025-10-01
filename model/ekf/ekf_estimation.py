import torch
from scipy.spatial.transform import Rotation as R
import numpy as np

class EKF:
    """
    19-state EKF for improved stability.
    - STATE SIMPLIFICATION: IMU scale factors have been removed to prevent instability.
    - TUNING: Process noise Q has been adjusted for more stable bias estimation.
    - Models IMU bias in the local frame.
    - Models a dynamically changing UWB NLOS bias.
    - Includes innovation gating and adaptive noise for UWB updates.
    """
    def __init__(self, device='cpu', dtype=torch.float32, dt=0.01):
        self.device = device
        self.dtype = dtype
        self.dt = dt
        
        # --- STATE SIMPLIFICATION: Changed from 25 to 19 states ---
        # State vector: [p_p, v_p, b_p, p_w, v_w, b_w, b_uwb]
        # (3pos, 3vel, 3bias) for pelvis + (3pos, 3vel, 3bias) for wrist + 1 UWB bias
        self.state_dim = 19
        
        self.x = torch.zeros(self.state_dim, 1, device=device, dtype=dtype)
        self.P = torch.eye(self.state_dim, device=device, dtype=dtype) * 0.1

        # --- TUNING: Adjusted Process Noise Covariance (Q) ---
        p_noise = 1e-4  # Lower position uncertainty from model
        v_noise = 1e-2  # Lower velocity uncertainty from model
        b_noise = 1e-6  # Bias changes very slowly
        uwb_b_noise = 0.1 # UWB bias also changes slowly
        
        # Sensor block now excludes scale: (pos, vel, bias)
        Q_diag_sensor = torch.tensor([
            p_noise, p_noise, p_noise, 
            v_noise, v_noise, v_noise,
            b_noise, b_noise, b_noise
        ], device=device, dtype=dtype)
        
        Q_diag = torch.cat((Q_diag_sensor, Q_diag_sensor, torch.tensor([uwb_b_noise], device=device, dtype=dtype)))
        self.Q = torch.diag(Q_diag)

        # Measurement Noise Covariance (R)
        self.R_dist_base = torch.eye(1, device=device, dtype=dtype) * 0.4**2
        self.R_zvu = torch.eye(3, device=device, dtype=dtype) * 0.02**2

        self.acc_p_buffer = []
        self.acc_w_buffer = []
        self.buffer_size = 5

        self.zvu_threshold = 0.0001

    def set_initial(self, p_p_init, v_p_init, p_w_init, v_w_init, 
                    b_p_init=None, b_w_init=None, b_uwb_init=0.0):
        self.x[0:3, 0] = p_p_init
        self.x[3:6, 0] = v_p_init
        if b_p_init is not None: self.x[6:9, 0] = b_p_init
        
        # Wrist states
        self.x[9:12, 0] = p_w_init
        self.x[12:15, 0] = v_w_init
        if b_w_init is not None: self.x[15:18, 0] = b_w_init
        
        # UWB bias
        self.x[18, 0] = b_uwb_init

    def predict(self, acc_p_local: torch.Tensor, R_p: torch.Tensor, acc_w_local: torch.Tensor, R_w: torch.Tensor):
        dt = self.dt
        
        # --- State and Correction ---
        b_p = self.x[6:9]
        b_w = self.x[15:18]
        acc_p_local_corr = acc_p_local.view(3,1) - b_p
        acc_w_local_corr = acc_w_local.view(3,1) - b_w

        # --- Use rotated acceleration ---
        acc_p_world = (R_p @ acc_p_local_corr)
        acc_w_world = (R_w @ acc_w_local_corr)

        # --- State Prediction with RK4 ---
        def _dynamics(state_vec):
            x_dot = torch.zeros_like(state_vec)
            x_dot[0:3] = state_vec[3:6]
            x_dot[3:6] = acc_p_world # Use acceleration directly
            x_dot[9:12] = state_vec[12:15]
            x_dot[12:15] = acc_w_world   # Use acceleration directly
            return x_dot
        
        k1 = _dynamics(self.x)
        k2 = _dynamics(self.x + 0.5 * dt * k1)
        k3 = _dynamics(self.x + 0.5 * dt * k2)
        k4 = _dynamics(self.x + dt * k3)
        self.x = self.x + (dt / 6.0) * (k1 + 2*k2 + 2*k3 + k4)

        # --- Update State Transition Jacobian F (simplified) ---
        F = torch.eye(self.state_dim, device=self.device, dtype=self.dtype)
        # Pelvis parts
        F[0:3, 3:6] = torch.eye(3) * dt
        F[3:6, 6:9] = -R_p * dt
        F[0:3, 6:9] = 0.5 * F[3:6, 6:9] * dt
        
        # Wrist parts (INDEXING UPDATE)
        F[9:12, 12:15] = torch.eye(3) * dt
        F[12:15, 15:18] = -R_w * dt
        F[9:12, 15:18] = 0.5 * F[12:15, 15:18] * dt
        
        # --- Covariance Prediction ---
        self.P = F @ self.P @ F.T + self.Q
        self.P = 0.5 * (self.P + self.P.T)

        return acc_w_world.squeeze(), acc_w_local_corr.squeeze()
    
    def update(self, y, h, H, R, K):
        innovation = y - h
        self.x = self.x + K @ innovation
        I = torch.eye(self.state_dim, device=self.device, dtype=self.dtype)
        self.P = (I - K @ H) @ self.P @ (I - K @ H).T + K @ R @ K.T
        self.P = 0.5 * (self.P + self.P.T) + 1e-9 * torch.eye(self.state_dim, device=self.device)

    def update_uwb_scalar_distance(self, dist_meas: float):
        # INDEXING UPDATES
        p_p_est, p_w_est = self.x[0:3, 0], self.x[9:12, 0]
        uwb_bias_est = self.x[18, 0]

        diff_vec = p_w_est - p_p_est
        dist_est = torch.linalg.norm(diff_vec)
        if dist_est < 1e-6: return 0.0

        h = (dist_est + uwb_bias_est).view(1,1)
        y = torch.tensor([[dist_meas]], device=self.device, dtype=self.dtype)

        # Adaptive Measurement Noise
        R_adaptive = self.R_dist_base.clone()
        innovation_abs = torch.abs(y - h).item()
        innovation_threshold = 0.25
        if innovation_abs > innovation_threshold:
            R_adaptive *= (innovation_abs / innovation_threshold)**2 * 10

        # Jacobian H (INDEXING UPDATES)
        H = torch.zeros((1, self.state_dim), device=self.device, dtype=self.dtype)
        H[0, 0:3] = -diff_vec / dist_est
        H[0, 9:12] = diff_vec / dist_est
        H[0, 18] = 1.0
        
        # Innovation Gating
        innovation = y - h
        S = H @ self.P @ H.T + R_adaptive
        nis = innovation.T @ torch.linalg.pinv(S) @ innovation
        gate_threshold = 3.84

        if nis.item() < gate_threshold:
            K = self.P @ H.T @ torch.linalg.pinv(S)
            self.update(y, h, H, R_adaptive, K)

        return h.item()

    def update_zero_velocity_wrist(self, acc_w_local: torch.Tensor):
        self.acc_w_buffer.append(acc_w_local)
        if len(self.acc_w_buffer) > self.buffer_size: self.acc_w_buffer.pop(0)

        acc_variance = torch.inf # Default to high variance (moving)
        if len(self.acc_w_buffer) == self.buffer_size:
            acc_variance = torch.var(torch.stack(self.acc_w_buffer), dim=0).mean()
            
            # If variance is low enough, perform a Zero-Velocity Update
            if acc_variance < self.zvu_threshold:
                y = torch.zeros(3, 1, device=self.device, dtype=self.dtype)
                h = self.x[12:15]
                H = torch.zeros((3, self.state_dim), device=self.device, dtype=self.dtype)
                H[:, 12:15] = torch.eye(3)
                S = H @ self.P @ H.T + self.R_zvu
                K = self.P @ H.T @ torch.linalg.pinv(S)
                self.update(y, h, H, self.R_zvu, K)
    
        return acc_variance 

    def update_velocity_damping_wrist(self):
        y = torch.zeros(3, 1, device=self.device, dtype=self.dtype)
        h = self.x[12:15] # INDEXING UPDATE
        H = torch.zeros((3, self.state_dim), device=self.device, dtype=self.dtype)
        H[:, 12:15] = torch.eye(3) # INDEXING UPDATE
        R_damping = torch.eye(3, device=self.device, dtype=self.dtype) * (0.5**2)
        S = H @ self.P @ H.T + R_damping
        K = self.P @ H.T @ torch.linalg.pinv(S)
        self.update(y, h, H, R_damping, K)

    def update_zero_velocity_pelvis(self, acc_p_local: torch.Tensor):
        self.acc_p_buffer.append(acc_p_local) 
        if len(self.acc_p_buffer) > self.buffer_size: self.acc_p_buffer.pop(0)

        if len(self.acc_p_buffer) == self.buffer_size:
            acc_variance = torch.var(torch.stack(self.acc_p_buffer), dim=0).mean()
            if acc_variance > 0.001: return 
            
            y = torch.zeros(3, 1, device=self.device, dtype=self.dtype)
            h = self.x[3:6]
            H = torch.zeros((3, self.state_dim), device=self.device, dtype=self.dtype)
            H[:, 3:6] = torch.eye(3)
            S = H @ self.P @ H.T + self.R_zvu
            K = self.P @ H.T @ torch.linalg.pinv(S)
            self.update(y, h, H, self.R_zvu, K)

    def update_velocity_damping_pelvis(self):
        y = torch.zeros(3, 1, device=self.device, dtype=self.dtype)
        h = self.x[3:6]
        H = torch.zeros((3, self.state_dim), device=self.device, dtype=self.dtype)
        H[:, 3:6] = torch.eye(3)
        R_damping = torch.eye(3, device=self.device, dtype=self.dtype) * (0.5**2)
        S = H @ self.P @ H.T + R_damping
        K = self.P @ H.T @ torch.linalg.pinv(S)
        self.update(y, h, H, R_damping, K)

    def update_kinematic_anchor(self, p_p_est, p_w_est, max_dist=0.75, strength=0.5):
        """
        Applies a soft constraint to keep the wrist within a plausible distance of the pelvis.
        This acts as a "sanity check" to prevent divergence during initialization.
        """
        diff_vec = p_w_est - p_p_est
        dist = torch.linalg.norm(diff_vec)

        if dist > max_dist:
            # The measurement 'y' is the desired maximum distance.
            y = torch.tensor([[max_dist]], device=self.device, dtype=self.dtype)
            # The current state 'h' is the measured distance.
            h = dist.view(1, 1)

            H = torch.zeros((1, self.state_dim), device=self.device, dtype=self.dtype)
            H[0, 0:3] = -diff_vec / dist
            H[0, 9:12] = diff_vec / dist

            # R represents our confidence in this constraint. A lower value means more confidence.
            R_anchor = torch.eye(1, device=self.device, dtype=self.dtype) * (strength**2)
            
            S = H @ self.P @ H.T + R_anchor
            K = self.P @ H.T @ torch.linalg.pinv(S)
            self.update(y, h, H, R_anchor, K)

    def get_state(self):
        return self.x.clone(), self.P.clone()