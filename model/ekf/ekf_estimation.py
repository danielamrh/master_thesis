import torch
from scipy.spatial.transform import Rotation as R
import numpy as np

class EKF:
    """
    19-state EKF with robust tuning and dynamic process noise for smoother output.
    - SMOOTHER TUNING: The default Q and R values are tuned to reduce jittering.
    - DYNAMIC Q: A new method allows the main loop to adjust process noise based on movement dynamics.
    """
    def __init__(self, device='cpu', dtype=torch.float32, dt=0.01):
        self.device = device
        self.dtype = dtype
        self.dt = dt
        
        self.state_dim = 19
        self.x = torch.zeros(self.state_dim, 1, device=device, dtype=dtype)
        self.P = torch.eye(self.state_dim, device=device, dtype=dtype) * 0.1

        # --- NEW TUNING: A less aggressive, smoother baseline ---
        p_noise = 1e-5     # Lower position uncertainty from model
        v_noise = 1e-3     # Lower velocity uncertainty for smoother integration
        b_noise = 2e-5     # Allow bias to adapt, but not erratically
        uwb_b_noise = 0.05 # UWB bias changes relatively slowly

        Q_diag_sensor = torch.tensor([
            p_noise, p_noise, p_noise, 
            v_noise, v_noise, v_noise,
            b_noise, b_noise, b_noise
        ], device=device, dtype=dtype)
        
        Q_diag = torch.cat((Q_diag_sensor, Q_diag_sensor, torch.tensor([uwb_b_noise], device=device, dtype=dtype)))
        self.Q_base = torch.diag(Q_diag)
        self.Q = self.Q_base.clone()

        # --- NEW TUNING: Increased R to be more skeptical of UWB noise ---
        # This is a key parameter for reducing jitter.
        self.R_dist_base = torch.eye(1, device=device, dtype=dtype) * 0.15**2 
        self.R_zvu = torch.eye(3, device=device, dtype=dtype) * (0.01**2)

        self.acc_p_buffer = []
        self.acc_w_buffer = []
        self.buffer_size = 5
        self.zvu_threshold = 0.0001

    def set_dynamic_process_noise(self, acc_magnitude):
        """
        Adjusts the process noise Q based on movement dynamics.
        - Increases Q during high acceleration to trust measurements more.
        - Decreases Q during low acceleration to trust the model more (smoother).
        """
        # Scale factor starts at 1.0 (calm) and can go up to e.g., 50.0 (dynamic)
        scale_factor = 1.0 + 49.0 * (1.0 - np.exp(-0.5 * acc_magnitude))
        
        self.Q = self.Q_base * scale_factor
        # Ensure bias noise doesn't grow excessively
        self.Q[6:9, 6:9] = self.Q_base[6:9, 6:9] * np.sqrt(scale_factor)
        self.Q[15:18, 15:18] = self.Q_base[15:18, 15:18] * np.sqrt(scale_factor)

    def set_initial(self, p_p_init, v_p_init, p_w_init, v_w_init, 
                    b_p_init=None, b_w_init=None, b_uwb_init=0.0):
        self.x[0:3, 0] = p_p_init
        self.x[3:6, 0] = v_p_init
        if b_p_init is not None: self.x[6:9, 0] = b_p_init
        
        self.x[9:12, 0] = p_w_init
        self.x[12:15, 0] = v_w_init
        if b_w_init is not None: self.x[15:18, 0] = b_w_init
        
        self.x[18, 0] = b_uwb_init

    def predict(self, acc_p_local: torch.Tensor, R_p: torch.Tensor, acc_w_local: torch.Tensor, R_w: torch.Tensor):
        dt = self.dt
        b_p, b_w = self.x[6:9], self.x[15:18]

        acc_p_local_corr = acc_p_local.view(3,1) - b_p 
        acc_w_local_corr = acc_w_local.view(3,1) - b_w 

        acc_p_world = (R_p @ acc_p_local_corr)
        acc_w_world = (R_w @ acc_w_local_corr)

        def _dynamics(state_vec):
            x_dot = torch.zeros_like(state_vec)
            x_dot[0:3] = state_vec[3:6]
            x_dot[3:6] = acc_p_world
            x_dot[9:12] = state_vec[12:15]
            x_dot[12:15] = acc_w_world
            return x_dot
        
        k1 = _dynamics(self.x)
        k2 = _dynamics(self.x + 0.5 * dt * k1)
        k3 = _dynamics(self.x + 0.5 * dt * k2)
        k4 = _dynamics(self.x + dt * k3)
        self.x = self.x + (dt / 6.0) * (k1 + 2*k2 + 2*k3 + k4)

        F = torch.eye(self.state_dim, device=self.device, dtype=self.dtype)
        F[0:3, 3:6] = torch.eye(3) * dt 
        F[3:6, 6:9] = -R_p * dt 
        F[0:3, 6:9] = 0.5 * F[3:6, 6:9] * dt 
        
        F[9:12, 12:15] = torch.eye(3) * dt
        F[12:15, 15:18] = -R_w * dt
        F[9:12, 15:18] = 0.5 * F[12:15, 15:18] * dt
        
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
        p_p_est, p_w_est = self.x[0:3, 0], self.x[9:12, 0]
        uwb_bias_est = self.x[18, 0]

        diff_vec = p_w_est - p_p_est
        dist_est = torch.linalg.norm(diff_vec)
        if dist_est < 1e-6: return 0.0

        h = (dist_est + uwb_bias_est).view(1,1)
        y = torch.tensor([[dist_meas]], device=self.device, dtype=self.dtype)

        R_adaptive = self.R_dist_base.clone()
        innovation_abs = torch.abs(y - h).item()
        if innovation_abs > 0.25:
            R_adaptive *= (innovation_abs / 0.25)**2 * 10

        H = torch.zeros((1, self.state_dim), device=self.device, dtype=self.dtype)
        H[0, 0:3] = -diff_vec / dist_est
        H[0, 9:12] = diff_vec / dist_est
        H[0, 18] = 1.0
        
        innovation = y - h
        S = H @ self.P @ H.T + R_adaptive
        nis = innovation.T @ torch.linalg.pinv(S) @ innovation
        if nis.item() < 3.84:
            K = self.P @ H.T @ torch.linalg.pinv(S)
            self.update(y, h, H, R_adaptive, K)

        return h.item()

    # --- Other update methods (ZVU, damping, anchor) remain the same ---
    def update_zero_velocity_wrist(self, acc_w_local: torch.Tensor):
        self.acc_w_buffer.append(acc_w_local)
        if len(self.acc_w_buffer) > self.buffer_size: self.acc_w_buffer.pop(0)

        acc_variance = torch.inf
        if len(self.acc_w_buffer) == self.buffer_size:
            acc_variance = torch.var(torch.stack(self.acc_w_buffer), dim=0).mean()
            
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
        h = self.x[12:15]
        H = torch.zeros((3, self.state_dim), device=self.device, dtype=self.dtype)
        H[:, 12:15] = torch.eye(3)
        R_damping = torch.eye(3, device=self.device, dtype=self.dtype) * (10.0**2) 
        S = H @ self.P @ H.T + R_damping
        K = self.P @ H.T @ torch.linalg.pinv(S)
        self.update(y, h, H, R_damping, K)

    def update_zero_velocity_pelvis(self, acc_p_local: torch.Tensor):
        self.acc_p_buffer.append(acc_p_local) 
        if len(self.acc_p_buffer) > self.buffer_size: self.acc_p_buffer.pop(0)

        if len(self.acc_p_buffer) == self.buffer_size:
            acc_variance = torch.var(torch.stack(self.acc_p_buffer), dim=0).mean()
            if acc_variance < self.zvu_threshold:
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
        R_damping = torch.eye(3, device=self.device, dtype=self.dtype) * (1.0**2) 
        S = H @ self.P @ H.T + R_damping
        K = self.P @ H.T @ torch.linalg.pinv(S)
        self.update(y, h, H, R_damping, K)

    def update_kinematic_anchor(self, max_dist=0.9, strength=0.2):
        p_p_est, p_w_est = self.x[0:3, 0], self.x[9:12, 0]
        diff_vec = p_w_est - p_p_est
        dist = torch.linalg.norm(diff_vec)

        if dist > max_dist:
            y = torch.tensor([[max_dist]], device=self.device, dtype=self.dtype)
            h = dist.view(1, 1)
            H = torch.zeros((1, self.state_dim), device=self.device, dtype=self.dtype)
            H[0, 0:3] = -diff_vec / dist
            H[0, 9:12] = diff_vec / dist
            R_anchor = torch.eye(1, device=self.device, dtype=self.dtype) * (strength**2)
            S = H @ self.P @ H.T + R_anchor
            K = self.P @ H.T @ torch.linalg.pinv(S)
            self.update(y, h, H, R_anchor, K)

    def get_state(self):
        return self.x.clone(), self.P.clone()