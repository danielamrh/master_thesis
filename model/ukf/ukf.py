import torch
import numpy as np

class UKF:
    """
    An adaptive Unscented Kalman Filter with robust NIS-based outlier rejection.
    This is the final, tuned version.
    """
    def __init__(self, device='cpu', dtype=torch.float32, 
                 alpha=1e-3, beta=2.0, kappa=0.0,
                 p_noise=0.1, v_noise=0.5, b_noise=0.01,
                 zvu_noise=0.01, uwb_noise=0.05,
                 nis_threshold=3.84): # Chi-squared threshold for 1-DoF, 95% confidence
        
        self.device = device
        self.dtype = dtype
        self.state_dim = 18
        self.dt = 0.01 

        self.x = torch.zeros(self.state_dim, device=device, dtype=dtype)
        self.P = torch.eye(self.state_dim, device=device, dtype=dtype)

        self.alpha = alpha
        self.beta = beta
        self.kappa = kappa
        self.lambda_ = self.alpha**2 * (self.state_dim + self.kappa) - self.state_dim
        
        self.Wc = torch.full((2 * self.state_dim + 1,), 1 / (2 * (self.state_dim + self.lambda_))).to(device, dtype)
        self.Wm = self.Wc.clone()
        self.Wc[0] = self.lambda_ / (self.state_dim + self.lambda_) + (1 - self.alpha**2 + self.beta)
        self.Wm[0] = self.lambda_ / (self.state_dim + self.lambda_)
        
        dt = self.dt
        Q_block = torch.block_diag(torch.eye(3, device=device, dtype=dtype) * dt**4 / 4 * p_noise,
                                   torch.eye(3, device=device, dtype=dtype) * dt**2 * v_noise,
                                   torch.eye(3, device=device, dtype=dtype) * dt * b_noise)
        self.Q = torch.block_diag(Q_block, Q_block)

        self.R_zvu = torch.eye(3, device=device, dtype=dtype) * zvu_noise**2
        
        self.R_uwb_base = torch.eye(1, device=device, dtype=dtype) * uwb_noise**2
        self.R_uwb = self.R_uwb_base.clone()
        self.innovation_history = []
        self.adaptive_window = 10
        
        # --- DEFINITIVE FIX: Added NIS threshold for outlier rejection ---
        self.nis_threshold = nis_threshold

    def set_initial(self, p_p_init, v_p_init, p_w_init, v_w_init, b_p_init=None, b_w_init=None):
        self.x[0:3], self.x[3:6] = p_p_init, v_p_init
        self.x[9:12], self.x[12:15] = p_w_init, v_w_init
        if b_p_init is not None: self.x[6:9] = b_p_init
        if b_w_init is not None: self.x[15:18] = b_w_init

    def generate_sigma_points(self):
        try:
            # Joseph form square root for better numerical stability
            L = torch.linalg.cholesky((self.state_dim + self.lambda_) * self.P)
        except torch._C._LinAlgError:
            self.P += 1e-6 * torch.eye(self.state_dim, device=self.device, dtype=self.dtype)
            L = torch.linalg.cholesky((self.state_dim + self.lambda_) * self.P)

        sigma_points = self.x.unsqueeze(1).repeat(1, 2 * self.state_dim + 1)
        sigma_points[:, 1:self.state_dim + 1] += L
        sigma_points[:, self.state_dim + 1:] -= L
        return sigma_points

    def state_transition_model(self, state, acc_p, acc_w):
        next_state = state.clone()
        dt = self.dt
        
        p_p, v_p, b_p = state[0:3], state[3:6], state[6:9]
        next_state[0:3] = p_p + v_p * dt + 0.5 * (acc_p - b_p) * dt**2
        next_state[3:6] = v_p + (acc_p - b_p) * dt
        
        p_w, v_w, b_w = state[9:12], state[12:15], state[15:18]
        next_state[9:12] = p_w + v_w * dt + 0.5 * (acc_w - b_w) * dt**2
        next_state[12:15] = v_w + (acc_w - b_w) * dt
        
        return next_state

    def predict(self, acc_p, acc_w):
        sigma_points = self.generate_sigma_points()
        
        sigma_points_pred = torch.zeros_like(sigma_points)
        for i in range(sigma_points.shape[1]):
            sigma_points_pred[:, i] = self.state_transition_model(sigma_points[:, i], acc_p, acc_w)
            
        self.x = (self.Wm * sigma_points_pred).sum(axis=1)
        
        x_diff = sigma_points_pred - self.x.unsqueeze(1)
        self.P = (self.Wc * x_diff) @ x_diff.T + self.Q
        self.P = 0.5 * (self.P + self.P.T)

    def update_uwb_scalar_distance(self, dist_meas: float):
        def measurement_model(state):
            p_p, p_w = state[0:3], state[9:12]
            return torch.linalg.norm(p_w - p_p).unsqueeze(0)
        
        y = torch.tensor([dist_meas], device=self.device, dtype=self.dtype)
        
        sigma_points = self.generate_sigma_points()
        
        y_sigma = torch.zeros(y.shape[0], sigma_points.shape[1], device=self.device, dtype=self.dtype)
        for i in range(sigma_points.shape[1]):
            y_sigma[:, i] = measurement_model(sigma_points[:, i])
            
        y_pred = (self.Wm * y_sigma).sum(axis=1)
        
        y_diff = y_sigma - y_pred.unsqueeze(1)
        S = (self.Wc * y_diff) @ y_diff.T + self.R_uwb
        
        innovation = y - y_pred
        
        # --- Step 1: Robust Outlier Rejection using NIS ---
        nis = (innovation.T @ torch.linalg.pinv(S) @ innovation).item()
        if nis > self.nis_threshold:
            return False, nis # Measurement rejected
        # ---------------------------------------------------
        
        x_diff = sigma_points - self.x.unsqueeze(1)
        T = (self.Wc * x_diff) @ y_diff.T
        
        K = T @ torch.linalg.pinv(S)
        
        self.x += K @ innovation
        self.P -= K @ S @ K.T
        
        self.P = 0.5 * (self.P + self.P.T) + 1e-6 * torch.eye(self.state_dim, device=self.device)

        self.innovation_history.append((innovation.T @ innovation).item())
        if len(self.innovation_history) > self.adaptive_window:
            self.innovation_history.pop(0)
            
        if len(self.innovation_history) == self.adaptive_window:
            avg_innovation_sq = np.mean(self.innovation_history)
            adapted_R = self.R_uwb_base + torch.eye(1, device=self.device) * avg_innovation_sq
            self.R_uwb = torch.clamp(adapted_R, min=self.R_uwb_base.item(), max=1.0)
            
        return True, nis # Measurement accepted

    def update_zero_velocity_pelvis(self, acc_p_local: torch.Tensor):
        acc_magnitude = torch.linalg.norm(acc_p_local)
        if acc_magnitude > 0.8:
            return

        def measurement_model(state):
            return state[3:6]
        
        y = torch.zeros(3, device=self.device, dtype=self.dtype)
        
        sigma_points = self.generate_sigma_points()
        
        y_sigma = torch.zeros(y.shape[0], sigma_points.shape[1], device=self.device, dtype=self.dtype)
        for i in range(sigma_points.shape[1]):
            y_sigma[:, i] = measurement_model(sigma_points[:, i])
            
        y_pred = (self.Wm * y_sigma).sum(axis=1)
        y_diff = y_sigma - y_pred.unsqueeze(1)
        S = (self.Wc * y_diff) @ y_diff.T + self.R_zvu
        x_diff = sigma_points - self.x.unsqueeze(1)
        T = (self.Wc * x_diff) @ y_diff.T
        K = T @ torch.linalg.pinv(S)
        innovation = y - y_pred
        self.x += K @ innovation
        self.P -= K @ S @ K.T
        self.P = 0.5 * (self.P + self.P.T) + 1e-6 * torch.eye(self.state_dim, device=self.device)

    def get_state(self):
        return self.x.clone(), self.P.clone()

