import torch
from scipy.spatial.transform import Rotation as R
import numpy as np

def wrap_angle(angle):
    return (angle + torch.pi) % (2 * torch.pi) - torch.pi

class EKFJointAngle:
    def __init__(self, bone_lengths, device='cpu', dtype=torch.float32, dt=0.01):
        self.device = device; self.dtype = dtype; self.dt = dt
        self.l1 = bone_lengths[0]; self.l2 = bone_lengths[1]
        
        self.state_dim = 7
        self.x = torch.zeros(self.state_dim, 1, device=device, dtype=dtype)
        self.P = torch.eye(self.state_dim, device=device, dtype=dtype) * 0.1

        # --- FINAL TUNING PARAMETERS ---
        q_pitch_el_pos = 1e-4
        q_pitch_el_vel = 7e-4
        q_roll_pos = 1e-4
        q_roll_vel = 5e-4
        q_uwb_bias = 1e-7
        uwb_std_dev = 0.2
        ori_std_dev = np.deg2rad(5.0)
        
        self.wrist_offset = torch.tensor([0.0, -0.02, 0.10], device=device, dtype=dtype)
        self.pelvis_offset = torch.tensor([0.10, 0.0, 0.0], device=device, dtype=dtype)
        # --- END TUNING ---

        self.Q = torch.diag(torch.tensor([
            q_pitch_el_pos, q_roll_pos, q_pitch_el_pos,
            q_pitch_el_vel, q_roll_vel, q_pitch_el_vel,
            q_uwb_bias], device=device, dtype=dtype))
        
        self.R_uwb = torch.eye(1, device=device, dtype=dtype) * (uwb_std_dev**2)
        self.R_damping = torch.eye(3, device=device, dtype=dtype) * (2.0**2)
        self.R_ori = torch.eye(3, device=device, dtype=dtype) * (ori_std_dev**2)

    def set_initial_state(self, initial_angles):
        self.x[0:3, 0] = initial_angles.squeeze(); self.x[6, 0] = 0.0

    def predict(self):
        dt = self.dt
        F = torch.eye(self.state_dim, device=self.device, dtype=self.dtype)
        F[0, 3] = dt 
        F[1, 4] = dt 
        F[2, 5] = dt
        self.x = F @ self.x 
        self.P = F @ self.P @ F.T + self.Q
        self.x[0:3] = wrap_angle(self.x[0:3])

    def _update_generic(self, y, h, H, R):
        innovation = y - h
        S = H @ self.P @ H.T + R
        K = self.P @ H.T @ torch.linalg.pinv(S)
        self.x = self.x + K @ innovation
        I = torch.eye(self.state_dim, device=self.device, dtype=self.dtype)
        self.P = (I - K @ H) @ self.P @ (I - K @ H).T + K @ R @ K.T
        self.P = 0.5 * (self.P + self.P.T)

    def update_velocity_damping(self):
        y = torch.zeros(3, 1, device=self.device, dtype=self.dtype)
        h = self.x[3:6]
        H = torch.zeros((3, self.state_dim), device=self.device, dtype=self.dtype); H[:, 3:6] = torch.eye(3)
        self._update_generic(y, h, H, self.R_damping)

    def expected_relative_orientation(self, angles=None):
        if angles is None: angles = self.x[0:3].squeeze()
        sh_pitch, sh_roll, el_pitch = angles[0], angles[1], angles[2]
        
        R_sh_roll = R.from_euler('y', sh_roll.item()).as_matrix()
        R_sh_pitch = R.from_euler('x', sh_pitch.item()).as_matrix()
        R_el_pitch = R.from_euler('x', el_pitch.item()).as_matrix()
        
        R_exp_relative = torch.tensor(R_sh_roll @ R_sh_pitch @ R_el_pitch, device=self.device, dtype=self.dtype)
        return R_exp_relative

    def update_orientation(self, R_w_world, R_p_world):
        R_meas_relative = R_p_world.T @ R_w_world
        R_exp_relative = self.expected_relative_orientation()
        R_error = R_exp_relative.T @ R_meas_relative
        error_aa = torch.tensor(R.from_matrix(R_error.cpu().numpy()).as_rotvec(), device=self.device, dtype=self.dtype).view(3, 1)
        
        R_adaptive_ori = self.R_ori.clone()
        error_angle_rad = torch.linalg.norm(error_aa)
        if error_angle_rad > np.deg2rad(15.0):
            scale = (error_angle_rad / np.deg2rad(15.0))**2
            R_adaptive_ori *= scale

        y = error_aa; h = torch.zeros(3, 1, device=self.device, dtype=self.dtype)
        H = torch.zeros((3, self.state_dim), device=self.device, dtype=self.dtype)
        epsilon = 1e-5
        for i in range(3):
            angles_plus = self.x[0:3].clone(); angles_plus[i] += epsilon
            R_exp_plus = self.expected_relative_orientation(angles=angles_plus)
            error_aa_plus = torch.tensor(R.from_matrix((R_exp_plus.T @ R_meas_relative).cpu().numpy()).as_rotvec(), device=self.device, dtype=self.dtype)
            angles_minus = self.x[0:3].clone(); angles_minus[i] -= epsilon
            R_exp_minus = self.expected_relative_orientation(angles=angles_minus)
            error_aa_minus = torch.tensor(R.from_matrix((R_exp_minus.T @ R_meas_relative).cpu().numpy()).as_rotvec(), device=self.device, dtype=self.dtype)
            H[:, i] = (error_aa_plus - error_aa_minus).squeeze() / (2 * epsilon)
            
        self._update_generic(y, h, H, R_adaptive_ori)

    def update_uwb_distance(self, uwb_distance, p_shoulder_world, p_pelvis_world, R_pelvis_world):
        h = self.h_forward_kinematics(p_shoulder_world, p_pelvis_world, R_pelvis_world)
        y = torch.tensor([[uwb_distance]], device=self.device, dtype=self.dtype)
        H = self._H_jacobian_numerical_uwb(p_shoulder_world, p_pelvis_world, R_pelvis_world)
        innovation = y - h

        if torch.abs(innovation).item() > 0.5:
            self.x[3:6] *= 0.1
            return

        R_adaptive = self.R_uwb.clone()
        innovation_abs = torch.abs(innovation).item()
        if innovation_abs > 0.03: 
            R_adaptive *= (innovation_abs / 0.03)**2

        S = H @ self.P @ H.T + R_adaptive
        nis = innovation.T @ torch.linalg.pinv(S) @ innovation
        if nis.item() > 3.84: return
        self._update_generic(y, h, H, R_adaptive)

    def h_forward_kinematics(self, p_shoulder, p_pelvis, R_pelvis, angles=None, bias=None):
        if angles is None: angles = self.x[0:3].squeeze()
        if bias is None: bias = self.x[6]
        
        sh_pitch, sh_roll, el_pitch = angles[0], angles[1], angles[2]
        
        R_sh_roll = torch.tensor(R.from_euler('y', sh_roll.item()).as_matrix(), device=self.device, dtype=self.dtype)
        R_sh_pitch = torch.tensor(R.from_euler('x', sh_pitch.item()).as_matrix(), device=self.device, dtype=self.dtype)
        R_el_pitch = torch.tensor(R.from_euler('x', el_pitch.item()).as_matrix(), device=self.device, dtype=self.dtype)

        R_shoulder = R_sh_roll @ R_sh_pitch
        v_upper_arm_local = torch.tensor([0.0, -self.l1, 0.0], device=self.device, dtype=self.dtype)
        p_elbow = p_shoulder + (R_pelvis @ R_shoulder @ v_upper_arm_local)
        
        R_elbow_world = R_pelvis @ R_shoulder @ R_el_pitch
        v_forearm_local = torch.tensor([0.0, -self.l2, 0.0], device=self.device, dtype=self.dtype)
        p_wrist_joint = p_elbow + (R_elbow_world @ v_forearm_local)
        
        p_wrist_sensor = p_wrist_joint + (R_elbow_world @ self.wrist_offset)
        p_pelvis_sensor = p_pelvis + (R_pelvis @ self.pelvis_offset)
        expected_dist = torch.linalg.norm(p_wrist_sensor - p_pelvis_sensor) + bias
        return expected_dist.view(1, 1)

    def _H_jacobian_numerical_uwb(self, p_shoulder, p_pelvis, R_pelvis, epsilon=1e-5):
        H = torch.zeros(1, self.state_dim, device=self.device, dtype=self.dtype)
        for i in range(3):
            angles_plus = self.x[0:3].clone(); angles_plus[i] += epsilon
            h_plus = self.h_forward_kinematics(p_shoulder, p_pelvis, R_pelvis, angles=angles_plus)
            angles_minus = self.x[0:3].clone(); angles_minus[i] -= epsilon
            h_minus = self.h_forward_kinematics(p_shoulder, p_pelvis, R_pelvis, angles=angles_minus)
            H[0, i] = (h_plus - h_minus) / (2 * epsilon)
        H[0, 6] = 1.0
        return H

    def update_zero_angular_velocity_prior(self):
        y = torch.zeros(3, 1, device=self.device, dtype=self.dtype)
        h = self.x[3:6]
        H = torch.zeros((3, self.state_dim), device=self.device, dtype=self.dtype); H[:, 3:6] = torch.eye(3)
        
        vel_std_dev_pitch = np.deg2rad(50.0)
        vel_std_dev_roll = np.deg2rad(15.0)
        R_zero_vel = torch.diag(torch.tensor([
            vel_std_dev_pitch**2, vel_std_dev_roll**2, vel_std_dev_pitch**2
        ], device=self.device, dtype=self.dtype))
        
        self._update_generic(y, h, H, R_zero_vel)

    def update_zero_angular_velocity(self):
        y = torch.zeros(3, 1, device=self.device, dtype=self.dtype)
        h = self.x[3:6]
        H = torch.zeros((3, self.state_dim), device=self.device, dtype=self.dtype); H[:, 3:6] = torch.eye(3)
        R_zavu = torch.eye(3, device=self.device, dtype=self.dtype) * (np.deg2rad(0.1)**2)
        self._update_generic(y, h, H, R_zavu)

    def get_state_angles(self):
        return self.x[0:3].clone().squeeze()