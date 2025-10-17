import torch
from scipy.spatial.transform import Rotation as R
import numpy as np

def wrap_angle(angle):
    return (angle + torch.pi) % (2 * torch.pi) - torch.pi

class EKFJointAngle:
    def __init__(self, shoulder_pelvis_offset, bone_lengths, device='cpu', dtype=torch.float32, dt=0.01):
        self.device = device; self.dtype = dtype; self.dt = dt
        self.l1 = bone_lengths[0]; self.l2 = bone_lengths[1]

        self.shoulder_pelvis_offset = shoulder_pelvis_offset.to(device=device, dtype=dtype)

        self.state_dim = 7
        self.x = torch.zeros(self.state_dim, 1, device=device, dtype=dtype)
        self.P = torch.eye(self.state_dim, device=device, dtype=dtype) * 0.1

        # --- TUNING PARAMETERS ---
        q_pitch_el_pos = 1e-4
        q_pitch_el_vel = 7e-3
        q_roll_pos = 1e-4
        q_roll_vel = 5e-4
        q_uwb_bias = 1e-5
        uwb_std_dev = 0.2
        ori_std_dev = np.deg2rad(5.0)
        
        self.wrist_offset = torch.tensor([0.0, 0.0, 0.10], device=device, dtype=dtype)
        self.pelvis_offset = torch.tensor([0.10, 0.0, 0.0], device=device, dtype=dtype)

        self.R_sh_to_el_offset = torch.eye(3, device=device, dtype=dtype)

        self.Q = torch.diag(torch.tensor([
            q_roll_pos, q_pitch_el_pos, q_pitch_el_pos, # Positional variances
            q_roll_vel, q_pitch_el_vel, q_pitch_el_vel, # Velocity variances
            q_uwb_bias], device=device, dtype=dtype)) # UWB bias variance
        
        self.R_uwb = torch.eye(1, device=device, dtype=dtype) * (uwb_std_dev**2) # UWB measurement noise
        self.R_damping = torch.eye(3, device=device, dtype=dtype) * (2.0**2) # Velocity damping measurement noise
        self.R_ori = torch.eye(3, device=device, dtype=dtype) * (ori_std_dev**2) # Orientation measurement noise

    def set_initial_state(self, initial_angles):
        self.x[0:3, 0] = initial_angles.squeeze(); self.x[6, 0] = 0.0 # UWB bias initial guess

    def predict(self):
        dt = self.dt # Time step

        # State transition matrix
        F = torch.eye(self.state_dim, device=self.device, dtype=self.dtype)
        F[0, 3] = dt # Roll angle updated by roll rate
        F[1, 4] = dt # Pitch angle updated by pitch rate
        F[2, 5] = dt # Elbow angle updated by elbow rate
        self.x = F @ self.x # State prediction
        self.P = F @ self.P @ F.T + self.Q # Covariance prediction
        self.x[0:3] = wrap_angle(self.x[0:3]) # Wrap angles to [-pi, pi]

    def _update_generic(self, y, h, H, R):
        innovation = y - h # Innovation
        S = H @ self.P @ H.T + R # Innovation covariance
        K = self.P @ H.T @ torch.linalg.pinv(S) # Kalman gain
        self.x = self.x + K @ innovation # State update
        I = torch.eye(self.state_dim, device=self.device, dtype=self.dtype) # Identity matrix
        self.P = (I - K @ H) @ self.P @ (I - K @ H).T + K @ R @ K.T # Covariance update
        self.P = 0.5 * (self.P + self.P.T) # Ensure symmetry

    def update_velocity_damping(self):
        y = torch.zeros(3, 1, device=self.device, dtype=self.dtype) # Measurement vector
        h = self.x[3:6] # Predicted velocity
        H = torch.zeros((3, self.state_dim), device=self.device, dtype=self.dtype); H[:, 3:6] = torch.eye(3) # Measurement matrix
        self._update_generic(y, h, H, self.R_damping) # Update step

    def expected_relative_orientation(self, angles=None):
        """
        Compute expected relative orientation R_w_from_pelvis based on current state angles.
        """
        # Extract angles from state vector if not provided
        if angles is None:
            angles = self.x[0:3].squeeze()
        sh_abduction, sh_flexion, el_flexion = angles[0], angles[1], angles[2]

        R_sh_from_pelvis_mat = R.from_euler('zyx', [sh_abduction.item(), sh_flexion.item(), 0]).as_matrix() # Shoulder rotation
        R_sh_from_pelvis = torch.tensor(R_sh_from_pelvis_mat, device=self.device, dtype=self.dtype) # Convert to tensor

        R_el_from_sh_mat = R.from_euler('zyx', [0, el_flexion.item(), 0]).as_matrix() # Elbow rotation
        R_el_from_sh = torch.tensor(R_el_from_sh_mat, device=self.device, dtype=self.dtype) # Convert to tensor

        R_exp_relative = R_sh_from_pelvis @ R_el_from_sh # Expected relative orientation

        return R_exp_relative
    
    def update_orientation(self, R_w_world, R_p_world):
        '''
        Update orientation measurement.
        '''
        R_meas_relative = R_p_world.T @ R_w_world # Measured relative orientation
        R_exp_relative = self.expected_relative_orientation() # Expected relative orientation from state
        R_error = R_exp_relative.T @ R_meas_relative # Rotation error
        error_aa = torch.tensor(R.from_matrix(R_error.cpu().numpy()).as_rotvec(), device=self.device, dtype=self.dtype).view(3, 1) # Rotation vector error
        
        R_adaptive_ori = self.R_ori.clone() # Start with base measurement noise
        error_angle_rad = torch.linalg.norm(error_aa) # Magnitude of rotation error
        # Adapt measurement noise if error is large
        if error_angle_rad > np.deg2rad(15.0): 
            scale = (error_angle_rad / np.deg2rad(15.0))**2 # Scale factor
            R_adaptive_ori *= scale

        # Prepare measurement update
        y = error_aa; h = torch.zeros(3, 1, device=self.device, dtype=self.dtype) # Predicted measurement is zero error
        H = torch.zeros((3, self.state_dim), device=self.device, dtype=self.dtype) # Measurement matrix
        epsilon = 1e-5

        # Numerical Jacobian computation
        for i in range(3): 
            angles_plus = self.x[0:3].clone(); angles_plus[i] += epsilon # Perturb angle i positively
            R_exp_plus = self.expected_relative_orientation(angles=angles_plus) # Expected orientation with perturbed angle
            error_aa_plus = torch.tensor(R.from_matrix((R_exp_plus.T @ R_meas_relative).cpu().numpy()).as_rotvec(), device=self.device, dtype=self.dtype) # Rotation vector error for positive perturbation
            angles_minus = self.x[0:3].clone(); angles_minus[i] -= epsilon # Perturb angle i negatively
            R_exp_minus = self.expected_relative_orientation(angles=angles_minus) # Expected orientation with perturbed angle
            error_aa_minus = torch.tensor(R.from_matrix((R_exp_minus.T @ R_meas_relative).cpu().numpy()).as_rotvec(), device=self.device, dtype=self.dtype) # Rotation vector error for negative perturbation
            H[:, i] = (error_aa_plus - error_aa_minus).squeeze() / (2 * epsilon) # Numerical derivative
            
        self._update_generic(y, h, H, R_adaptive_ori) # Perform update

    def update_uwb_distance(self, uwb_distance, R_pelvis_world): 
        '''
        Update UWB distance measurement.
        '''
        h = self.h_forward_kinematics(R_pelvis_world) # Predicted UWB distance
        y = torch.tensor([[uwb_distance]], device=self.device, dtype=self.dtype) # Measurement vector
        H = self._H_jacobian_numerical_uwb(R_pelvis_world) # Measurement matrix
        innovation = y - h # Innovation

        # Reject update if innovation is too large
        if torch.abs(innovation).item() > 0.5:
            self.x[3:6] *= 0.1
            return

        # Adaptive measurement noise based on innovation magnitude
        R_adaptive = self.R_uwb.clone()
        innovation_abs = torch.abs(innovation).item()
        if innovation_abs > 0.05: 
            R_adaptive *= (innovation_abs / 0.05)**2

        # NIS test to potentially reject outlier
        S = H @ self.P @ H.T + R_adaptive # Innovation covariance
        nis = innovation.T @ torch.linalg.pinv(S) @ innovation # NIS calculation
        if nis.item() > 3.84: return # Reject if NIS exceeds chi-squared threshold
        self._update_generic(y, h, H, R_adaptive) # Perform update

    def h_forward_kinematics(self, R_pelvis, angles=None, bias=None):
        """
        Compute expected UWB distance measurement based on current state angles and pelvis orientation.
        """
        if angles is None: angles = self.x[0:3].squeeze() # Extract angles from state vector
        if bias is None: bias = self.x[6] # UWB bias from state vector

        sh_abduction, sh_flexion, el_flexion = angles[0], angles[1], angles[2] # Unpack angles

        # 1. Calculate pelvis position in world frame
        p_shoulder = R_pelvis @ self.shoulder_pelvis_offset

        # 2. Calculate joint rotations from angles
        R_shoulder_mat = R.from_euler('zyx', [sh_abduction.item(), sh_flexion.item(), 0]).as_matrix() # Shoulder rotation matrix
        R_shoulder_from_pelvis = torch.tensor(R_shoulder_mat, device=self.device, dtype=self.dtype) 

        R_el_flexion_mat = R.from_euler('zyx', [0, el_flexion.item(), 0]).as_matrix() # Elbow rotation matrix
        R_el_flexion = torch.tensor(R_el_flexion_mat, device=self.device, dtype=self.dtype)

        # 3. Calculate world-frame positions of elbow and wrist joints
        R_shoulder_world = R_pelvis @ R_shoulder_from_pelvis

        # 3. Calculate world-frame positions of elbow and wrist joints
        R_elbow_world = R_shoulder_world @ R_el_flexion 

        # Vectors from joints to next segments in local frames
        v_upper_arm_local = torch.tensor([self.l1, 0.0, 0.0], device=self.device, dtype=self.dtype)
        v_forearm_local = torch.tensor([self.l2, 0.0, 0.0], device=self.device, dtype=self.dtype)
        
        # Elbow position in world frame
        p_elbow = p_shoulder + (R_shoulder_world @ v_upper_arm_local)
        # Wrist position in world frame
        p_wrist_joint = p_elbow + (R_elbow_world @ v_forearm_local)

        # 4. Calculate the final world-frame positions of the two UWB sensors
        p_wrist_sensor = p_wrist_joint + (R_elbow_world @ self.wrist_offset) # Wrist sensor relative to wrist joint
        p_pelvis_sensor = R_pelvis @ self.pelvis_offset # Pelvis sensor relative to pelvis joint

        # 5. The expected distance is the norm of the difference vector
        expected_dist = torch.linalg.norm(p_wrist_sensor - p_pelvis_sensor) + bias # Add UWB bias
        return expected_dist.view(1, 1)

    def _H_jacobian_numerical_uwb(self, R_pelvis, epsilon=1e-5): 
        H = torch.zeros(1, self.state_dim, device=self.device, dtype=self.dtype) # Initialize Jacobian matrix

        # Numerical Jacobian computation
        for i in range(3):
            angles_plus = self.x[0:3].clone(); angles_plus[i] += epsilon # Perturb angle i positively
            h_plus = self.h_forward_kinematics(R_pelvis, angles=angles_plus) # Predicted measurement with perturbed angle
            angles_minus = self.x[0:3].clone(); angles_minus[i] -= epsilon # Perturb angle i negatively
            h_minus = self.h_forward_kinematics(R_pelvis, angles=angles_minus) # Predicted measurement with perturbed angle
            H[0, i] = (h_plus - h_minus) / (2 * epsilon) # Numerical derivative

        H[0, 6] = 1.0 # Derivative w.r.t. UWB bias
        return H

    def update_zero_angular_velocity_prior(self):
        y = torch.zeros(3, 1, device=self.device, dtype=self.dtype) # Measurement vector
        h = self.x[3:6] # Predicted angular velocity
        H = torch.zeros((3, self.state_dim), device=self.device, dtype=self.dtype); H[:, 3:6] = torch.eye(3) # Measurement matrix
        
        vel_std_dev_pitch = np.deg2rad(50.0) # Pitch velocity standard deviation
        vel_std_dev_roll = np.deg2rad(15.0) # Roll velocity standard deviation

        # Measurement noise covariance
        R_zero_vel = torch.diag(torch.tensor([
            vel_std_dev_pitch**2, vel_std_dev_roll**2, vel_std_dev_pitch**2
        ], device=self.device, dtype=self.dtype))
        
        self._update_generic(y, h, H, R_zero_vel) # Perform update

    def update_zero_angular_velocity(self):
        y = torch.zeros(3, 1, device=self.device, dtype=self.dtype) # Measurement vector
        h = self.x[3:6] # Predicted angular velocity
        H = torch.zeros((3, self.state_dim), device=self.device, dtype=self.dtype); H[:, 3:6] = torch.eye(3) # Measurement matrix
        R_zavu = torch.eye(3, device=self.device, dtype=self.dtype) * (np.deg2rad(0.1)**2) # Very tight measurement noise
        self._update_generic(y, h, H, R_zavu) # Perform update

    def get_state_angles(self):
        return self.x[0:3].clone().squeeze() # Return current joint angles