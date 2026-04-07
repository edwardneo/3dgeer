import numpy as np

class CameraModelParameters:
    def __init__(self, camera_model, focal_length, principal_point, resolution, radial_coeffs, max_angle):
        self.camera_model = camera_model
        self.focal_length = np.array(focal_length, dtype=np.float32)
        self.principal_point = np.array(principal_point, dtype=np.float32)
        self.resolution = np.array(resolution, dtype=np.float32)
        self.radial_coeffs = radial_coeffs
        self.max_angle = max_angle