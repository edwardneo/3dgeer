import numpy as np
import torch
import math

from .camera_model import CameraModelParameters
from .raymap import image_points_to_camera_rays_kb

def unpack_camera_intrinsics(K, fov_mod=1): # one image
    """
    Given a 3x3 camera intrinsic matrix K, extract the focal length and principal point.
    The focal length is scaled by fov_mod to allow for adjusting the field of view.

    Args:
        K: A 3x3 numpy array representing the camera intrinsic matrix.
        fov_mod: A scaling factor for the focal length to adjust the field of view.
    Returns:
        focal_length: A tuple (focal_length_x, focal_length_y) representing the focal length in pixels.
        principal_point: A tuple (principal_point_x, principal_point_y) representing the principal point in pixels.
    """
    focal_length = (K[0, 0] * fov_mod, K[1, 1] * fov_mod)
    principal_point = (K[0, 2], K[1, 2])

    return focal_length, principal_point

def compute_max_distance_to_border(image_size_component: float, principal_point_component: float) -> float:
    """Given an image size component (x or y) and corresponding principal point component (x or y),
    returns the maximum distance (in image domain units) from the principal point to either image boundary."""
    center = 0.5 * image_size_component
    if principal_point_component > center:
        return principal_point_component
    else:
        return image_size_component - principal_point_component


def compute_max_radius(image_size: np.ndarray, principal_point: np.ndarray) -> float:
    """Compute the maximum radius from the principal point to the image boundaries."""
    max_diag = np.array(
        [
            compute_max_distance_to_border(image_size[0], principal_point[0]),
            compute_max_distance_to_border(image_size[1], principal_point[1]),
        ]
    )
    return np.linalg.norm(max_diag).item()

def _tanfov_from_raymap(raymap, min_rz: float = 1e-3, max_tan: float = 1e4):
    """Derive (tanfovx, tanfovy) from the actual ray-direction extents of a raymap.

    Used in KB/EQ mode to replace the ``fov_mod``-based heuristic, which
    systematically under-estimates the FOV and causes the PBF frustum-clipping
    in the CUDA kernel (``computePBF``/``computeAABB_*``) to cull valid
    edge-of-image Gaussians during training.

    Parameters
    ----------
    raymap : numpy.ndarray or torch.Tensor, shape (H, W, 3)
        Per-pixel camera-space ray directions (rx, ry, rz).  Rays are assumed
        to point forward (rz > 0 for in-image pixels).
    min_rz : float
        Pixels whose rz is at or below this threshold are ignored to avoid
        division by zero / near-infinite tangent values (e.g. rays at ≥90°).
    max_tan : float
        Hard upper cap on the returned tangent values (prevents infinities
        from slipping through; ``atan(1e4) ≈ 89.99°``).

    Returns
    -------
    (tanfovx, tanfovy) : (float, float) or (None, None)
        Maximum absolute tangent values in x and y.  Returns ``(None, None)``
        when no valid pixels are found so the caller can keep its default.
    """
    if isinstance(raymap, torch.Tensor):
        arr = raymap.detach().cpu().numpy() if raymap.is_cuda else raymap.numpy()
    else:
        arr = np.asarray(raymap, dtype=np.float32)

    rz = arr[:, :, 2]
    valid = rz > min_rz
    if not valid.any():
        return None, None

    safe_rz = np.where(valid, rz, 1.0)
    tanx = np.abs(arr[:, :, 0]) / safe_rz
    tany = np.abs(arr[:, :, 1]) / safe_rz
    tanfovx = float(np.clip(tanx[valid].max(), 0.0, max_tan))
    tanfovy = float(np.clip(tany[valid].max(), 0.0, max_tan))
    return tanfovx, tanfovy

def focal2fov(focal, pixels):
    return 2*math.atan(pixels/(2*focal))

def focal2fov2(focal, pixels):
    return pixels / focal

def focal2halffov2(focal, pixels):
    return pixels / 2 / focal

def fov_sample2ray(fovx, fovy, interval):
    """Build symmetric 1-D arrays of ray-direction half-angles in [-fov, +fov].

    Each element is spaced `interval` radians apart, starting at interval/2.
    Returns (theta_arr, phi_arr) as sorted float tensors.
    """
    theta_arr = torch.arange(interval / 2, fovx, interval)
    theta_arr, _ = torch.sort(torch.cat((-theta_arr, theta_arr)))
    phi_arr = torch.arange(interval / 2, fovy, interval)
    phi_arr, _ = torch.sort(torch.cat((-phi_arr, phi_arr)))

    return theta_arr.float(), phi_arr.float()

def mirror_transform(m, z, xi=0.0): #1.1
    """Apply the omnidirectional mapping to a tangent array m.

    Mirror transform tan(θ); reference: Appendix D.2.
    """
    return m / (1+xi*(z/(torch.abs(z)))*(1+m**2)**0.5)

def get_camera_tanfov(camera_model, Ks, width, height, step=0.002, fov_mod=1, data_device="cuda"):
    # Ks [..., C, 3, 3]
    K = Ks.to("cpu").squeeze() # one image

    focal_length, principal_point = unpack_camera_intrinsics(K, fov_mod)

    if camera_model == "pinhole":
        return width / (2 * focal_length[0]), height / (2 * focal_length[1]), None, None
    elif camera_model == "fisheye":
        # get image pixel points
        u_grid, v_grid = np.meshgrid(np.arange(width), np.arange(height))
        grid = torch.from_numpy(np.stack([u_grid, v_grid], axis=0)).float()  # [2,H,W]
        grid_flat = grid.reshape(2, -1)  # [2,N]
        points = grid_flat.T  # [N,2]
        image_points = points.float()

        # get raymap
        max_radius_pixels = compute_max_radius(np.array((width, height)).astype(np.float64), np.array(principal_point))
        fov_angle_x = 2.0 * max_radius_pixels / focal_length[0]
        fov_angle_y = 2.0 * max_radius_pixels / focal_length[1]
        max_angle = np.max([fov_angle_x, fov_angle_y]) / 2.0

        camera_model_parameters = CameraModelParameters(
            camera_model=camera_model,
            focal_length=focal_length,
            principal_point=principal_point,
            resolution=(width, height),
            radial_coeffs=[0, 0, 0, 0], # TODO: support KB model
            max_angle=max_angle
        )
        rays = image_points_to_camera_rays_kb(camera_model_parameters, image_points, newton_iterations=3, device="cpu")

        tanfovx, tanfovy = _tanfov_from_raymap(rays)
        return tanfovx, tanfovy, None, None
    
    else: # BEAP (TODO)
        FoVx = focal2fov(focal_length[0], width)
        FoVy = focal2fov(focal_length[1], height)
        arr_theta, arr_phi = fov_sample2ray(FoVx/2, FoVy/2, step)

        cos_theta = torch.cos(arr_theta)
        cos_phi = torch.cos(arr_phi)

        cos_theta = torch.where(torch.abs(cos_theta) < 1e-7, torch.full_like(cos_theta, 1e-7), cos_theta).to(data_device)
        cos_phi = torch.where(torch.abs(cos_phi) < 1e-7, torch.full_like(cos_phi, 1e-7), cos_phi).to(data_device)

        tan_theta = torch.tan(arr_theta).to(data_device)
        tan_phi = torch.tan(arr_phi).to(data_device)

        mirror_transformed_tan_theta = mirror_transform(tan_theta, cos_theta).to(data_device)
        mirror_transformed_tan_phi = mirror_transform(tan_phi, cos_phi).to(data_device)

        tanfovx = np.tan(FoVx * 0.5)
        tanfovy = np.tan(FoVy * 0.5)

        print("Mirror Tan Length", len(mirror_transformed_tan_theta), len(mirror_transformed_tan_phi))
        print("Image Dimensions", width, height)

        return tanfovx, tanfovy, mirror_transformed_tan_theta, mirror_transformed_tan_phi