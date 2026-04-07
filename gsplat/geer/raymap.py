import numpy as np
import torch

from .camera_model import CameraModelParameters

# ---------------- Polynomial Evaluation ----------------
def _eval_poly_horner(poly_coefficients: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    """Evaluates a polynomial y=f(x) using Horner's scheme"""
    y = torch.zeros_like(x)
    for fi in torch.flip(poly_coefficients, dims=(0,)):
        y = y * x + fi
    return y

def _eval_poly_inverse_horner_newton(
    poly_coefficients: torch.Tensor,
    poly_derivative_coefficients: torch.Tensor,
    inverse_poly_approximation_coefficients: torch.Tensor,
    newton_iterations: int,
    y: torch.Tensor,
) -> torch.Tensor:
    """Evaluates inverse x=f^{-1}(y) using Horner + Newton iterations"""
    x = _eval_poly_horner(inverse_poly_approximation_coefficients, y)
    assert newton_iterations >= 0
    x_iter = [torch.zeros_like(x) for _ in range(newton_iterations + 1)]
    x_iter[0] = x
    for i in range(newton_iterations):
        dfdx = _eval_poly_horner(poly_derivative_coefficients, x_iter[i])
        residuals = _eval_poly_horner(poly_coefficients, x_iter[i]) - y
        x_iter[i + 1] = x_iter[i] - residuals / dfdx
    return x_iter[newton_iterations]

def image_points_to_camera_rays_kb(
    camera_model_parameters: CameraModelParameters,
    image_points: torch.Tensor,
    newton_iterations: int = 3,
    min_2d_norm: float = 1e-6,
    device: str = "cpu",
):
    dtype: torch.dtype = torch.float32

    principal_point = torch.tensor(camera_model_parameters.principal_point, dtype=dtype, device=device)
    focal_length = torch.tensor(camera_model_parameters.focal_length, dtype=dtype, device=device)
    resolution = torch.tensor(camera_model_parameters.resolution.astype(np.int32), device=device)
    max_angle = float(camera_model_parameters.max_angle)

    min_2d_norm = torch.tensor(min_2d_norm, dtype=dtype, device=device)

    # Radial polynomial
    k1, k2, k3, k4 = camera_model_parameters.radial_coeffs
    forward_poly = torch.tensor([0, 1, 0, k1, 0, k2, 0, k3, 0, k4], dtype=dtype, device=device)
    dforward_poly = torch.tensor([1, 0, 3*k1, 0, 5*k2, 0, 7*k3, 0, 9*k4], dtype=dtype, device=device)

    # Approx backward polynomial (linear approx)
    max_normalized_dist = np.max(camera_model_parameters.resolution / 2 / camera_model_parameters.focal_length)
    approx_backward_poly = torch.tensor([0, max_angle / max_normalized_dist], dtype=dtype, device=device)

    image_points = image_points.to(dtype)
    normalized_image_points = (image_points - principal_point) / focal_length
    deltas = torch.linalg.norm(normalized_image_points, axis=1, keepdims=True)

    thetas = _eval_poly_inverse_horner_newton(
        forward_poly, dforward_poly, approx_backward_poly, newton_iterations, deltas
    )

    cam_rays = torch.cat([
        torch.sin(thetas) * normalized_image_points / torch.clamp(deltas, min=min_2d_norm),
        torch.cos(thetas)
    ], dim=1)
    mask = deltas.flatten() < min_2d_norm
    cam_rays[mask, :] = normalized_image_points.new_tensor([0, 0, 1])

    cam_rays = cam_rays.reshape(resolution[0], resolution[1], 3) # [W,H,3]
    return cam_rays