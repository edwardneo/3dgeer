import argparse
import os
import threading
import time

import torch
import viser
from pathlib import Path
from gsplat.distributed import cli
from gsplat.rendering import rasterization

from nerfview import CameraState, RenderTabState, apply_float_colormap
from gsplat_viewer import GsplatViewer, GsplatRenderTabState

from omegaconf import OmegaConf

from datasets.driving_dataset import DrivingDataset
from models.gaussians.basics import dataclass_camera, dataclass_gs
from utils.misc import import_str


def _default_config_path_from_ckpt(ckpt_path: str) -> str:
    ckpt_dir = os.path.dirname(os.path.abspath(ckpt_path))
    return os.path.join(ckpt_dir, "config.yaml")


def _set_frame_on_trainer(trainer, frame_idx: int) -> None:
    if hasattr(trainer, "cur_frame") and isinstance(trainer.cur_frame, torch.Tensor):
        trainer.cur_frame[...] = int(frame_idx)
    else:
        trainer.cur_frame = torch.tensor(int(frame_idx), device=trainer.device)

    for model in getattr(trainer, "models", {}).values():
        set_cur_frame = getattr(model, "set_cur_frame", None)
        if callable(set_cur_frame):
            set_cur_frame(int(frame_idx))
        elif hasattr(model, "cur_frame"):
            setattr(model, "cur_frame", int(frame_idx))


def _merge_gaussians(gs_list):
    if not gs_list:
        return None
    keys = ("_means", "_scales", "_quats", "_rgbs", "_opacities")
    merged = {k: torch.cat([g[k] for g in gs_list], dim=0) for k in keys}
    return dataclass_gs(
        _means=merged["_means"],
        _scales=merged["_scales"],
        _quats=merged["_quats"],
        _rgbs=merged["_rgbs"],
        _opacities=merged["_opacities"],
        detach_keys=[],
        extras=None,
    )


def main(local_rank: int, world_rank, world_size: int, args):
    torch.manual_seed(42)
    device = torch.device("cuda", local_rank) if torch.cuda.is_available() else torch.device("cpu")

    ckpt_path = os.path.abspath(args.ckpt)
    if not os.path.isfile(ckpt_path):
        raise FileNotFoundError(ckpt_path)

    config_path = os.path.abspath(args.config or _default_config_path_from_ckpt(ckpt_path))
    if not os.path.isfile(config_path):
        raise FileNotFoundError(
            f"Could not find config.yaml at {config_path}. Pass --config explicitly."
        )

    cfg = OmegaConf.load(config_path)

    dataset = DrivingDataset(data_cfg=cfg.data)
    num_frames = int(dataset.num_img_timesteps)
    if num_frames <= 0:
        raise ValueError(f"Dataset reports num_img_timesteps={num_frames}")

    trainer = import_str(cfg.trainer.type)(
        **cfg.trainer,
        num_timesteps=dataset.num_img_timesteps,
        model_config=cfg.model,
        num_train_images=len(dataset.train_image_set),
        num_full_images=len(dataset.full_image_set),
        test_set_indices=dataset.test_timesteps,
        scene_aabb=dataset.get_aabb().reshape(2, 3),
        device=device,
    )
    trainer.resume_from_checkpoint(ckpt_path=ckpt_path, load_only_model=True)
    trainer.set_eval()
    _set_frame_on_trainer(trainer, 0)
    current_frame = 0
    playing = False
    fps = 10.0
    loop = True
    state_lock = threading.Lock()

    # register and open viewer
    @torch.no_grad()
    def viewer_render_fn(camera_state: CameraState, render_tab_state: RenderTabState):
        assert isinstance(render_tab_state, GsplatRenderTabState)
        with state_lock:
            frame_idx = int(current_frame)
        _set_frame_on_trainer(trainer, frame_idx)

        if render_tab_state.preview_render:
            width = render_tab_state.render_width
            height = render_tab_state.render_height
        else:
            width = render_tab_state.viewer_width
            height = render_tab_state.viewer_height
        c2w = camera_state.c2w
        K = camera_state.get_K((width, height))
        c2w = torch.from_numpy(c2w).float().to(device)
        K = torch.from_numpy(K).float().to(device)
        viewmat = c2w.inverse()

        RENDER_MODE_MAP = {
            "rgb": "RGB",
            "depth(accumulated)": "D",
            "depth(expected)": "ED",
            "alpha": "RGB",
        }

        rendering_mode = getattr(render_tab_state, "rendering_mode", "standard")
        with_ut = rendering_mode == "gut + eval3d"
        with_geer = rendering_mode == "geer + eval3d"
        with_eval3d = with_ut or with_geer
        camera_model = render_tab_state.camera_model
        is_fisheye = camera_model == "fisheye"

        # Only pass distortion coefficients when UT or GEER is selected.
        if with_ut or with_geer:
            # - pinhole/ortho: 6 radial + 2 tangential + 2 thin prism (pad s3/s4 with 0)
            # - fisheye: 4 radial only
            if is_fisheye:
                radial_coeffs = torch.tensor(
                    [
                        float(getattr(render_tab_state, "radial_k1", 0.0)),
                        float(getattr(render_tab_state, "radial_k2", 0.0)),
                        float(getattr(render_tab_state, "radial_k3", 0.0)),
                        float(getattr(render_tab_state, "radial_k4", 0.0)),
                    ],
                    device=device,
                )
                tangential_coeffs = None
                thin_prism_coeffs = None
            else:
                radial_coeffs = torch.tensor(
                    [
                        float(getattr(render_tab_state, "radial_k1", 0.0)),
                        float(getattr(render_tab_state, "radial_k2", 0.0)),
                        float(getattr(render_tab_state, "radial_k3", 0.0)),
                        float(getattr(render_tab_state, "radial_k4", 0.0)),
                        float(getattr(render_tab_state, "radial_k5", 0.0)),
                        float(getattr(render_tab_state, "radial_k6", 0.0)),
                    ],
                    device=device,
                )
                tangential_coeffs = torch.tensor(
                    [
                        float(getattr(render_tab_state, "tangential_p1", 0.0)),
                        float(getattr(render_tab_state, "tangential_p2", 0.0)),
                    ],
                    device=device,
                )
                thin_prism_coeffs = torch.tensor(
                    [
                        float(getattr(render_tab_state, "thin_prism_s1", 0.0)),
                        float(getattr(render_tab_state, "thin_prism_s2", 0.0)),
                        0.0,
                        0.0,
                    ],
                    device=device,
                )
        else:
            radial_coeffs = None
            tangential_coeffs = None
            thin_prism_coeffs = None

        cam = dataclass_camera(
            camtoworlds=c2w,
            camtoworlds_gt=c2w,
            Ks=K,
            H=int(height),
            W=int(width),
        )

        gs_per_class = []
        for class_name in getattr(trainer, "gaussian_classes", {}).keys():
            model = trainer.models.get(class_name, None)
            if model is None:
                continue
            gs = model.get_gaussians(cam)
            if gs is None:
                continue
            gs_per_class.append(gs)

        merged = _merge_gaussians(gs_per_class)
        if merged is None:
            render_tab_state.total_gs_count = 0
            render_tab_state.rendered_gs_count = 0
            return (torch.zeros((height, width, 3), device="cpu")).numpy()

        means = merged.means
        quats = merged.quats
        scales = merged.scales
        opacities = merged.opacities.squeeze(-1)
        colors = merged.rgbs

        render_colors, render_alphas, info = rasterization(
            means,  # [N, 3]
            quats,  # [N, 4]
            scales,  # [N, 3]
            opacities,  # [N]
            colors,  # [N, 3]
            viewmat[None],  # [1, 4, 4]
            K[None],  # [1, 3, 3]
            width,
            height,
            near_plane=render_tab_state.near_plane,
            far_plane=render_tab_state.far_plane,
            radius_clip=render_tab_state.radius_clip,
            eps2d=render_tab_state.eps2d,
            backgrounds=torch.tensor([render_tab_state.backgrounds], device=device)
            / 255.0,
            render_mode=RENDER_MODE_MAP[render_tab_state.render_mode],
            rasterize_mode=render_tab_state.rasterize_mode,
            camera_model=camera_model,
            packed=False,
            with_ut=with_ut,
            with_geer=with_geer,
            with_eval3d=with_eval3d,
            radial_coeffs=(
                radial_coeffs[None, ...] if (with_eval3d and radial_coeffs is not None) else None
            ),
            tangential_coeffs=(
                tangential_coeffs[None, ...]
                if (with_eval3d and tangential_coeffs is not None)
                else None
            ),
            thin_prism_coeffs=(
                thin_prism_coeffs[None, ...]
                if (with_eval3d and thin_prism_coeffs is not None)
                else None
            ),
        )
        render_tab_state.total_gs_count = int(means.shape[0])
        radii = info.get("radii", None)
        if radii is None:
            render_tab_state.rendered_gs_count = 0
        else:
            rendered_mask = (radii > 0).all(-1) if radii.ndim > 1 else (radii > 0)
            render_tab_state.rendered_gs_count = int(rendered_mask.sum().item())

        if render_tab_state.render_mode == "rgb":
            # colors represented with sh are not guranteed to be in [0, 1]
            render_colors = render_colors[0, ..., 0:3].clamp(0, 1)
            renders = render_colors.cpu().numpy()
        elif render_tab_state.render_mode in ["depth(accumulated)", "depth(expected)"]:
            # normalize depth to [0, 1]
            depth = render_colors[0, ..., 0:1]
            if render_tab_state.normalize_nearfar:
                near_plane = render_tab_state.near_plane
                far_plane = render_tab_state.far_plane
            else:
                near_plane = depth.min()
                far_plane = depth.max()
            depth_norm = (depth - near_plane) / (far_plane - near_plane + 1e-10)
            depth_norm = torch.clip(depth_norm, 0, 1)
            if render_tab_state.inverse:
                depth_norm = 1 - depth_norm
            renders = (
                apply_float_colormap(depth_norm, render_tab_state.colormap)
                .cpu()
                .numpy()
            )
        elif render_tab_state.render_mode == "alpha":
            alpha = render_alphas[0, ..., 0:1]
            renders = (
                apply_float_colormap(alpha, render_tab_state.colormap).cpu().numpy()
            )
        return renders

    server = viser.ViserServer(port=args.port, verbose=False)
    with server.gui.add_folder("Time"):
        time_slider = server.gui.add_slider(
            "Frame",
            min=0,
            max=max(0, num_frames - 1),
            step=1,
            initial_value=0,
            hint="Scrub through dataset timesteps (0-indexed).",
        )

        if hasattr(server.gui, "add_button"):
            play_pause_btn = server.gui.add_button("Play / Pause")
            play_checkbox = None
        else:
            play_pause_btn = None
            play_checkbox = server.gui.add_checkbox("Playing", initial_value=False)

        fps_slider = server.gui.add_slider(
            "FPS",
            min=0.5,
            max=60.0,
            step=0.5,
            initial_value=fps,
            hint="Playback speed in frames per second.",
        )
        loop_checkbox = server.gui.add_checkbox("Loop", initial_value=True)

    viewer = GsplatViewer(
        server=server,
        render_fn=viewer_render_fn,
        output_dir=Path(args.output_dir),
        mode="rendering",
    )

    @time_slider.on_update
    def _(_event: viser.GuiEvent) -> None:
        nonlocal current_frame  # type: ignore[misc]
        with state_lock:
            current_frame = int(time_slider.value)
        viewer.rerender(_event)

    @fps_slider.on_update
    def _(_event: viser.GuiEvent) -> None:
        nonlocal fps  # type: ignore[misc]
        with state_lock:
            fps = float(fps_slider.value)

    @loop_checkbox.on_update
    def _(_event: viser.GuiEvent) -> None:
        nonlocal loop  # type: ignore[misc]
        with state_lock:
            loop = bool(loop_checkbox.value)

    if play_pause_btn is not None:
        on_click = getattr(play_pause_btn, "on_click", None)
        if callable(on_click):

            @play_pause_btn.on_click
            def _(_event: viser.GuiEvent) -> None:
                nonlocal playing  # type: ignore[misc]
                with state_lock:
                    playing = not playing
        else:

            @play_pause_btn.on_update  # type: ignore[attr-defined]
            def _(_event: viser.GuiEvent) -> None:
                nonlocal playing  # type: ignore[misc]
                with state_lock:
                    playing = not playing
    else:

        @play_checkbox.on_update  # type: ignore[union-attr]
        def _(_event: viser.GuiEvent) -> None:
            nonlocal playing  # type: ignore[misc]
            with state_lock:
                playing = bool(play_checkbox.value)  # type: ignore[union-attr]

    def playback_loop() -> None:
        nonlocal current_frame  # type: ignore[misc]
        nonlocal playing  # type: ignore[misc]
        last = time.time()
        carry = 0.0
        while True:
            now = time.time()
            dt = now - last
            last = now

            with state_lock:
                _playing = bool(playing)
                _fps = float(max(0.0, fps))
                _loop = bool(loop)
                _frame = int(current_frame)

            if not _playing or _fps <= 0.0:
                time.sleep(0.01)
                continue

            carry += dt * _fps
            step = int(carry)
            if step <= 0:
                time.sleep(0.005)
                continue
            carry -= float(step)

            nxt = _frame + step
            if nxt >= num_frames:
                if _loop:
                    nxt = nxt % num_frames
                else:
                    nxt = num_frames - 1
                    with state_lock:
                        playing = False
                    if play_checkbox is not None:
                        play_checkbox.value = False

            with state_lock:
                current_frame = int(nxt)
            time_slider.value = int(nxt)
            viewer.rerender(None)
            time.sleep(0.001)

    threading.Thread(target=playback_loop, daemon=True).start()

    print("Viewer running... Ctrl+C to exit.")
    time.sleep(100000)


if __name__ == "__main__":
    """
    # Use single GPU to view the scene
    CUDA_VISIBLE_DEVICES=0 python tools/simple_viewer.py \\
        --ckpt results/kitti/exp01/checkpoint_final.pth \\
        --port 8082
    """
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output_dir", type=str, default="results/", help="where to dump outputs"
    )
    parser.add_argument(
        "--ckpt", type=str, required=True, help="path to the DriveStudio .pth checkpoint"
    )
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="path to config.yaml (defaults to <ckpt_dir>/config.yaml)",
    )
    parser.add_argument(
        "--port", type=int, default=8080, help="port for the viewer server"
    )
    parser.add_argument(
        "--with_ut", action="store_true", help="use uncentered transform"
    )
    parser.add_argument(
        "--with_geer", action="store_true", help="use 3dgeer"
    )
    parser.add_argument("--with_eval3d", action="store_true", help="use eval 3D")
    args = parser.parse_args()
    assert not (args.with_ut and args.with_geer), "cannot render with ut and geer"
    assert not (args.with_geer and not args.with_eval3d), "cannot render geer without eval 3d"

    cli(main, args, verbose=True)
