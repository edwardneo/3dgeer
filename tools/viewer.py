import argparse
import json
import os
import threading
import time
from typing import Callable, Literal, Optional, Tuple, Union

import imageio.v2 as imageio
import numpy as np
import torch
import viser
import viser.transforms as tf
from pathlib import Path
from gsplat.distributed import cli
from gsplat.rendering import rasterization
from gsplat.cuda._wrapper import compute_raymap

from nerfview import (
    CameraState,
    RenderTabState,
    Viewer,
    apply_float_colormap,
    populate_general_render_tab,
)
from nerfview.render_panel import Keyframe

from omegaconf import OmegaConf

from datasets.base.pixel_source import get_rays
from datasets.driving_dataset import DrivingDataset
from models.gaussians.basics import dataclass_camera, dataclass_gs
from utils.misc import import_str


_NERFVIEW_CAMERA_PATH_SCALE_RATIO = 10.0


class GsplatRenderTabState(RenderTabState):
    total_gs_count: int = 0
    rendered_gs_count: int = 0

    max_sh_degree: int = 5
    near_plane: float = 1e-2
    far_plane: float = 1e2
    radius_clip: float = 0.0
    eps2d: float = 0.3
    background_mode: Literal["color", "sky"] = "sky"
    backgrounds: Tuple[float, float, float] = (0.0, 0.0, 0.0)
    render_mode: Literal[
        "rgb", "depth(accumulated)", "depth(expected)", "alpha"
    ] = "rgb"
    normalize_nearfar: bool = False
    inverse: bool = False
    colormap: Literal[
        "turbo", "viridis", "magma", "inferno", "cividis", "gray"
    ] = "turbo"
    rasterize_mode: Literal["classic", "antialiased"] = "classic"
    camera_model: Literal["pinhole", "ortho", "fisheye"] = "pinhole"
    rendering_mode: Literal["standard", "gut + eval3d", "geer + eval3d"] = "standard"
    radial_k1: float = 0.0
    radial_k2: float = 0.0
    radial_k3: float = 0.0
    radial_k4: float = 0.0
    radial_k5: float = 0.0
    radial_k6: float = 0.0
    tangential_p1: float = 0.0
    tangential_p2: float = 0.0
    thin_prism_s1: float = 0.0
    thin_prism_s2: float = 0.0
    selected_gaussian_types: Tuple[str, ...] = ()


class GsplatViewer(Viewer):
    def __init__(
        self,
        server: viser.ViserServer,
        render_fn: Callable,
        output_dir: Path,
        mode: Literal["rendering", "training"] = "rendering",
        num_frames: int = 1,
        initial_fps: float = 10.0,
        gaussian_types: Tuple[str, ...] = (),
        has_sky: bool = False,
        time_enabled: bool = True,
        initial_render_settings: Optional[dict] = None,
    ):
        self.num_frames = int(num_frames)
        self.initial_fps = float(initial_fps)
        self.gaussian_types = tuple(gaussian_types)
        self.has_sky = bool(has_sky)
        self.time_enabled = bool(time_enabled)
        self.initial_render_settings = dict(initial_render_settings or {})
        super().__init__(server, render_fn, output_dir, mode)
        server.gui.set_panel_label("drivestudio viewer")

    def _init_rendering_tab(self):
        self.render_tab_state = GsplatRenderTabState()
        for attr, value in self.initial_render_settings.items():
            setattr(self.render_tab_state, attr, value)
        if not self.render_tab_state.selected_gaussian_types:
            self.render_tab_state.selected_gaussian_types = self.gaussian_types
        self._rendering_tab_handles = {}
        self._nerfview_render_handles = {}
        self._rendering_folder = None
        self._playback_camera_view_folder = None

    def _populate_rendering_tab(self):
        server = self.server
        mode_status = server.gui.add_markdown(
            "**Mode:** Render. Switching saves render settings.",
            order=-1001.0,
        )
        mode_button = server.gui.add_button(
            "Switch to Export Mode",
            hint="Saves render mode settings.",
            color="blue",
            order=-1000.0,
        )
        self._rendering_tab_handles["mode_status"] = mode_status
        self._rendering_tab_handles["mode_button"] = mode_button

        self._playback_camera_view_folder = server.gui.add_folder(
            "Playback / Camera View"
        )
        with self._playback_camera_view_folder:
            time_slider = server.gui.add_slider(
                "Frame",
                min=0,
                max=max(0, self.num_frames - 1),
                step=1,
                initial_value=0,
                hint="Scrub through dataset timesteps (0-indexed).",
            )

            fps_slider = server.gui.add_number(
                "FPS",
                min=0.5,
                max=60.0,
                step=0.5,
                initial_value=self.initial_fps,
                hint="Playback speed in frames per second.",
            )

            playback_play_button = server.gui.add_button(
                "Play", icon=viser.Icon.PLAYER_PLAY
            )
            playback_pause_button = server.gui.add_button(
                "Pause", icon=viser.Icon.PLAYER_PAUSE, visible=False
            )

        self._gaussians_folder = server.gui.add_folder("Gaussians")

        with self._gaussians_folder:
            total_gs_count_number = server.gui.add_number(
                "Total",
                initial_value=self.render_tab_state.total_gs_count,
                disabled=True,
                hint="Total number of splats in the scene.",
            )
            rendered_gs_count_number = server.gui.add_number(
                "Rendered",
                initial_value=self.render_tab_state.rendered_gs_count,
                disabled=True,
                hint="Number of splats rendered.",
            )

            gaussian_type_checkboxes = {}
            selected_gaussian_types = set(self.render_tab_state.selected_gaussian_types)
            all_gaussians_checkbox = server.gui.add_checkbox(
                "All",
                initial_value=selected_gaussian_types == set(self.gaussian_types),
                hint="Render all Gaussian types.",
            )
            checkbox_sync = {"active": False}

            for gaussian_type in self.gaussian_types:
                gaussian_type_checkboxes[gaussian_type] = server.gui.add_checkbox(
                    gaussian_type,
                    initial_value=gaussian_type in selected_gaussian_types,
                    hint=f"Render {gaussian_type} Gaussians.",
                )

            def _selected_gaussian_types_from_checkboxes() -> Tuple[str, ...]:
                return tuple(
                    gaussian_type
                    for gaussian_type, checkbox in gaussian_type_checkboxes.items()
                    if checkbox.value
                )

            @all_gaussians_checkbox.on_update
            def _(_) -> None:
                if checkbox_sync["active"]:
                    return
                checkbox_sync["active"] = True
                for checkbox in gaussian_type_checkboxes.values():
                    checkbox.value = bool(all_gaussians_checkbox.value)
                checkbox_sync["active"] = False
                self.render_tab_state.selected_gaussian_types = (
                    self.gaussian_types if all_gaussians_checkbox.value else ()
                )
                self.rerender(_)

            for gaussian_type, checkbox in gaussian_type_checkboxes.items():
                @checkbox.on_update
                def _(_, gaussian_type=gaussian_type, checkbox=checkbox) -> None:
                    if checkbox_sync["active"]:
                        return
                    selected = _selected_gaussian_types_from_checkboxes()
                    self.render_tab_state.selected_gaussian_types = selected
                    checkbox_sync["active"] = True
                    all_gaussians_checkbox.value = len(selected) == len(
                        self.gaussian_types
                    )
                    checkbox_sync["active"] = False
                    self.rerender(_)

        self._rendering_folder = server.gui.add_folder("Rendering")
        with self._rendering_folder:
            output_view_dropdown = server.gui.add_dropdown(
                "Output View",
                ("rgb", "depth(accumulated)", "depth(expected)", "alpha"),
                initial_value=self.render_tab_state.render_mode,
                hint="Output to view.",
            )

            @output_view_dropdown.on_update
            def _(_) -> None:
                if "depth" in output_view_dropdown.value:
                    normalize_nearfar_checkbox.disabled = False
                    inverse_checkbox.disabled = False
                else:
                    normalize_nearfar_checkbox.disabled = True
                    inverse_checkbox.disabled = True

                if output_view_dropdown.value == "rgb":
                    colormap_dropdown.disabled = True
                else:
                    colormap_dropdown.disabled = False

                self.render_tab_state.render_mode = output_view_dropdown.value
                self.rerender(_)

            rendering_mode_dropdown = server.gui.add_dropdown(
                "Rendering Mode",
                ("standard", "gut + eval3d", "geer + eval3d"),
                initial_value=self.render_tab_state.rendering_mode,
                hint="Choose standard rendering or distortion-aware modes.",
            )

            @rendering_mode_dropdown.on_update
            def _(_) -> None:
                self.render_tab_state.rendering_mode = rendering_mode_dropdown.value
                _sync_distortion_ui_enabled()
                self.rerender(_)

            use_sky_checkbox = server.gui.add_checkbox(
                "Sky",
                initial_value=self.has_sky
                and self.render_tab_state.background_mode == "sky",
                disabled=not self.has_sky,
                hint="Use the learned sky model as the background.",
            )
            self.render_tab_state.background_mode = (
                "sky" if use_sky_checkbox.value else "color"
            )

            @use_sky_checkbox.on_update
            def _(_) -> None:
                backgrounds_slider.disabled = bool(use_sky_checkbox.value)
                self.render_tab_state.background_mode = (
                    "sky" if use_sky_checkbox.value else "color"
                )
                self.rerender(_)

            backgrounds_slider = server.gui.add_rgb(
                "Background Color",
                initial_value=self.render_tab_state.backgrounds,
                disabled=use_sky_checkbox.value,
                hint="Flat background color used when Background Mode is color.",
            )

            @backgrounds_slider.on_update
            def _(_) -> None:
                self.render_tab_state.backgrounds = backgrounds_slider.value
                self.rerender(_)

            near_far_plane_vec2 = server.gui.add_vector2(
                "Near/Far",
                initial_value=(
                    self.render_tab_state.near_plane,
                    self.render_tab_state.far_plane,
                ),
                min=(1e-3, 1e1),
                max=(1e1, 1e10),
                step=1e-3,
                hint="Near and far plane for rendering.",
            )

            @near_far_plane_vec2.on_update
            def _(_) -> None:
                self.render_tab_state.near_plane = near_far_plane_vec2.value[0]
                self.render_tab_state.far_plane = near_far_plane_vec2.value[1]
                self.rerender(_)

            radius_clip_slider = server.gui.add_number(
                "Radius Clip",
                initial_value=self.render_tab_state.radius_clip,
                min=0.0,
                max=100.0,
                step=1.0,
                hint="2D radius clip for rendering.",
            )

            @radius_clip_slider.on_update
            def _(_) -> None:
                self.render_tab_state.radius_clip = radius_clip_slider.value
                self.rerender(_)

            eps2d_slider = server.gui.add_number(
                "2D Epsilon",
                initial_value=self.render_tab_state.eps2d,
                min=0.0,
                max=1.0,
                step=0.01,
                hint="Epsilon added to the eigenvalues of projected 2D covariance matrices.",
            )

            @eps2d_slider.on_update
            def _(_) -> None:
                self.render_tab_state.eps2d = eps2d_slider.value
                self.rerender(_)

            rasterize_mode_dropdown = server.gui.add_dropdown(
                "Anti-Aliasing",
                ("classic", "antialiased"),
                initial_value=self.render_tab_state.rasterize_mode,
                hint="Whether to use classic or antialiased rasterization.",
            )

            @rasterize_mode_dropdown.on_update
            def _(_) -> None:
                self.render_tab_state.rasterize_mode = rasterize_mode_dropdown.value
                self.rerender(_)

            normalize_nearfar_checkbox = server.gui.add_checkbox(
                "Normalize Near/Far",
                initial_value=self.render_tab_state.normalize_nearfar,
                disabled="depth" not in self.render_tab_state.render_mode,
                hint="Normalize depth with near/far plane.",
            )

            @normalize_nearfar_checkbox.on_update
            def _(_) -> None:
                self.render_tab_state.normalize_nearfar = (
                    normalize_nearfar_checkbox.value
                )
                self.rerender(_)

            inverse_checkbox = server.gui.add_checkbox(
                "Inverse",
                initial_value=self.render_tab_state.inverse,
                disabled="depth" not in self.render_tab_state.render_mode,
                hint="Inverse the depth.",
            )

            @inverse_checkbox.on_update
            def _(_) -> None:
                self.render_tab_state.inverse = inverse_checkbox.value
                self.rerender(_)

            colormap_dropdown = server.gui.add_dropdown(
                "Colormap",
                ("turbo", "viridis", "magma", "inferno", "cividis", "gray"),
                initial_value=self.render_tab_state.colormap,
                disabled=self.render_tab_state.render_mode == "rgb",
                hint="Colormap used for rendering depth/alpha.",
            )

            @colormap_dropdown.on_update
            def _(_) -> None:
                self.render_tab_state.colormap = colormap_dropdown.value
                self.rerender(_)

        self._camera_intrinsics_folder = server.gui.add_folder("Camera Intrinsics")
        with self._camera_intrinsics_folder:
            camera_model_dropdown = server.gui.add_dropdown(
                "Camera Model",
                ("pinhole", "ortho", "fisheye"),
                initial_value=self.render_tab_state.camera_model,
                hint="Camera model used for rendering.",
                order=0.0,
            )

            @camera_model_dropdown.on_update
            def _(_) -> None:
                self.render_tab_state.camera_model = camera_model_dropdown.value
                _sync_distortion_ui_enabled()
                self.rerender(_)

            viewer_res_slider = self.server.gui.add_slider(
                "Viewer Res",
                min=64,
                max=2048,
                step=1,
                initial_value=self.render_tab_state.viewer_res,
                hint="Maximum resolution of the viewer rendered image.",
                order=2.0,
            )

            @viewer_res_slider.on_update
            def _(_) -> None:
                self.render_tab_state.viewer_res = int(viewer_res_slider.value)
                self.rerender(_)

            self._rendering_tab_handles["viewer_res_slider"] = viewer_res_slider

            radial_k1_slider = server.gui.add_slider(
                "Radial k1", min=-1.0, max=1.0, step=0.001, initial_value=0.0
            )
            radial_k2_slider = server.gui.add_slider(
                "Radial k2", min=-1.0, max=1.0, step=0.001, initial_value=0.0
            )
            radial_k3_slider = server.gui.add_slider(
                "Radial k3", min=-1.0, max=1.0, step=0.001, initial_value=0.0
            )
            radial_k4_slider = server.gui.add_slider(
                "Radial k4", min=-1.0, max=1.0, step=0.001, initial_value=0.0
            )
            radial_k5_slider = server.gui.add_slider(
                "Radial k5", min=-1.0, max=1.0, step=0.001, initial_value=0.0
            )
            radial_k6_slider = server.gui.add_slider(
                "Radial k6", min=-1.0, max=1.0, step=0.001, initial_value=0.0
            )
            tangential_p1_slider = server.gui.add_slider(
                "Tangential p1", min=-1.0, max=1.0, step=0.001, initial_value=0.0
            )
            tangential_p2_slider = server.gui.add_slider(
                "Tangential p2", min=-1.0, max=1.0, step=0.001, initial_value=0.0
            )
            thin_prism_s1_slider = server.gui.add_slider(
                "Thin prism s1", min=-1.0, max=1.0, step=0.001, initial_value=0.0
            )
            thin_prism_s2_slider = server.gui.add_slider(
                "Thin prism s2", min=-1.0, max=1.0, step=0.001, initial_value=0.0
            )

            for attr, slider in (
                ("radial_k1", radial_k1_slider),
                ("radial_k2", radial_k2_slider),
                ("radial_k3", radial_k3_slider),
                ("radial_k4", radial_k4_slider),
                ("radial_k5", radial_k5_slider),
                ("radial_k6", radial_k6_slider),
                ("tangential_p1", tangential_p1_slider),
                ("tangential_p2", tangential_p2_slider),
                ("thin_prism_s1", thin_prism_s1_slider),
                ("thin_prism_s2", thin_prism_s2_slider),
            ):

                @slider.on_update
                def _(_, attr=attr, slider=slider) -> None:
                    setattr(self.render_tab_state, attr, float(slider.value))
                    self.rerender(_)

            def _sync_distortion_ui_enabled() -> None:
                sliders = (
                    radial_k1_slider,
                    radial_k2_slider,
                    radial_k3_slider,
                    radial_k4_slider,
                    radial_k5_slider,
                    radial_k6_slider,
                    tangential_p1_slider,
                    tangential_p2_slider,
                    thin_prism_s1_slider,
                    thin_prism_s2_slider,
                )
                fisheye = camera_model_dropdown.value == "fisheye"
                for slider in sliders[:4]:
                    slider.disabled = False
                for slider in sliders[4:]:
                    slider.disabled = fisheye

            _sync_distortion_ui_enabled()

        playback_handles = {
            "time_slider": time_slider,
            "playback_play_button": playback_play_button,
            "playback_pause_button": playback_pause_button,
            "fps_slider": fps_slider,
        }
        rendering_handles = {
            "total_gs_count_number": total_gs_count_number,
            "rendered_gs_count_number": rendered_gs_count_number,
            "all_gaussians_checkbox": all_gaussians_checkbox,
            "gaussian_type_checkboxes": gaussian_type_checkboxes,
            "near_far_plane_vec2": near_far_plane_vec2,
            "radius_clip_slider": radius_clip_slider,
            "eps2d_slider": eps2d_slider,
            "use_sky_checkbox": use_sky_checkbox,
            "backgrounds_slider": backgrounds_slider,
            "render_mode_dropdown": output_view_dropdown,
            "normalize_nearfar_checkbox": normalize_nearfar_checkbox,
            "inverse_checkbox": inverse_checkbox,
            "colormap_dropdown": colormap_dropdown,
            "rasterize_mode_dropdown": rasterize_mode_dropdown,
            "camera_model_dropdown": camera_model_dropdown,
            "distortion_mode_dropdown": rendering_mode_dropdown,
        }
        self._rendering_tab_handles.update(playback_handles)
        self._rendering_tab_handles.update(rendering_handles)

        export_folder = self.server.gui.add_folder("Export")
        self._export_folder = export_folder

        extra_handles = _flatten_gui_handles(self._rendering_tab_handles)
        if self.mode == "training":
            extra_handles.update(_flatten_gui_handles(self._training_tab_handles))
        handles = populate_general_render_tab(
            self.server,
            output_dir=self.output_dir,
            folder=export_folder,
            render_tab_state=self.render_tab_state,
            extra_handles=extra_handles,
            time_enabled=self.time_enabled,
        )
        self._nerfview_render_handles = handles
        self._rendering_tab_handles.update(handles)
        loop_checkbox = handles["loop_checkbox"]
        loop_checkbox.value = False
        loop_checkbox.visible = False
        loop_checkbox._impl.update_cb.clear()
        fov_degrees_slider = handles["fov_degrees_slider"]
        fov_degrees_slider.visible = False
        render_res_vec2 = handles["render_res_vec2"]
        with export_folder:
            export_fov_slider = self.server.gui.add_slider(
                "FOV",
                initial_value=fov_degrees_slider.value,
                min=0.1,
                max=175.0,
                step=0.01,
                hint="Default export FOV, can be overriden per keyframe by clicking on the camera frustum.",
                order=render_res_vec2.order + 0.01,
            )
        with self._camera_intrinsics_folder:
            camera_fov_slider = self.server.gui.add_slider(
                "FOV",
                initial_value=fov_degrees_slider.value,
                min=0.1,
                max=175.0,
                step=0.01,
                hint="Vertical field of view for the viewer camera.",
                order=1.0,
            )

            fov_sync = {"active": False}

            def sync_fov_sliders(value: float, source) -> None:
                if fov_sync["active"]:
                    return
                fov_sync["active"] = True
                try:
                    value = float(value)
                    fov_degrees_slider.value = value
                    if source is not camera_fov_slider:
                        camera_fov_slider.value = value
                    if source is not export_fov_slider:
                        export_fov_slider.value = value
                finally:
                    fov_sync["active"] = False

            @camera_fov_slider.on_update
            def _(_) -> None:
                sync_fov_sliders(camera_fov_slider.value, camera_fov_slider)

        @export_fov_slider.on_update
        def _(_) -> None:
            sync_fov_sliders(export_fov_slider.value, export_fov_slider)

        handles["fov_degrees_slider"] = camera_fov_slider
        handles["export_fov_degrees_slider"] = export_fov_slider
        self._rendering_tab_handles["fov_degrees_slider"] = camera_fov_slider
        self._rendering_tab_handles["export_fov_degrees_slider"] = export_fov_slider

        load_path_button = handles["load_camera_path_button"]
        save_path_button = handles["save_camera_path_button"]
        trajectory_name_text = handles["trajectory_name_text"]
        camera_path = _camera_path_from_callbacks(load_path_button._impl.update_cb)
        assert camera_path is not None, "Unable to find Nerfview CameraPath"
        load_path_button._impl.update_cb.clear()
        save_path_button._impl.update_cb.clear()
        _register_camera_path_loader(
            load_path_button=load_path_button,
            output_dir=self.output_dir,
            camera_path=camera_path,
            handles=handles,
            time_enabled=self.time_enabled,
            scale_ratio=_NERFVIEW_CAMERA_PATH_SCALE_RATIO,
        )
        _register_camera_path_saver(
            save_path_button=save_path_button,
            output_dir=self.output_dir,
            camera_path=camera_path,
            handles=handles,
            scale_ratio=_NERFVIEW_CAMERA_PATH_SCALE_RATIO,
        )

    def _after_render(self):
        self._rendering_tab_handles[
            "total_gs_count_number"
        ].value = self.render_tab_state.total_gs_count
        self._rendering_tab_handles[
            "rendered_gs_count_number"
        ].value = self.render_tab_state.rendered_gs_count


def _register_camera_path_loader(
    load_path_button,
    output_dir: Path,
    camera_path,
    handles: dict,
    time_enabled: bool,
    scale_ratio: float,
) -> None:
    trajectory_name_text = handles["trajectory_name_text"]

    @load_path_button.on_click
    def _(event: viser.GuiEvent) -> None:
        if event.client is None:
            return
        camera_path_dir = output_dir / "camera_paths"
        camera_path_dir.mkdir(parents=True, exist_ok=True)
        camera_path_files = sorted(camera_path_dir.glob("*.json"))
        camera_path_filenames = [path.name for path in camera_path_files]

        with event.client.gui.add_modal("Load Path") as modal:
            if not camera_path_filenames:
                event.client.gui.add_markdown("No existing paths found")
            else:
                event.client.gui.add_markdown("Select existing camera path:")
                camera_path_dropdown = event.client.gui.add_dropdown(
                    label="Camera Path",
                    options=camera_path_filenames,
                    initial_value=camera_path_filenames[0],
                )
                load_button = event.client.gui.add_button("Load")

                @load_button.on_click
                def _(_) -> None:
                    json_path = camera_path_dir / camera_path_dropdown.value
                    with open(json_path, "r") as f:
                        json_data = json.load(f)

                    _load_camera_path_json(
                        json_data=json_data,
                        camera_path=camera_path,
                        handles=handles,
                        time_enabled=time_enabled,
                        scale_ratio=scale_ratio,
                    )
                    trajectory_name_text.value = json_path.stem
                    modal.close()
                    camera_path._server.scene.set_global_visibility(True)

            cancel_button = event.client.gui.add_button("Cancel")

            @cancel_button.on_click
            def _(_) -> None:
                modal.close()


def _register_camera_path_saver(
    save_path_button,
    output_dir: Path,
    camera_path,
    handles: dict,
    scale_ratio: float,
) -> None:
    @save_path_button.on_click
    def _(event: viser.GuiEvent) -> None:
        if event.client is None:
            return
        json_outfile = (
            output_dir
            / "camera_paths"
            / f"{handles['trajectory_name_text'].value}.json"
        )
        json_outfile.parent.mkdir(parents=True, exist_ok=True)
        json_data = _camera_path_to_json(
            camera_path=camera_path,
            handles=handles,
            scale_ratio=scale_ratio,
        )
        with open(json_outfile.absolute(), "w") as outfile:
            json.dump(json_data, outfile)
        print(f"Camera path saved to {json_outfile.absolute()}")


def _camera_path_to_json(camera_path, handles: dict, scale_ratio: float) -> dict:
    fov_degrees_slider = handles["fov_degrees_slider"]
    render_res_vec2 = handles["render_res_vec2"]
    loop_checkbox = handles["loop_checkbox"]
    tension_slider = handles["tension_slider"]
    transition_sec_number = handles["transition_sec_number"]
    framerate_number = handles["framerate_number"]
    duration_number = handles["duration_number"]

    keyframes = []
    for keyframe, _ in camera_path._keyframes.values():
        pose = tf.SE3.from_rotation_and_translation(
            tf.SO3(keyframe.wxyz) @ tf.SO3.from_x_radians(np.pi),
            keyframe.position / scale_ratio,
        )
        frame_number = _keyframe_frame_number(camera_path, keyframe)
        keyframes.append(
            {
                "matrix": pose.as_matrix().flatten().tolist(),
                "fov": (
                    np.rad2deg(keyframe.override_fov_rad)
                    if keyframe.override_fov_enabled
                    else float(fov_degrees_slider.value)
                ),
                "aspect": float(keyframe.aspect),
                "override_transition_enabled": keyframe.override_transition_enabled,
                "override_transition_sec": keyframe.override_transition_sec,
                "override_time_enabled": True,
                "override_time_val": frame_number,
                "render_time": frame_number,
                "frame_number": frame_number,
            }
        )

    num_render_frames = int(
        float(framerate_number.value) * float(duration_number.value)
    )
    camera_path_list = []
    for frame_idx in range(num_render_frames):
        denom = max(1, num_render_frames - 1)
        path_time = float(frame_idx) / float(denom)
        maybe_pose_and_fov = camera_path.interpolate_pose_and_fov_rad(path_time)
        if maybe_pose_and_fov is None:
            break
        if len(maybe_pose_and_fov) == 3:
            pose, fov, frame_number = maybe_pose_and_fov
        else:
            pose, fov = maybe_pose_and_fov
            frame_number = path_time
        frame_number = _clamp_normalized_time(frame_number)
        pose = tf.SE3.from_rotation_and_translation(
            pose.rotation() @ tf.SO3.from_x_radians(np.pi),
            pose.translation() / scale_ratio,
        )
        camera_path_list.append(
            {
                "camera_to_world": pose.as_matrix().flatten().tolist(),
                "fov": np.rad2deg(fov),
                "aspect": float(render_res_vec2.value[0])
                / float(render_res_vec2.value[1]),
                "render_time": frame_number,
                "frame_number": frame_number,
            }
        )

    return {
        "default_fov": float(fov_degrees_slider.value),
        "default_transition_sec": float(transition_sec_number.value),
        "keyframes": keyframes,
        "render_height": float(render_res_vec2.value[1]),
        "render_width": float(render_res_vec2.value[0]),
        "fps": float(framerate_number.value),
        "seconds": float(duration_number.value),
        "is_cycle": bool(loop_checkbox.value),
        "smoothness_value": float(tension_slider.value),
        "camera_path": camera_path_list,
        "trajectory": [
            {
                "pose": frame["camera_to_world"],
                "frame_number": frame["frame_number"],
            }
            for frame in camera_path_list
        ],
    }


def _keyframe_frame_number(camera_path, keyframe) -> float:
    if keyframe.override_time_enabled and keyframe.override_time_val is not None:
        return _clamp_normalized_time(keyframe.override_time_val)
    return _clamp_normalized_time(camera_path.default_render_time)


def _load_camera_path_json(
    json_data: dict,
    camera_path,
    handles: dict,
    time_enabled: bool,
    scale_ratio: float = _NERFVIEW_CAMERA_PATH_SCALE_RATIO,
) -> None:
    fov_degrees_slider = handles["fov_degrees_slider"]
    render_res_vec2 = handles["render_res_vec2"]
    loop_checkbox = handles["loop_checkbox"]
    tension_slider = handles["tension_slider"]
    transition_sec_number = handles["transition_sec_number"]
    framerate_number = handles["framerate_number"]
    duration_number = handles["duration_number"]
    render_time_slider = handles.get("render_time")

    if "default_fov" in json_data:
        fov_degrees_slider.value = float(json_data["default_fov"])
    if "render_width" in json_data and "render_height" in json_data:
        render_res_vec2.value = (
            float(json_data["render_width"]),
            float(json_data["render_height"]),
        )
    if "fps" in json_data:
        framerate_number.value = float(json_data["fps"])
    if "seconds" in json_data:
        duration_number.value = float(json_data["seconds"])
    if "is_cycle" in json_data:
        loop_checkbox.value = bool(json_data["is_cycle"])
    if "smoothness_value" in json_data:
        tension_slider.value = float(json_data["smoothness_value"])
    if "default_transition_sec" in json_data:
        transition_sec_number.value = float(json_data["default_transition_sec"])

    camera_path.reset()
    camera_path.time_enabled = bool(time_enabled)
    camera_path.loop = bool(loop_checkbox.value)
    camera_path.tension = float(tension_slider.value)
    camera_path.framerate = float(framerate_number.value)
    camera_path.default_fov = float(fov_degrees_slider.value) / 180.0 * np.pi
    camera_path.default_transition_sec = float(transition_sec_number.value)

    keyframes = json_data.get("keyframes", [])
    inferred_frame_numbers = _infer_keyframe_frame_numbers(json_data)
    for frame_idx, frame in enumerate(keyframes):
        pose = tf.SE3.from_matrix(np.array(frame["matrix"]).reshape(4, 4))
        pose = tf.SE3.from_rotation_and_translation(
            pose.rotation() @ tf.SO3.from_x_radians(np.pi),
            pose.translation(),
        )
        frame_number = _saved_frame_number(frame)
        has_frame_number = frame_number is not None
        if frame_number is None:
            frame_number = inferred_frame_numbers[frame_idx]
            has_frame_number = frame_number is not None
        if frame_number is None:
            frame_number = camera_path.default_render_time
        frame_number = _clamp_normalized_time(frame_number)
        camera_path.add_camera(
            Keyframe(
                position=pose.translation() * scale_ratio,
                wxyz=pose.rotation().wxyz,
                override_fov_enabled=abs(
                    float(frame["fov"]) - float(json_data.get("default_fov", 0.0))
                )
                > 1e-3,
                override_fov_rad=float(frame["fov"]) / 180.0 * np.pi,
                override_time_enabled=bool(
                    frame.get("override_time_enabled", False) or has_frame_number
                ),
                override_time_val=frame_number,
                aspect=float(frame["aspect"]),
                override_transition_enabled=bool(
                    frame.get("override_transition_enabled", False)
                ),
                override_transition_sec=frame.get("override_transition_sec", None),
            ),
        )

    if render_time_slider is not None and keyframes:
        first_frame_number = _saved_frame_number(keyframes[0])
        if first_frame_number is not None:
            render_time_slider.value = _clamp_normalized_time(first_frame_number)
            camera_path.default_render_time = render_time_slider.value

    duration_number.value = camera_path.compute_duration()
    camera_path.update_spline()
    for callback in handles.get("_after_load_camera_path_callbacks", ()):
        callback()


def _infer_keyframe_frame_numbers(json_data: dict) -> list:
    keyframes = json_data.get("keyframes", [])
    camera_path = json_data.get("camera_path", [])
    out = [None for _ in keyframes]
    if not keyframes or not camera_path:
        return out

    default_transition = float(json_data.get("default_transition_sec", 0.5))

    def transition_seconds(frame: dict) -> float:
        if frame.get("override_transition_enabled", False):
            value = frame.get("override_transition_sec", None)
            if value is not None:
                return float(value)
        return default_transition

    cumulative = [0.0]
    total = 0.0
    for frame in keyframes[1:]:
        total += transition_seconds(frame)
        cumulative.append(total)
    if json_data.get("is_cycle", False) and keyframes:
        total += transition_seconds(keyframes[0])

    duration = max(float(json_data.get("seconds", 0.0)), total, cumulative[-1], 1e-8)
    max_render_idx = max(0, len(camera_path) - 1)
    for idx, elapsed in enumerate(cumulative):
        path_time = _clamp_normalized_time(elapsed / duration)
        render_idx = int(round(path_time * max_render_idx))
        render_idx = max(0, min(max_render_idx, render_idx))
        frame_number = _saved_frame_number(camera_path[render_idx])
        if frame_number is None:
            frame_number = path_time
        out[idx] = _clamp_normalized_time(frame_number)
    return out


def _saved_frame_number(frame: dict) -> Optional[float]:
    for key in ("frame_number", "render_time", "override_time_val"):
        value = frame.get(key, None)
        if value is not None:
            return float(value)
    return None


def _clamp_normalized_time(value: float) -> float:
    return float(max(0.0, min(1.0, float(value))))


def _flatten_gui_handles(handles: dict, prefix: str = "") -> dict:
    flat_handles = {}
    for name, handle in handles.items():
        key = f"{prefix}{name}"
        if hasattr(handle, "disabled"):
            flat_handles[key] = handle
        elif isinstance(handle, dict):
            flat_handles.update(_flatten_gui_handles(handle, prefix=f"{key}."))
    return flat_handles


def _normalized_time_from_frame(frame_idx: int, num_frames: int) -> float:
    if num_frames <= 1:
        return 0.0
    return float(max(0, min(frame_idx, num_frames - 1))) / float(num_frames - 1)


def _frame_from_normalized_time(render_time: float, num_frames: int) -> int:
    return int(round(max(0.0, min(1.0, float(render_time))) * max(0, num_frames - 1)))


def _camera_path_from_callbacks(callbacks):
    for callback in callbacks:
        for cell in getattr(callback, "__closure__", None) or ():
            try:
                value = cell.cell_contents
            except ValueError:
                continue
            if all(
                hasattr(value, attr)
                for attr in ("_keyframes", "default_render_time", "update_spline")
            ):
                return value
    return None


def _last_camera_path_keyframe(camera_path):
    if camera_path is None or not camera_path._keyframes:
        return None
    return next(reversed(camera_path._keyframes.values()))[0]


def _ensure_camera_path_time_enabled(camera_path, num_frames: int) -> None:
    if camera_path is None or num_frames <= 1:
        return
    camera_path.time_enabled = True
    camera_path.update_spline()


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


def _get_viewdirs(
    width: int,
    height: int,
    c2w: torch.Tensor,
    K: torch.Tensor,
    camera_model: str = "pinhole",
    radial_coeffs: Optional[torch.Tensor] = None,
    tangential_coeffs: Optional[torch.Tensor] = None,
    thin_prism_coeffs: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    if K.is_cuda:
        raymap = compute_raymap(
            K[None],
            width,
            height,
            camera_model=camera_model,
            radial_coeffs=radial_coeffs[None] if radial_coeffs is not None else None,
            tangential_coeffs=(
                tangential_coeffs[None] if tangential_coeffs is not None else None
            ),
            thin_prism_coeffs=(
                thin_prism_coeffs[None] if thin_prism_coeffs is not None else None
            ),
        )[0]
        viewdirs = raymap.reshape(-1, 3) @ c2w[:3, :3].T
        viewdirs = viewdirs / (torch.linalg.norm(viewdirs, dim=-1, keepdim=True) + 1e-8)
        return viewdirs.reshape(height, width, 3)

    x, y = torch.meshgrid(
        torch.arange(width, device=c2w.device),
        torch.arange(height, device=c2w.device),
        indexing="xy",
    )
    _, viewdirs, _ = get_rays(x.flatten(), y.flatten(), c2w, K)
    return viewdirs.reshape(height, width, 3)


def _get_image_infos(
    width: int,
    height: int,
    c2w: torch.Tensor,
    K: torch.Tensor,
    camera_model: str = "pinhole",
    radial_coeffs: Optional[torch.Tensor] = None,
    tangential_coeffs: Optional[torch.Tensor] = None,
    thin_prism_coeffs: Optional[torch.Tensor] = None,
    extra_infos: Optional[dict] = None,
) -> dict:
    viewdirs = _get_viewdirs(
        width,
        height,
        c2w,
        K,
        camera_model=camera_model,
        radial_coeffs=radial_coeffs,
        tangential_coeffs=tangential_coeffs,
        thin_prism_coeffs=thin_prism_coeffs,
    )
    x, y = torch.meshgrid(
        torch.arange(width, device=c2w.device),
        torch.arange(height, device=c2w.device),
        indexing="xy",
    )
    pixel_coords = torch.stack([y.float() / height, x.float() / width], dim=-1)
    image_infos = {"viewdirs": viewdirs, "pixel_coords": pixel_coords}
    if extra_infos:
        image_infos.update(extra_infos)
    return image_infos


def _rendering_mode_from_trainer(trainer) -> str:
    render_mode = trainer.render_cfg.get("render_mode", "default")
    if render_mode == "ut":
        return "gut + eval3d"
    if render_mode == "geer":
        return "geer + eval3d"
    return "standard"


def _initial_render_settings_from_trainer(trainer) -> dict:
    render_cfg = trainer.render_cfg

    settings = {"rendering_mode": _rendering_mode_from_trainer(trainer)}
    for name in ("near_plane", "far_plane", "radius_clip", "eps2d"):
        value = render_cfg.get(name)
        if value is not None:
            settings[name] = float(value)

    antialiased = render_cfg.get("antialiased")
    if antialiased is not None:
        settings["rasterize_mode"] = "antialiased" if antialiased else "classic"
    return settings


def _frame_from_render_state(
    render_tab_state: GsplatRenderTabState,
    num_frames: int,
    frame_getter: Optional[Callable[[], int]],
) -> int:
    if render_tab_state.preview_render and frame_getter is not None:
        return int(frame_getter())
    if render_tab_state.preview_render:
        preview_time = float(render_tab_state.preview_time)
        return _frame_from_normalized_time(preview_time, num_frames)
    if frame_getter is not None:
        return int(frame_getter())
    return 0


def make_viewer_render_fn(
    trainer,
    device,
    num_frames: int,
    frame_getter: Optional[Callable[[], int]] = None,
    image_context_getter: Optional[Callable[[int, int, int, torch.device], dict]] = None,
):
    @torch.no_grad()
    def viewer_render_fn(camera_state: CameraState, render_tab_state: RenderTabState):
        assert isinstance(render_tab_state, GsplatRenderTabState)
        render_mode = render_tab_state.render_mode
        rendering_mode = render_tab_state.rendering_mode
        preview_render = bool(render_tab_state.preview_render)
        render_width = int(render_tab_state.render_width)
        render_height = int(render_tab_state.render_height)
        viewer_width = int(render_tab_state.viewer_width)
        viewer_height = int(render_tab_state.viewer_height)
        camera_model = render_tab_state.camera_model
        selected_gaussian_types = tuple(render_tab_state.selected_gaussian_types)
        background_mode = render_tab_state.background_mode
        backgrounds = tuple(render_tab_state.backgrounds)
        near_plane = float(render_tab_state.near_plane)
        far_plane = float(render_tab_state.far_plane)
        radius_clip = float(render_tab_state.radius_clip)
        eps2d = float(render_tab_state.eps2d)
        rasterize_mode = render_tab_state.rasterize_mode
        normalize_nearfar = bool(render_tab_state.normalize_nearfar)
        inverse = bool(render_tab_state.inverse)
        colormap = render_tab_state.colormap
        radial_k1 = float(render_tab_state.radial_k1)
        radial_k2 = float(render_tab_state.radial_k2)
        radial_k3 = float(render_tab_state.radial_k3)
        radial_k4 = float(render_tab_state.radial_k4)
        radial_k5 = float(render_tab_state.radial_k5)
        radial_k6 = float(render_tab_state.radial_k6)
        tangential_p1 = float(render_tab_state.tangential_p1)
        tangential_p2 = float(render_tab_state.tangential_p2)
        thin_prism_s1 = float(render_tab_state.thin_prism_s1)
        thin_prism_s2 = float(render_tab_state.thin_prism_s2)

        frame_idx = _frame_from_render_state(
            render_tab_state, num_frames, frame_getter
        )
        _set_frame_on_trainer(trainer, frame_idx)

        if preview_render:
            width = render_width
            height = render_height
        else:
            width = viewer_width
            height = viewer_height
        c2w = camera_state.c2w
        K = camera_state.get_K((width, height))
        c2w = torch.from_numpy(c2w).float().to(device)
        K = torch.from_numpy(K).float().to(device)
        viewmat = c2w.inverse()

        render_mode_map = {
            "rgb": "RGB",
            "depth(accumulated)": "D",
            "depth(expected)": "ED",
            "alpha": "RGB",
        }

        with_ut = rendering_mode == "gut + eval3d"
        with_geer = rendering_mode == "geer + eval3d"
        with_eval3d = with_ut or with_geer
        is_fisheye = camera_model == "fisheye"

        if with_ut or with_geer:
            if is_fisheye:
                radial_coeffs = torch.tensor(
                    [
                        radial_k1,
                        radial_k2,
                        radial_k3,
                        radial_k4,
                    ],
                    device=device,
                )
                tangential_coeffs = None
                thin_prism_coeffs = None
            else:
                radial_coeffs = torch.tensor(
                    [
                        radial_k1,
                        radial_k2,
                        radial_k3,
                        radial_k4,
                        radial_k5,
                        radial_k6,
                    ],
                    device=device,
                )
                tangential_coeffs = torch.tensor(
                    [
                        tangential_p1,
                        tangential_p2,
                    ],
                    device=device,
                )
                thin_prism_coeffs = torch.tensor(
                    [
                        thin_prism_s1,
                        thin_prism_s2,
                        0.0,
                        0.0,
                    ],
                    device=device,
                )
        else:
            radial_coeffs = None
            tangential_coeffs = None
            thin_prism_coeffs = None

        def get_image_infos() -> dict:
            extra_infos = (
                image_context_getter(frame_idx, int(height), int(width), device)
                if image_context_getter is not None
                else None
            )
            return _get_image_infos(
                width,
                height,
                c2w,
                K,
                camera_model=camera_model,
                radial_coeffs=radial_coeffs,
                tangential_coeffs=tangential_coeffs,
                thin_prism_coeffs=thin_prism_coeffs,
                extra_infos=extra_infos,
            )

        cam = dataclass_camera(
            camtoworlds=c2w,
            camtoworlds_gt=c2w,
            Ks=K,
            H=int(height),
            W=int(width),
        )

        selected_gaussian_types_set = set(selected_gaussian_types)
        all_gaussian_types = tuple(trainer.gaussian_classes.keys())
        all_gs_per_class = []
        gs_per_class = []
        for class_name in all_gaussian_types:
            model = trainer.models[class_name]
            gs = model.get_gaussians(cam)
            if gs is not None:
                all_gs_per_class.append(gs)
            if class_name in selected_gaussian_types_set and gs is not None:
                gs_per_class.append(gs)

        merged = _merge_gaussians(gs_per_class)
        merged_all = _merge_gaussians(all_gs_per_class)
        use_sky_background = (
            background_mode == "sky"
            and render_mode == "rgb"
            and "Sky" in trainer.models
        )

        def rasterize_merged_gaussians(merged_gs, backgrounds_arg, render_mode_arg):
            return rasterization(
                merged_gs.means,
                merged_gs.quats,
                merged_gs.scales,
                merged_gs.opacities.squeeze(-1),
                merged_gs.rgbs,
                viewmat[None],
                K[None],
                width,
                height,
                near_plane=near_plane,
                far_plane=far_plane,
                radius_clip=radius_clip,
                eps2d=eps2d,
                backgrounds=backgrounds_arg,
                render_mode=render_mode_arg,
                rasterize_mode=rasterize_mode,
                camera_model=camera_model,
                packed=False,
                with_ut=with_ut,
                with_geer=with_geer,
                with_eval3d=with_eval3d,
                radial_coeffs=(
                    radial_coeffs[None, ...]
                    if radial_coeffs is not None
                    else None
                ),
                tangential_coeffs=(
                    tangential_coeffs[None, ...]
                    if tangential_coeffs is not None
                    else None
                ),
                thin_prism_coeffs=(
                    thin_prism_coeffs[None, ...]
                    if thin_prism_coeffs is not None
                    else None
                ),
            )

        def render_full_alpha():
            if merged_all is None:
                return None
            if (
                merged is not None
                and selected_gaussian_types_set == set(all_gaussian_types)
            ):
                return render_alphas[0, ..., 0:1].clamp(0, 1)
            _, full_alphas, _ = rasterize_merged_gaussians(
                merged_all,
                torch.zeros((1, 3), device=device),
                "RGB",
            )
            return full_alphas[0, ..., 0:1].clamp(0, 1)

        if merged is None:
            render_tab_state.total_gs_count = 0
            render_tab_state.rendered_gs_count = 0
            if use_sky_background:
                image_infos = get_image_infos()
                rgb_sky = trainer.models["Sky"](image_infos).clamp(0, 1)
                full_alpha = render_full_alpha()
                if full_alpha is not None:
                    rgb_sky = rgb_sky * (1.0 - full_alpha)
                return trainer.affine_transformation(rgb_sky, image_infos).clamp(0, 1).cpu().numpy()
            background = torch.tensor(
                backgrounds, device=device
            ).float() / 255.0
            render_colors = background[None, None, :].expand(height, width, 3)
            if "Affine" in trainer.models:
                image_infos = get_image_infos()
                render_colors = trainer.affine_transformation(
                    render_colors, image_infos
                ).clamp(0, 1)
            return render_colors.cpu().numpy()

        background_color = (
            torch.zeros((1, 3), device=device)
            if use_sky_background
            else torch.tensor([backgrounds], device=device).float()
            / 255.0
        )

        render_colors, render_alphas, info = rasterize_merged_gaussians(
            merged,
            background_color,
            render_mode_map[render_mode],
        )
        render_tab_state.total_gs_count = int(merged.means.shape[0])
        radii = info.get("radii", None)
        if radii is None:
            render_tab_state.rendered_gs_count = 0
        else:
            rendered_mask = (radii > 0).all(-1) if radii.ndim > 1 else (radii > 0)
            render_tab_state.rendered_gs_count = int(rendered_mask.sum().item())

        if render_mode == "rgb":
            if render_colors.shape[-1] < 3:
                raise RuntimeError(
                    "RGB render returned fewer than 3 channels "
                    f"(shape={tuple(render_colors.shape)})."
                )
            render_colors = render_colors[0, ..., 0:3].clamp(0, 1)
            image_infos = None
            if use_sky_background:
                image_infos = get_image_infos()
                rgb_sky = trainer.models["Sky"](image_infos).clamp(0, 1)
                alpha = render_full_alpha()
                if alpha is None:
                    alpha = render_alphas[0, ..., 0:1].clamp(0, 1)
                render_colors = (render_colors + rgb_sky * (1.0 - alpha)).clamp(0, 1)
            if "Affine" in trainer.models:
                if image_infos is None:
                    image_infos = get_image_infos()
                render_colors = trainer.affine_transformation(
                    render_colors, image_infos
                ).clamp(0, 1)
            renders = render_colors.cpu().numpy()
        elif render_mode in ["depth(accumulated)", "depth(expected)"]:
            depth = render_colors[0, ..., 0:1]
            if normalize_nearfar:
                depth_near = near_plane
                depth_far = far_plane
            else:
                depth_near = depth.min()
                depth_far = depth.max()
            depth_norm = (depth - depth_near) / (depth_far - depth_near + 1e-10)
            depth_norm = torch.clip(depth_norm, 0, 1)
            if inverse:
                depth_norm = 1 - depth_norm
            renders = (
                apply_float_colormap(depth_norm, colormap)
                .cpu()
                .numpy()
            )
        elif render_mode == "alpha":
            alpha = render_alphas[0, ..., 0:1]
            renders = (
                apply_float_colormap(alpha, colormap).cpu().numpy()
            )
        return renders

    return viewer_render_fn


def create_gsplat_viewer(
    trainer,
    port: int = 8080,
    output_dir: Union[str, Path] = "results/",
    mode: Literal["rendering", "training"] = "rendering",
    num_frames: Optional[int] = None,
    initial_fps: float = 10.0,
    initial_camera: Optional[dict] = None,
    image_context_getter: Optional[
        Callable[[int, int, int, torch.device], dict]
    ] = None,
    camera_snap_options: Tuple[str, ...] = (),
    camera_snap_getter: Optional[Callable[[str, int], dict]] = None,
    camera_snap_setter: Optional[Callable[[str], None]] = None,
):
    num_frames = int(num_frames or getattr(trainer, "num_timesteps", 1))
    trainer_frame = getattr(trainer, "cur_frame", 0)
    current_frame = int(trainer_frame.item()) if hasattr(trainer_frame, "item") else int(trainer_frame)
    playing = False
    fps = float(initial_fps)
    state_lock = threading.Lock()

    def frame_getter() -> int:
        with state_lock:
            return int(current_frame)

    server = viser.ViserServer(port=port, verbose=False)
    if initial_camera is not None:
        server.initial_camera.position = initial_camera["position"]
        server.initial_camera.look_at = initial_camera["look_at"]
        server.initial_camera.up_direction = initial_camera["up_direction"]
        server.initial_camera.fov = initial_camera["fov"]

    viewer = GsplatViewer(
        server=server,
        render_fn=make_viewer_render_fn(
            trainer,
            trainer.device,
            num_frames=num_frames,
            frame_getter=frame_getter,
            image_context_getter=image_context_getter,
        ),
        output_dir=Path(output_dir),
        mode=mode,
        num_frames=num_frames,
        initial_fps=initial_fps,
        gaussian_types=tuple(getattr(trainer, "gaussian_classes", {}).keys()),
        has_sky="Sky" in trainer.models,
        time_enabled=num_frames > 1,
        initial_render_settings=_initial_render_settings_from_trainer(trainer),
    )

    if initial_camera is not None:
        viewer._rendering_tab_handles["fov_degrees_slider"].value = float(
            np.rad2deg(initial_camera["fov"])
        )

        @server.on_client_connect
        def _set_initial_camera(client: viser.ClientHandle) -> None:
            client.camera.position = initial_camera["position"]
            client.camera.look_at = initial_camera["look_at"]
            client.camera.up_direction = initial_camera["up_direction"]
            client.camera.fov = initial_camera["fov"]

    time_slider = viewer._rendering_tab_handles["time_slider"]
    playback_play_button = viewer._rendering_tab_handles["playback_play_button"]
    playback_pause_button = viewer._rendering_tab_handles["playback_pause_button"]
    fps_slider = viewer._rendering_tab_handles["fps_slider"]
    render_time_slider = viewer._rendering_tab_handles["render_time"]
    mode_status = viewer._rendering_tab_handles["mode_status"]
    mode_button = viewer._rendering_tab_handles["mode_button"]
    add_keyframe_button = viewer._rendering_tab_handles["add_keyframe_button"]
    trajectory_play_button = viewer._rendering_tab_handles["play_button"]
    trajectory_pause_button = viewer._rendering_tab_handles["pause_button"]
    transition_sec_number = viewer._rendering_tab_handles["transition_sec_number"]
    trajectory_fps_number = viewer._rendering_tab_handles["framerate_number"]
    trajectory_duration_number = viewer._rendering_tab_handles["duration_number"]
    trajectory_name_text = viewer._rendering_tab_handles["trajectory_name_text"]
    render_res_vec2 = viewer._rendering_tab_handles["render_res_vec2"]
    show_keyframe_checkbox = viewer._rendering_tab_handles["show_keyframe_checkbox"]
    show_spline_checkbox = viewer._rendering_tab_handles["show_spline_checkbox"]
    preview_render_stop_button = viewer._rendering_tab_handles[
        "preview_render_stop_button"
    ]
    dump_video_button = viewer._rendering_tab_handles["dump_video_button"]
    playback_camera_view_controls = [
        time_slider,
        fps_slider,
        playback_play_button,
        playback_pause_button,
    ]

    def sync_trajectory_time_to_frame(frame_idx: int) -> None:
        if render_time_slider is None:
            return
        render_time_slider.value = _normalized_time_from_frame(frame_idx, num_frames)

    if render_time_slider is not None:
        sync_trajectory_time_to_frame(current_frame)
        render_time_slider.visible = False

    camera_path = None
    if add_keyframe_button is not None:
        original_callbacks = list(add_keyframe_button._impl.update_cb)
        camera_path = _camera_path_from_callbacks(original_callbacks)
        _ensure_camera_path_time_enabled(camera_path, num_frames)
        if camera_path is not None and original_callbacks:
            add_keyframe_button._impl.update_cb.clear()

            @add_keyframe_button.on_click
            def _(event: viser.GuiEvent) -> None:
                with state_lock:
                    frame_idx = int(current_frame)
                keyframe_time = _normalized_time_from_frame(frame_idx, num_frames)
                synced_transition_sec = (
                    pending_synced_transition_seconds()
                    if time_sync_checkbox.value
                    else None
                )
                if render_time_slider is not None:
                    render_time_slider.value = keyframe_time
                camera_path.default_render_time = keyframe_time

                for callback in original_callbacks:
                    callback(event)

                keyframe = _last_camera_path_keyframe(camera_path)
                if keyframe is not None:
                    keyframe.override_time_enabled = True
                    keyframe.override_time_val = keyframe_time
                    if synced_transition_sec is not None:
                        keyframe.override_transition_enabled = True
                        keyframe.override_transition_sec = synced_transition_sec
                    trajectory_duration_number.value = camera_path.compute_duration()
                    camera_path.update_spline()
                update_time_sync_transition()

    with viewer._export_folder:
        time_sync_checkbox = server.gui.add_checkbox(
            "Sync Time",
            initial_value=False,
            hint="Set Transition from the playback frame distance to the last keyframe, divided by export FPS.",
            order=transition_sec_number.order + 0.01,
        )
    viewer._rendering_tab_handles["time_sync_checkbox"] = time_sync_checkbox

    def last_keyframe_frame() -> Optional[int]:
        keyframe = _last_camera_path_keyframe(camera_path)
        if keyframe is None:
            return None
        if keyframe.override_time_enabled and keyframe.override_time_val is not None:
            render_time = float(keyframe.override_time_val)
        elif camera_path is not None:
            render_time = float(camera_path.default_render_time)
        else:
            return None
        return _frame_from_normalized_time(render_time, num_frames)

    def pending_synced_transition_seconds() -> Optional[float]:
        previous_frame = last_keyframe_frame()
        if previous_frame is None:
            return None
        with state_lock:
            frame_idx = int(current_frame)
        fps_value = max(float(trajectory_fps_number.value), 1e-8)
        return abs(frame_idx - previous_frame) / fps_value

    def synced_transition_seconds() -> float:
        return float(pending_synced_transition_seconds() or 0.0)

    def set_transition_field_value(value: float, trigger_callbacks: bool) -> None:
        value = float(value)
        if trigger_callbacks:
            transition_sec_number.value = value
            return
        if transition_sec_number.value == value:
            return
        transition_sec_number._impl.value = type(transition_sec_number._impl.value)(
            value
        )
        transition_sec_number._queue_update("value", transition_sec_number.value)

    def commit_transition_field_to_default() -> None:
        if camera_path is None:
            return
        camera_path.default_transition_sec = float(transition_sec_number.value)
        trajectory_duration_number.value = camera_path.compute_duration()
        camera_path.update_spline()

    def update_time_sync_transition() -> None:
        if not time_sync_checkbox.value:
            return
        transition_sec_number.disabled = True
        set_transition_field_value(
            synced_transition_seconds(), trigger_callbacks=False
        )

    @time_sync_checkbox.on_update
    def _(_event: viser.GuiEvent) -> None:
        if time_sync_checkbox.value:
            update_time_sync_transition()
            return
        transition_sec_number.disabled = False
        commit_transition_field_to_default()

    attached_preview_slider = {"handle": None}
    export_playing = {"value": False}
    export_playback_frame = {"value": 0}
    export_preview_slider_update = {"active": False}

    def set_playback_controls_disabled(disabled: bool) -> None:
        nonlocal playing
        if disabled:
            with state_lock:
                playing = False
            playback_play_button.visible = True
            playback_pause_button.visible = False
        for handle in playback_camera_view_controls:
            handle.disabled = disabled

    def export_actions_available() -> bool:
        if camera_path is None or len(camera_path._keyframes) < 2:
            return False
        return (
            int(
                float(trajectory_fps_number.value)
                * float(trajectory_duration_number.value)
            )
            > 0
        )

    def sync_export_action_enabled() -> None:
        disabled = (
            not export_actions_available()
            or bool(export_playing["value"])
            or bool(viewer.render_tab_state.preview_render)
        )
        dump_video_button.disabled = disabled
        trajectory_play_button.disabled = not export_actions_available()

    def pause_export_trajectory() -> None:
        trajectory_play_button.visible = True
        trajectory_pause_button.visible = False
        export_playing["value"] = False
        set_playback_controls_disabled(False)
        sync_export_action_enabled()

    def exit_render_preview() -> None:
        if not viewer.render_tab_state.preview_render:
            return
        callbacks = list(preview_render_stop_button._impl.update_cb)
        if callbacks:
            for callback in callbacks:
                callback(None)
            return
        viewer.render_tab_state.preview_render = False
        preview_render_stop_button.visible = False
        server.scene.set_global_visibility(True)
        sync_export_action_enabled()

    def stop_export_interaction() -> None:
        pause_export_trajectory()
        exit_render_preview()

    def sync_export_guides_visibility(render_mode_enabled: bool) -> None:
        if camera_path is None:
            return
        if render_mode_enabled:
            camera_path.set_keyframes_visible(False)
            camera_path.show_spline = False
            server.scene.remove_by_name("/preview_camera")
        else:
            camera_path.set_keyframes_visible(bool(show_keyframe_checkbox.value))
            camera_path.show_spline = bool(show_spline_checkbox.value)
            preview_frame_slider = get_preview_frame_slider()
            if preview_frame_slider is not None:
                for callback in list(preview_frame_slider._impl.update_cb):
                    callback(None)
        camera_path.update_spline()

    ui_mode = {"value": None}

    def set_viewer_ui_mode(mode: Literal["render", "export"]) -> None:
        previous_mode = ui_mode["value"]
        if previous_mode == mode:
            return

        if mode == "render" and previous_mode == "export":
            stop_export_interaction()

        ui_mode["value"] = mode

        render_mode_enabled = mode == "render"
        viewer._gaussians_folder.visible = render_mode_enabled
        viewer._rendering_folder.visible = render_mode_enabled
        viewer._camera_intrinsics_folder.visible = render_mode_enabled
        viewer._export_folder.visible = not render_mode_enabled
        viewer._playback_camera_view_folder.visible = True
        sync_export_guides_visibility(render_mode_enabled)
        sync_export_action_enabled()

        if mode == "render":
            mode_status.content = "**Mode:** Render. Switching saves render settings."
            mode_button.label = "Switch to Export Mode"
            mode_button.hint = "Saves render mode settings."
            mode_button.color = "blue"
        else:
            mode_status.content = (
                "**Mode:** Export. Switching back preserves export trajectory."
            )
            mode_button.label = "Switch to Render Mode"
            mode_button.hint = "Preserves export trajectory."
            mode_button.color = "blue"

    set_viewer_ui_mode("render")

    @mode_button.on_click
    def _(_event: viser.GuiEvent) -> None:
        if ui_mode["value"] == "render":
            set_viewer_ui_mode("export")
            return
        set_viewer_ui_mode("render")

    def get_preview_frame_slider():
        preview_frame_slider = viewer._nerfview_render_handles.get(
            "preview_frame_slider"
        )
        if (
            preview_frame_slider is not None
            and not preview_frame_slider._impl.removed
        ):
            viewer._rendering_tab_handles["preview_frame_slider"] = preview_frame_slider
        else:
            preview_frame_slider = None
        return preview_frame_slider

    def place_preview_frame_slider() -> None:
        preview_frame_slider = get_preview_frame_slider()
        if preview_frame_slider is None:
            return
        play_order = min(trajectory_play_button.order, trajectory_pause_button.order)
        preview_frame_slider.order = play_order - 0.01

    def sync_frame_to_trajectory_preview() -> None:
        nonlocal current_frame
        preview_frame_slider = get_preview_frame_slider()
        if preview_frame_slider is None or camera_path is None:
            return
        max_frame_idx = max(
            1,
            int(
                float(trajectory_fps_number.value)
                * float(trajectory_duration_number.value)
            )
            - 1,
        )
        maybe_pose_fov_time = camera_path.interpolate_pose_and_fov_rad(
            float(preview_frame_slider.value) / float(max_frame_idx)
        )
        if maybe_pose_fov_time is None or len(maybe_pose_fov_time) != 3:
            return
        preview_time = float(maybe_pose_fov_time[2])
        viewer.render_tab_state.preview_time = preview_time
        frame_idx = _frame_from_normalized_time(preview_time, num_frames)
        with state_lock:
            current_frame = frame_idx
        if int(time_slider.value) != frame_idx:
            time_slider.value = frame_idx

    def sync_scene_to_export_frame(export_frame: int) -> None:
        nonlocal current_frame
        if camera_path is None:
            return
        max_frame_idx = max(
            1,
            int(
                float(trajectory_fps_number.value)
                * float(trajectory_duration_number.value)
            )
            - 1,
        )
        maybe_pose_fov_time = camera_path.interpolate_pose_and_fov_rad(
            float(export_frame) / float(max_frame_idx)
        )
        if maybe_pose_fov_time is None:
            return
        if len(maybe_pose_fov_time) == 3:
            pose, fov_rad, preview_time = maybe_pose_fov_time
        else:
            pose, fov_rad = maybe_pose_fov_time
            preview_time = viewer.render_tab_state.preview_time
        preview_time = float(preview_time)
        viewer.render_tab_state.preview_time = preview_time
        viewer.render_tab_state.preview_fov = float(fov_rad)
        viewer.render_tab_state.preview_aspect = camera_path.get_aspect()
        frame_idx = _frame_from_normalized_time(preview_time, num_frames)
        with state_lock:
            current_frame = frame_idx
        if int(time_slider.value) != frame_idx:
            time_slider.value = frame_idx
        if viewer.render_tab_state.preview_render:
            for client in server.get_clients().values():
                with client.atomic():
                    client.camera.wxyz = pose.rotation().wxyz
                    client.camera.position = pose.translation()
                    client.camera.fov = float(fov_rad)
        viewer.rerender(None)

    def attach_preview_frame_sync() -> None:
        preview_frame_slider = get_preview_frame_slider()
        if (
            preview_frame_slider is None
            or preview_frame_slider is attached_preview_slider["handle"]
        ):
            return
        place_preview_frame_slider()
        attached_preview_slider["handle"] = preview_frame_slider

        @preview_frame_slider.on_update
        def _(_event: viser.GuiEvent) -> None:
            if export_playing["value"] and not export_preview_slider_update["active"]:
                export_playback_frame["value"] = int(preview_frame_slider.value)
            sync_frame_to_trajectory_preview()

    attach_preview_frame_sync()

    def sync_after_trajectory_load() -> None:
        stop_export_interaction()
        attach_preview_frame_sync()
        place_preview_frame_slider()
        export_playback_frame["value"] = 0
        preview_frame_slider = get_preview_frame_slider()
        if (
            preview_frame_slider is not None
            and not preview_frame_slider._impl.removed
        ):
            try:
                preview_frame_slider.value = 0
            except RuntimeError:
                pass
        sync_scene_to_export_frame(0)
        sync_export_action_enabled()

    viewer._nerfview_render_handles.setdefault(
        "_after_load_camera_path_callbacks", []
    ).append(sync_after_trajectory_load)

    @trajectory_fps_number.on_update
    @trajectory_duration_number.on_update
    def _(_event: viser.GuiEvent) -> None:
        attach_preview_frame_sync()
        place_preview_frame_slider()
        update_time_sync_transition()
        sync_export_action_enabled()

    if camera_path is not None:
        original_update_spline_for_actions = camera_path.update_spline

        def update_spline_with_action_sync(*args, **kwargs):
            result = original_update_spline_for_actions(*args, **kwargs)
            sync_export_action_enabled()
            return result

        camera_path.update_spline = update_spline_with_action_sync
        sync_export_action_enabled()

        trajectory_play_button._impl.update_cb.clear()
        trajectory_pause_button._impl.update_cb.clear()
        dump_video_button._impl.update_cb.clear()

        @trajectory_play_button.on_click
        def _(_event: viser.GuiEvent) -> None:
            trajectory_play_button.visible = False
            trajectory_pause_button.visible = True
            export_playing["value"] = True
            dump_video_button.disabled = True
            trajectory_play_button.disabled = True
            set_playback_controls_disabled(True)
            preview_frame_slider = get_preview_frame_slider()
            if (
                preview_frame_slider is not None
                and not preview_frame_slider._impl.removed
            ):
                export_playback_frame["value"] = int(preview_frame_slider.value)

            def play_trajectory() -> None:
                last = time.time()
                carry = 0.0
                while not trajectory_play_button.visible:
                    fps_value = float(max(0.1, trajectory_fps_number.value))
                    max_frame = max(
                        1,
                        int(
                            float(trajectory_fps_number.value)
                            * float(trajectory_duration_number.value)
                        ),
                    )
                    now = time.time()
                    carry += (now - last) * fps_value
                    last = now
                    step = int(carry)
                    if step > 0:
                        carry -= float(step)
                        export_playback_frame["value"] = (
                            export_playback_frame["value"] + step
                        ) % max_frame
                        attach_preview_frame_sync()
                        preview_frame_slider = get_preview_frame_slider()
                        if (
                            preview_frame_slider is not None
                            and not preview_frame_slider._impl.removed
                        ):
                            export_preview_slider_update["active"] = True
                            try:
                                preview_frame_slider.value = export_playback_frame[
                                    "value"
                                ]
                            except RuntimeError:
                                pass
                            finally:
                                export_preview_slider_update["active"] = False
                        sync_scene_to_export_frame(export_playback_frame["value"])
                    time.sleep(0.005)
                export_playing["value"] = False
                sync_export_action_enabled()

            threading.Thread(target=play_trajectory, daemon=True).start()

        @trajectory_pause_button.on_click
        def _(_event: viser.GuiEvent) -> None:
            trajectory_play_button.visible = True
            trajectory_pause_button.visible = False
            export_playing["value"] = False
            set_playback_controls_disabled(False)

        def trajectory_pose_fov_time(export_frame: int, max_frame: int):
            nonlocal current_frame
            maybe_pose_fov_time = camera_path.interpolate_pose_and_fov_rad(
                float(export_frame) / float(max(1, max_frame - 1))
            )
            if maybe_pose_fov_time is None:
                return None
            if len(maybe_pose_fov_time) == 3:
                pose, fov_rad, preview_time = maybe_pose_fov_time
            else:
                pose, fov_rad = maybe_pose_fov_time
                preview_time = viewer.render_tab_state.preview_time
            viewer.render_tab_state.preview_time = float(preview_time)
            viewer.render_tab_state.preview_fov = float(fov_rad)
            viewer.render_tab_state.preview_aspect = camera_path.get_aspect()
            frame_idx = _frame_from_normalized_time(float(preview_time), num_frames)
            with state_lock:
                current_frame = frame_idx
            return pose, float(fov_rad)

        def render_trajectory_frame(pose, fov_rad: float, width: int, height: int):
            c2w = pose.as_matrix()
            camera_state = CameraState(
                fov=float(fov_rad),
                aspect=float(width) / float(height),
                c2w=np.asarray(c2w, dtype=np.float32),
            )
            with viewer.lock:
                image = viewer.render_fn(camera_state, viewer.render_tab_state)
                viewer._after_render()
                if isinstance(image, tuple):
                    image = image[0]
            image = np.asarray(image)
            if np.issubdtype(image.dtype, np.floating):
                image = (np.clip(image, 0.0, 1.0) * 255.0).astype(np.uint8)
            return image

        def set_dump_controls_disabled(disabled: bool):
            handles_to_disable = _flatten_gui_handles(
                viewer._rendering_tab_handles
            ).values()
            original_disabled = []
            for handle in handles_to_disable:
                if getattr(handle._impl, "removed", False):
                    continue
                try:
                    current_disabled = bool(handle.disabled)
                    handle.disabled = disabled
                except (AssertionError, RuntimeError):
                    continue
                original_disabled.append((handle, current_disabled))
            return original_disabled

        def restore_dump_controls(original_disabled) -> None:
            for handle, disabled in original_disabled:
                if getattr(handle._impl, "removed", False):
                    continue
                try:
                    handle.disabled = disabled
                except (AssertionError, RuntimeError):
                    continue

        @dump_video_button.on_click
        def _(event: viser.GuiEvent) -> None:
            del event
            max_frame = int(
                float(trajectory_fps_number.value)
                * float(trajectory_duration_number.value)
            )
            if max_frame <= 0:
                return

            original_preview_render = bool(viewer.render_tab_state.preview_render)
            original_disabled = set_dump_controls_disabled(True)
            viewer.render_tab_state.preview_render = True
            server.scene.set_global_visibility(False)

            def dump() -> None:
                video_dir = viewer.output_dir / "videos"
                video_dir.mkdir(parents=True, exist_ok=True)
                output_path = video_dir / f"traj_{trajectory_name_text.value}.mp4"
                writer = imageio.get_writer(
                    str(output_path),
                    fps=float(trajectory_fps_number.value),
                )
                try:
                    width = int(render_res_vec2.value[0])
                    height = int(render_res_vec2.value[1])
                    viewer.render_tab_state.render_width = width
                    viewer.render_tab_state.render_height = height
                    preview_frame_slider = get_preview_frame_slider()
                    for export_frame in range(max_frame):
                        if (
                            preview_frame_slider is not None
                            and not preview_frame_slider._impl.removed
                        ):
                            try:
                                preview_frame_slider.value = export_frame
                            except RuntimeError:
                                preview_frame_slider = get_preview_frame_slider()
                        pose_fov = trajectory_pose_fov_time(export_frame, max_frame)
                        if pose_fov is None:
                            continue
                        pose, fov_rad = pose_fov
                        image = render_trajectory_frame(
                            pose, fov_rad, width=width, height=height
                        )
                        writer.append_data(image)
                finally:
                    writer.close()
                print(f"Video saved to {output_path}")

            try:
                dump_thread = threading.Thread(target=dump)
                dump_thread.start()
                dump_thread.join()
            finally:
                viewer.render_tab_state.preview_render = original_preview_render
                server.scene.set_global_visibility(True)
                restore_dump_controls(original_disabled)
                sync_export_action_enabled()

    def apply_camera_state(camera_state: dict) -> None:
        fov = camera_state.get("fov")
        if fov is not None:
            viewer._rendering_tab_handles["fov_degrees_slider"].value = float(
                np.rad2deg(fov)
            )
        for client in server.get_clients().values():
            with client.atomic():
                client.camera.position = camera_state["position"]
                if "wxyz" in camera_state:
                    client.camera.wxyz = camera_state["wxyz"]
                else:
                    client.camera.look_at = camera_state["look_at"]
                    client.camera.up_direction = camera_state["up_direction"]
                if fov is not None:
                    client.camera.fov = fov

    def keyframe_camera_state(keyframe) -> dict:
        position = np.asarray(keyframe.position, dtype=np.float64)
        return {
            "position": tuple(float(x) for x in position),
            "wxyz": tuple(float(x) for x in keyframe.wxyz),
            "fov": float(keyframe.override_fov_rad)
            if keyframe.override_fov_enabled
            else None,
        }

    if camera_snap_options and camera_snap_getter is not None:
        base_camera_snap_options = tuple(camera_snap_options)
        keyframe_camera_view_ids = {}
        keyframe_camera_view_labels = {}
        snap_camera_dropdown = None

        def current_camera_view_options() -> Tuple[str, ...]:
            keyframe_camera_view_ids.clear()
            keyframe_camera_view_labels.clear()
            keyframe_options = []
            if camera_path is not None:
                for index, (keyframe_id, _keyframe_tuple) in enumerate(
                    camera_path._keyframes.items()
                ):
                    label = f"Keyframe {index}"
                    keyframe_camera_view_ids[label] = keyframe_id
                    keyframe_camera_view_labels[keyframe_id] = label
                    keyframe_options.append(label)
            return base_camera_snap_options + tuple(keyframe_options)

        def sync_camera_view_options() -> None:
            if snap_camera_dropdown is None:
                return
            previous_value = snap_camera_dropdown.value
            previous_keyframe_id = keyframe_camera_view_ids.get(previous_value)
            options = current_camera_view_options()
            if not options:
                return
            next_value = previous_value if previous_value in options else None
            if next_value is None and previous_keyframe_id is not None:
                next_value = keyframe_camera_view_labels.get(previous_keyframe_id)
            snap_camera_dropdown.options = options
            if next_value is not None and snap_camera_dropdown.value != next_value:
                snap_camera_dropdown.value = next_value

        def get_camera_view_state(label: str, frame_idx: int) -> dict:
            keyframe_id = keyframe_camera_view_ids.get(label)
            if keyframe_id is not None and camera_path is not None:
                keyframe_tuple = camera_path._keyframes.get(keyframe_id)
                if keyframe_tuple is not None:
                    return keyframe_camera_state(keyframe_tuple[0])
            return camera_snap_getter(label, frame_idx)

        def set_dataset_camera_view(label: str) -> None:
            if label not in keyframe_camera_view_ids and camera_snap_setter is not None:
                camera_snap_setter(label)

        with viewer._playback_camera_view_folder:
            snap_camera_dropdown = server.gui.add_dropdown(
                "Camera View",
                current_camera_view_options(),
                initial_value=base_camera_snap_options[0],
                hint="Dataset camera (first trajectory frame) or keyframe used as the snap target for the viewer camera.",
            )
            snap_camera_button = server.gui.add_button("Snap to Camera")
            playback_camera_view_controls.extend(
                (snap_camera_dropdown, snap_camera_button)
            )

            @snap_camera_dropdown.on_update
            def _(_event: viser.GuiEvent) -> None:
                set_dataset_camera_view(snap_camera_dropdown.value)
                viewer.rerender(_event)

            @snap_camera_button.on_click
            def _(_event: viser.GuiEvent) -> None:
                set_dataset_camera_view(snap_camera_dropdown.value)
                with state_lock:
                    frame_idx = int(current_frame)
                apply_camera_state(
                    get_camera_view_state(snap_camera_dropdown.value, frame_idx)
                )
                viewer.rerender(_event)

        if camera_path is not None:
            original_add_camera = camera_path.add_camera
            original_update_spline = camera_path.update_spline

            def add_camera_with_camera_view_sync(*args, **kwargs):
                result = original_add_camera(*args, **kwargs)
                sync_camera_view_options()
                update_time_sync_transition()
                return result

            def update_spline_with_camera_view_sync(*args, **kwargs):
                result = original_update_spline(*args, **kwargs)
                sync_camera_view_options()
                update_time_sync_transition()
                return result

            camera_path.add_camera = add_camera_with_camera_view_sync
            camera_path.update_spline = update_spline_with_camera_view_sync
            sync_camera_view_options()

        viewer._rendering_tab_handles.update(
            {
                "snap_camera_dropdown": snap_camera_dropdown,
                "snap_camera_button": snap_camera_button,
            }
        )

    @time_slider.on_update
    def _(_event: viser.GuiEvent) -> None:
        nonlocal current_frame
        with state_lock:
            current_frame = int(time_slider.value)
            frame_idx = int(current_frame)
        sync_trajectory_time_to_frame(frame_idx)
        update_time_sync_transition()
        viewer.rerender(_event)

    @fps_slider.on_update
    def _(_event: viser.GuiEvent) -> None:
        nonlocal fps
        with state_lock:
            fps = float(fps_slider.value)

    @playback_play_button.on_click
    def _(_event: viser.GuiEvent) -> None:
        nonlocal playing
        with state_lock:
            playing = True
        playback_play_button.visible = False
        playback_pause_button.visible = True

    @playback_pause_button.on_click
    def _(_event: viser.GuiEvent) -> None:
        nonlocal playing
        with state_lock:
            playing = False
        playback_play_button.visible = True
        playback_pause_button.visible = False
    def playback_loop() -> None:

        nonlocal current_frame
        nonlocal playing
        last = time.time()
        carry = 0.0
        while True:
            now = time.time()
            dt = now - last
            last = now

            with state_lock:
                _playing = bool(playing)
                _fps = float(max(0.0, fps))
                _frame = int(current_frame)

            if not _playing or _fps <= 0.0 or num_frames <= 1:
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
                nxt = nxt % num_frames

            with state_lock:
                current_frame = int(nxt)
            time_slider.value = int(nxt)
            viewer.rerender(None)
            time.sleep(0.001)

    threading.Thread(target=playback_loop, daemon=True).start()
    return viewer


def _camera_state_from_camera(camera, frame_idx: int) -> dict:
    num_frames = int(getattr(camera, "num_frames", 1))
    idx = int(max(0, min(frame_idx, num_frames - 1)))
    c2w = camera.cam_to_worlds[idx].detach().cpu().numpy()
    K = camera.intrinsics[idx].detach().cpu().numpy()

    height = float(getattr(camera, "HEIGHT"))
    fy = float(K[1, 1])
    position = c2w[:3, 3].astype(np.float64)
    look_at = (position + c2w[:3, 2]).astype(np.float64)
    up_direction = (-c2w[:3, 1]).astype(np.float64)
    fov = float(2.0 * np.arctan2(height * 0.5, fy))

    return {
        "position": tuple(float(x) for x in position),
        "look_at": tuple(float(x) for x in look_at),
        "up_direction": tuple(float(x) for x in up_direction),
        "fov": fov,
    }


def _dataset_camera_tools(dataset) -> dict:
    pixel_source = getattr(dataset, "pixel_source", None)
    camera_list = getattr(pixel_source, "camera_list", None)
    camera_data = getattr(pixel_source, "camera_data", None)
    if not camera_list or camera_data is None:
        return {}

    def label_for(cam_id) -> str:
        camera = camera_data[cam_id]
        cam_name = getattr(camera, "cam_name", str(cam_id))
        return f"{cam_id}: {cam_name}"

    camera_options = tuple(label_for(cam_id) for cam_id in camera_list)
    label_to_id = {label_for(cam_id): cam_id for cam_id in camera_list}
    selected = {"cam_id": camera_list[0]}

    def set_active_camera(label: str) -> None:
        selected["cam_id"] = label_to_id[label]

    def get_camera_state(label: str, frame_idx: int) -> dict:
        set_active_camera(label)
        return _camera_state_from_camera(camera_data[selected["cam_id"]], frame_idx)

    def get_context(
        frame_idx: int, height: int, width: int, device: torch.device
    ) -> dict:
        camera = camera_data[selected["cam_id"]]
        unique_img_idx = getattr(camera, "unique_img_idx", None)
        normalized_time = getattr(camera, "normalized_time", None)
        num_frames = int(getattr(camera, "num_frames", 1))
        idx = int(max(0, min(frame_idx, num_frames - 1)))
        context = {
            "frame_idx": torch.full(
                (height, width), idx, dtype=torch.long, device=device
            )
        }
        if unique_img_idx is not None:
            img_idx = unique_img_idx[idx].to(device=device, dtype=torch.long)
            context["img_idx"] = torch.full(
                (height, width), int(img_idx.item()), dtype=torch.long, device=device
            )
        if normalized_time is not None:
            normed_time = normalized_time[idx].to(device=device, dtype=torch.float32)
            context["normed_time"] = torch.ones(
                (height, width), dtype=torch.float32, device=device
            ) * normed_time
        return context

    return {
        "initial_camera": _camera_state_from_camera(camera_data[camera_list[0]], 0),
        "image_context_getter": get_context,
        "camera_snap_options": camera_options,
        "camera_snap_getter": get_camera_state,
        "camera_snap_setter": set_active_camera,
    }


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

    data_cfg = cfg.data
    data_cfg.viewer = True

    dataset = DrivingDataset(data_cfg=data_cfg)
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
    dataset_camera_tools = _dataset_camera_tools(dataset)

    create_gsplat_viewer(
        trainer,
        port=args.port,
        output_dir=args.output_dir,
        mode="rendering",
        num_frames=num_frames,
        initial_fps=float(getattr(cfg.render, "fps", 10.0)),
        initial_camera=dataset_camera_tools.get("initial_camera"),
        image_context_getter=dataset_camera_tools.get("image_context_getter"),
        camera_snap_options=dataset_camera_tools.get("camera_snap_options", ()),
        camera_snap_getter=dataset_camera_tools.get("camera_snap_getter"),
        camera_snap_setter=dataset_camera_tools.get("camera_snap_setter"),
    )
    print("Viewer running... Ctrl+C to exit.")
    time.sleep(100000)


if __name__ == "__main__":
    """
    # Use single GPU to view the scene
    CUDA_VISIBLE_DEVICES=0 python tools/viewer.py \\
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
    
    args = parser.parse_args()
    cli(main, args, verbose=True)
