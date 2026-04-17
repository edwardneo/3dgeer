import viser
from pathlib import Path
from typing import Literal
from typing import Tuple, Callable
from nerfview import Viewer, RenderTabState


class GsplatRenderTabState(RenderTabState):
    # non-controlable parameters
    total_gs_count: int = 0
    rendered_gs_count: int = 0

    # controlable parameters
    max_sh_degree: int = 5
    near_plane: float = 1e-2
    far_plane: float = 1e2
    radius_clip: float = 0.0
    eps2d: float = 0.3
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


class GsplatViewer(Viewer):
    """
    Viewer for gsplat.
    """

    def __init__(
        self,
        server: viser.ViserServer,
        render_fn: Callable,
        output_dir: Path,
        mode: Literal["rendering", "training"] = "rendering",
    ):
        super().__init__(server, render_fn, output_dir, mode)
        server.gui.set_panel_label("gsplat viewer")

    def _init_rendering_tab(self):
        self.render_tab_state = GsplatRenderTabState()
        self._rendering_tab_handles = {}
        self._rendering_folder = self.server.gui.add_folder("Rendering")

    def _populate_rendering_tab(self):
        server = self.server
        with self._rendering_folder:
            with server.gui.add_folder("Gsplat"):
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

                max_sh_degree_number = server.gui.add_number(
                    "Max SH",
                    initial_value=self.render_tab_state.max_sh_degree,
                    min=0,
                    max=5,
                    step=1,
                    hint="Maximum SH degree used",
                )

                @max_sh_degree_number.on_update
                def _(_) -> None:
                    self.render_tab_state.max_sh_degree = int(
                        max_sh_degree_number.value
                    )
                    self.rerender(_)

                near_far_plane_vec2 = server.gui.add_vector2(
                    "Near/Far",
                    initial_value=(
                        self.render_tab_state.near_plane,
                        self.render_tab_state.far_plane,
                    ),
                    min=(1e-3, 1e1),
                    max=(1e1, 1e3),
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
                    hint="Epsilon added to the egienvalues of projected 2D covariance matrices.",
                )

                @eps2d_slider.on_update
                def _(_) -> None:
                    self.render_tab_state.eps2d = eps2d_slider.value
                    self.rerender(_)

                backgrounds_slider = server.gui.add_rgb(
                    "Background",
                    initial_value=self.render_tab_state.backgrounds,
                    hint="Background color for rendering.",
                )

                @backgrounds_slider.on_update
                def _(_) -> None:
                    self.render_tab_state.backgrounds = backgrounds_slider.value
                    self.rerender(_)

                render_mode_dropdown = server.gui.add_dropdown(
                    "Render Mode",
                    ("rgb", "depth(accumulated)", "depth(expected)", "alpha"),
                    initial_value=self.render_tab_state.render_mode,
                    hint="Render mode to use.",
                )

                @render_mode_dropdown.on_update
                def _(_) -> None:
                    if "depth" in render_mode_dropdown.value:
                        normalize_nearfar_checkbox.disabled = False
                        inverse_checkbox.disabled = False
                    else:
                        normalize_nearfar_checkbox.disabled = True
                        inverse_checkbox.disabled = True
                    self.render_tab_state.render_mode = render_mode_dropdown.value
                    self.rerender(_)

                normalize_nearfar_checkbox = server.gui.add_checkbox(
                    "Normalize Near/Far",
                    initial_value=self.render_tab_state.normalize_nearfar,
                    disabled=True,
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
                    disabled=True,
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
                    hint="Colormap used for rendering depth/alpha.",
                )

                @colormap_dropdown.on_update
                def _(_) -> None:
                    self.render_tab_state.colormap = colormap_dropdown.value
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

                camera_model_dropdown = server.gui.add_dropdown(
                    "Camera Model",
                    ("pinhole", "ortho", "fisheye"),
                    initial_value=self.render_tab_state.camera_model,
                    hint="Camera model used for rendering.",
                )

                @camera_model_dropdown.on_update
                def _(_) -> None:
                    self.render_tab_state.camera_model = camera_model_dropdown.value
                    _sync_distortion_ui_enabled()
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

                with server.gui.add_folder("Distortion Coefficients"):
                    radial_k1_slider = server.gui.add_slider(
                        "Radial k1",
                        min=-1.0,
                        max=1.0,
                        step=0.001,
                        initial_value=self.render_tab_state.radial_k1,
                        hint="Radial distortion coefficient k1.",
                    )
                    radial_k2_slider = server.gui.add_slider(
                        "Radial k2",
                        min=-1.0,
                        max=1.0,
                        step=0.001,
                        initial_value=self.render_tab_state.radial_k2,
                        hint="Radial distortion coefficient k2.",
                    )
                    radial_k3_slider = server.gui.add_slider(
                        "Radial k3",
                        min=-1.0,
                        max=1.0,
                        step=0.001,
                        initial_value=self.render_tab_state.radial_k3,
                        hint="Radial distortion coefficient k3.",
                    )
                    radial_k4_slider = server.gui.add_slider(
                        "Radial k4",
                        min=-1.0,
                        max=1.0,
                        step=0.001,
                        initial_value=self.render_tab_state.radial_k4,
                        hint="Radial distortion coefficient k4.",
                    )
                    radial_k5_slider = server.gui.add_slider(
                        "Radial k5",
                        min=-1.0,
                        max=1.0,
                        step=0.001,
                        initial_value=self.render_tab_state.radial_k5,
                        hint="Radial distortion coefficient k5 (pinhole only).",
                    )
                    radial_k6_slider = server.gui.add_slider(
                        "Radial k6",
                        min=-1.0,
                        max=1.0,
                        step=0.001,
                        initial_value=self.render_tab_state.radial_k6,
                        hint="Radial distortion coefficient k6 (pinhole only).",
                    )
                    tangential_p1_slider = server.gui.add_slider(
                        "Tangential p1",
                        min=-1.0,
                        max=1.0,
                        step=0.001,
                        initial_value=self.render_tab_state.tangential_p1,
                        hint="Tangential distortion coefficient p1 (pinhole only).",
                    )
                    tangential_p2_slider = server.gui.add_slider(
                        "Tangential p2",
                        min=-1.0,
                        max=1.0,
                        step=0.001,
                        initial_value=self.render_tab_state.tangential_p2,
                        hint="Tangential distortion coefficient p2 (pinhole only).",
                    )
                    thin_prism_s1_slider = server.gui.add_slider(
                        "Thin prism s1",
                        min=-1.0,
                        max=1.0,
                        step=0.001,
                        initial_value=self.render_tab_state.thin_prism_s1,
                        hint="Thin prism coefficient s1 (pinhole only).",
                    )
                    thin_prism_s2_slider = server.gui.add_slider(
                        "Thin prism s2",
                        min=-1.0,
                        max=1.0,
                        step=0.001,
                        initial_value=self.render_tab_state.thin_prism_s2,
                        hint="Thin prism coefficient s2 (pinhole only).",
                    )

                @radial_k1_slider.on_update
                def _(_) -> None:
                    self.render_tab_state.radial_k1 = float(radial_k1_slider.value)
                    self.rerender(_)

                @radial_k2_slider.on_update
                def _(_) -> None:
                    self.render_tab_state.radial_k2 = float(radial_k2_slider.value)
                    self.rerender(_)

                @radial_k3_slider.on_update
                def _(_) -> None:
                    self.render_tab_state.radial_k3 = float(radial_k3_slider.value)
                    self.rerender(_)

                @radial_k4_slider.on_update
                def _(_) -> None:
                    self.render_tab_state.radial_k4 = float(radial_k4_slider.value)
                    self.rerender(_)

                @radial_k5_slider.on_update
                def _(_) -> None:
                    self.render_tab_state.radial_k5 = float(radial_k5_slider.value)
                    self.rerender(_)

                @radial_k6_slider.on_update
                def _(_) -> None:
                    self.render_tab_state.radial_k6 = float(radial_k6_slider.value)
                    self.rerender(_)

                @tangential_p1_slider.on_update
                def _(_) -> None:
                    self.render_tab_state.tangential_p1 = float(tangential_p1_slider.value)
                    self.rerender(_)

                @tangential_p2_slider.on_update
                def _(_) -> None:
                    self.render_tab_state.tangential_p2 = float(tangential_p2_slider.value)
                    self.rerender(_)

                @thin_prism_s1_slider.on_update
                def _(_) -> None:
                    self.render_tab_state.thin_prism_s1 = float(thin_prism_s1_slider.value)
                    self.rerender(_)

                @thin_prism_s2_slider.on_update
                def _(_) -> None:
                    self.render_tab_state.thin_prism_s2 = float(thin_prism_s2_slider.value)
                    self.rerender(_)

                def _sync_distortion_ui_enabled() -> None:
                    # If rendering mode is standard, disable all distortion sliders.
                    if rendering_mode_dropdown.value == "standard":
                        radial_k1_slider.disabled = True
                        radial_k2_slider.disabled = True
                        radial_k3_slider.disabled = True
                        radial_k4_slider.disabled = True
                        radial_k5_slider.disabled = True
                        radial_k6_slider.disabled = True
                        tangential_p1_slider.disabled = True
                        tangential_p2_slider.disabled = True
                        thin_prism_s1_slider.disabled = True
                        thin_prism_s2_slider.disabled = True
                        return

                    # Otherwise: in pinhole/ortho enable all; in fisheye enable only k1..k4.
                    fisheye = camera_model_dropdown.value == "fisheye"
                    radial_k1_slider.disabled = False
                    radial_k2_slider.disabled = False
                    radial_k3_slider.disabled = False
                    radial_k4_slider.disabled = False
                    radial_k5_slider.disabled = fisheye
                    radial_k6_slider.disabled = fisheye
                    tangential_p1_slider.disabled = fisheye
                    tangential_p2_slider.disabled = fisheye
                    thin_prism_s1_slider.disabled = fisheye
                    thin_prism_s2_slider.disabled = fisheye

                _sync_distortion_ui_enabled()

        self._rendering_tab_handles.update(
            {
                "total_gs_count_number": total_gs_count_number,
                "rendered_gs_count_number": rendered_gs_count_number,
                "near_far_plane_vec2": near_far_plane_vec2,
                "radius_clip_slider": radius_clip_slider,
                "eps2d_slider": eps2d_slider,
                "backgrounds_slider": backgrounds_slider,
                "render_mode_dropdown": render_mode_dropdown,
                "normalize_nearfar_checkbox": normalize_nearfar_checkbox,
                "inverse_checkbox": inverse_checkbox,
                "colormap_dropdown": colormap_dropdown,
                "rasterize_mode_dropdown": rasterize_mode_dropdown,
                "camera_model_dropdown": camera_model_dropdown,
                "distortion_mode_dropdown": rendering_mode_dropdown,
                "radial_k1_slider": radial_k1_slider,
                "radial_k2_slider": radial_k2_slider,
                "radial_k3_slider": radial_k3_slider,
                "radial_k4_slider": radial_k4_slider,
                "radial_k5_slider": radial_k5_slider,
                "radial_k6_slider": radial_k6_slider,
                "tangential_p1_slider": tangential_p1_slider,
                "tangential_p2_slider": tangential_p2_slider,
                "thin_prism_s1_slider": thin_prism_s1_slider,
                "thin_prism_s2_slider": thin_prism_s2_slider,
            }
        )
        super()._populate_rendering_tab()

    def _after_render(self):
        # Update the GUI elements with current values
        self._rendering_tab_handles[
            "total_gs_count_number"
        ].value = self.render_tab_state.total_gs_count
        self._rendering_tab_handles[
            "rendered_gs_count_number"
        ].value = self.render_tab_state.rendered_gs_count
