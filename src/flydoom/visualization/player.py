"""PyVista neural-activity player and deterministic composite-frame renderer."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from flydoom.visualization.layout import load_swc
from flydoom.visualization.trace import EpisodeTrace

BACKGROUND = "#020b0d"


def activity_colors(values: np.ndarray) -> np.ndarray:
    """Map magnitude to dim blue-gray, cyan, then bright white."""
    magnitude = np.clip(np.abs(values), 0.0, 1.0)[:, None]
    resting = np.asarray([28, 54, 67], dtype=float)
    cyan = np.asarray([0, 225, 235], dtype=float)
    white = np.asarray([255, 255, 255], dtype=float)
    low = resting + np.minimum(magnitude * 2, 1) * (cyan - resting)
    high = cyan + np.maximum(magnitude * 2 - 1, 0) * (white - cyan)
    return np.where(magnitude <= 0.5, low, high).astype(np.uint8)


def _activity_geometry(pv, trace: EpisodeTrace, mask: np.ndarray, skeleton_dir: Path | None):
    owners = np.flatnonzero(mask)
    if skeleton_dir is None:
        return pv.PolyData(trace.positions[owners]), owners, False
    point_parts: list[np.ndarray] = []
    line_parts: list[np.ndarray] = []
    owner_parts: list[np.ndarray] = []
    offset = 0
    for owner in owners:
        path = skeleton_dir / f"{trace.body_ids[owner]}.swc"
        if not path.is_file():
            continue
        points, lines = load_swc(path)
        point_parts.append(points)
        line_parts.append(lines + offset)
        owner_parts.append(np.full(len(points), owner, dtype=np.int64))
        offset += len(points)
    if not point_parts:
        return pv.PolyData(trace.positions[owners]), owners, False
    points = np.concatenate(point_parts)
    centered = points - points.mean(axis=0, keepdims=True)
    scale = np.percentile(np.linalg.norm(centered, axis=1), 95)
    mesh = pv.PolyData(centered / max(float(scale), 1e-6))
    lines = np.concatenate(line_parts)
    mesh.lines = np.column_stack((np.full(len(lines), 2), lines)).ravel()
    return mesh, np.concatenate(owner_parts), True


class ActivityPlayer:
    """Interactive playback with scrub, speed, camera, filtering, and exports."""

    def __init__(
        self,
        trace: EpisodeTrace,
        *,
        region: str | None = None,
        type_: str | None = None,
        skeleton_dir: str | Path | None = None,
    ):
        try:
            import pyvista as pv
        except ImportError as exc:
            raise RuntimeError("Install the visualization extra: pip install -e '.[viz]'") from exc
        self.pv = pv
        self.trace = trace
        self.index = 0
        self.speed = 1.0
        self._phase = 0.0
        self.playing = False
        self.region = region
        self.type_ = type_
        self.skeleton_dir = Path(skeleton_dir) if skeleton_dir else None
        self.mask = self._mask()
        self.plotter = pv.Plotter(shape=(2, 1), window_size=(1280, 720))
        self.plotter.set_background(BACKGROUND)
        self.plotter.subplot(0, 0)
        self.frame_actor = self.plotter.add_mesh(
            self._frame_mesh(trace.frames[0]),
            texture=pv.numpy_to_texture(trace.frames[0]),
            lighting=False,
        )
        self.plotter.camera_position = "xy"
        self.plotter.subplot(1, 0)
        self.mesh, self.owners, self.as_lines = _activity_geometry(
            pv, trace, self.mask, self.skeleton_dir
        )
        self.mesh["activity"] = np.abs(trace.activity[0, self.owners])
        self.actor = self._add_activity_actor()

        self.status = self.plotter.add_text(
            "", position="lower_left", color="#c8fbff", font_size=11
        )
        self.plotter.add_slider_widget(
            lambda value: self.set_frame(round(value)),
            (0, len(trace.frames) - 1),
            value=0,
            title="time",
            pointa=(0.18, 0.03),
            pointb=(0.82, 0.03),
        )
        self.plotter.add_key_event("space", self.toggle)
        self.plotter.add_key_event("Right", lambda: self.set_frame(self.index + 1))
        self.plotter.add_key_event("Left", lambda: self.set_frame(self.index - 1))
        self.plotter.add_key_event("bracketright", lambda: self.set_speed(self.speed * 2))
        self.plotter.add_key_event("bracketleft", lambda: self.set_speed(self.speed / 2))
        self.plotter.add_key_event("c", lambda: self.plotter.camera.Azimuth(15))
        self.plotter.add_key_event("r", self.cycle_region)
        self.plotter.add_key_event("t", self.cycle_type)
        self.plotter.add_key_event("s", lambda: self.export_png("flydoom-frame.png"))
        self.set_frame(0)

    def _add_activity_actor(self):
        self.glow_actor = self.plotter.add_mesh(
            self.mesh,
            scalars="activity",
            cmap=["#1c3643", "#00e1eb", "#ffffff"],
            clim=(0, 1),
            point_size=15 if not self.as_lines else 2,
            line_width=7,
            render_points_as_spheres=not self.as_lines,
            opacity=0.10,
            emissive=True,
            show_scalar_bar=False,
        )
        return self.plotter.add_mesh(
            self.mesh,
            scalars="activity",
            cmap=["#1c3643", "#00e1eb", "#ffffff"],
            clim=(0, 1),
            point_size=7 if not self.as_lines else 2,
            line_width=2,
            render_points_as_spheres=not self.as_lines,
            emissive=True,
            show_scalar_bar=False,
        )

    def _mask(self) -> np.ndarray:
        mask = np.ones(self.trace.activity.shape[1], dtype=bool)
        if self.region:
            mask &= self.trace.regions == self.region
        if self.type_:
            mask &= self.trace.types == self.type_
        if not mask.any():
            raise ValueError("Region/type filters selected no neurons")
        return mask

    def _frame_mesh(self, frame: np.ndarray):
        height, width = frame.shape[:2]
        return self.pv.Plane(center=(0, 0, 0), direction=(0, 0, 1), i_size=width / height)

    def set_filter(self, *, region: str | None = None, type_: str | None = None) -> None:
        self.region, self.type_ = region, type_
        self.mask = self._mask()
        self.plotter.remove_actor(self.actor)
        self.plotter.remove_actor(self.glow_actor)
        self.mesh, self.owners, self.as_lines = _activity_geometry(
            self.pv, self.trace, self.mask, self.skeleton_dir
        )
        self.mesh["activity"] = np.abs(self.trace.activity[self.index, self.owners])
        self.actor = self._add_activity_actor()
        self.set_frame(self.index)

    def set_speed(self, speed: float) -> None:
        self.speed = float(np.clip(speed, 0.125, 8.0))
        self.set_frame(self.index)

    def cycle_region(self) -> None:
        values = [None, *sorted(set(self.trace.regions) - {"unknown"})]
        current = values.index(self.region) if self.region in values else 0
        self.set_filter(region=values[(current + 1) % len(values)], type_=self.type_)

    def cycle_type(self) -> None:
        values = [None, *sorted(set(self.trace.types) - {"unknown"})]
        current = values.index(self.type_) if self.type_ in values else 0
        self.set_filter(region=self.region, type_=values[(current + 1) % len(values)])

    def set_frame(self, index: int) -> None:
        self.index = int(np.clip(index, 0, len(self.trace.frames) - 1))
        values = np.abs(self.trace.activity[self.index, self.owners])
        if self.index:
            values = np.maximum(
                values, 0.45 * np.abs(self.trace.activity[self.index - 1, self.owners])
            )
        if self.index > 1:
            values = np.maximum(
                values, 0.20 * np.abs(self.trace.activity[self.index - 2, self.owners])
            )
        self.mesh["activity"] = values
        texture = self.pv.numpy_to_texture(self.trace.frames[self.index])
        self.frame_actor.SetTexture(texture)
        action = int(self.trace.actions[self.index])
        name = self.trace.action_name(action)
        self.status.set_text(
            "lower_left",
            f"t={self.index / self.trace.fps:6.2f}s   action={name}   "
            f"reward={self.trace.rewards[self.index]:+.3f}   speed={self.speed:g}x",
        )
        self.plotter.render()

    def toggle(self) -> None:
        self.playing = not self.playing

    def _tick(self) -> None:
        if self.playing:
            self._phase += self.speed
            if self._phase >= 1:
                advance = int(self._phase)
                self._phase -= advance
                self.set_frame((self.index + advance) % len(self.trace.frames))

    def show(self) -> None:
        self.plotter.add_timer_event(
            max_steps=2**31 - 1,
            duration=round(1000 / self.trace.fps),
            callback=lambda _step: self._tick(),
        )
        self.plotter.show()

    def export_png(self, path: str | Path) -> None:
        self.plotter.screenshot(path)


def render_brain_image(
    trace: EpisodeTrace,
    index: int,
    *,
    size: tuple[int, int] = (1280, 520),
    skeleton_dir: str | Path | None = None,
) -> np.ndarray:
    """Render one off-screen 3D activity panel with a subtle two-layer glow."""
    try:
        import pyvista as pv
    except ImportError as exc:
        raise RuntimeError("Install the visualization extra: pip install -e '.[viz]'") from exc
    plotter = pv.Plotter(off_screen=True, window_size=size)
    plotter.set_background(BACKGROUND)
    mesh, owners, as_lines = _activity_geometry(
        pv,
        trace,
        np.ones(trace.activity.shape[1], dtype=bool),
        Path(skeleton_dir) if skeleton_dir else None,
    )
    values = np.abs(trace.activity[index, owners])
    if index:
        values = np.maximum(values, 0.45 * np.abs(trace.activity[index - 1, owners]))
    if index > 1:
        values = np.maximum(values, 0.20 * np.abs(trace.activity[index - 2, owners]))
    mesh["activity"] = values
    plotter.add_mesh(
        mesh,
        scalars="activity",
        cmap=["#1c3643", "#00e1eb", "#ffffff"],
        clim=(0, 1),
        point_size=12 if not as_lines else 2,
        line_width=5,
        render_points_as_spheres=not as_lines,
        opacity=0.10,
        emissive=True,
        show_scalar_bar=False,
    )
    plotter.add_mesh(
        mesh,
        scalars="activity",
        cmap=["#1c3643", "#00e1eb", "#ffffff"],
        clim=(0, 1),
        point_size=4 if not as_lines else 1,
        line_width=1,
        render_points_as_spheres=not as_lines,
        emissive=True,
        show_scalar_bar=False,
    )
    plotter.camera.azimuth = index * 0.25
    image = plotter.screenshot(return_img=True)
    plotter.close()
    return image
