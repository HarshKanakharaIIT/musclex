"""
Mask Tool for MuscleX

Reuses the existing MuscleX ImageViewerWidget and DisplayOptionsPanel.
Drawing is performed directly on the same Matplotlib canvas as the image.

Tools:
    - Rectangle
    - Oval
    - Polygon
    - Pencil
    - Eraser
    - Undo / Redo
    - Clear
    - Save / Load mask (.npy)

The existing ImageViewerWidget provides:
    - zoom / pan
    - intensity controls
    - grayscale / inverse grayscale
    - optional double zoom

The image coordinate system is intentionally the same as ImageViewerWidget:
event.xdata -> X pixel coordinate
event.ydata -> Y pixel coordinate
mask[y, x] -> image pixel
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
from matplotlib.colors import ListedColormap
from matplotlib.path import Path as MplPath
from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QApplication,
    QFileDialog,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)

from .widgets.image_viewer_widget import ImageViewerWidget

try:
    from musclex.ui.stylesheet import stylesheet
except ImportError:
    stylesheet = ""


class MaskImageViewer(ImageViewerWidget):
    """Existing MuscleX viewer with pan disabled while a mask tool is drawing."""

    def __init__(self, parent=None):
        super().__init__(
            parent=parent,
            show_display_panel=True,
            show_double_zoom=True,
        )
        self.mask_drawing_enabled = False

    def _handle_pan_start(self, event):
        if self.mask_drawing_enabled:
            return
        super()._handle_pan_start(event)

    def _handle_pan_drag(self, event):
        if self.mask_drawing_enabled:
            return
        super()._handle_pan_drag(event)


class MaskTool(QWidget):
    """
    Interactive image mask editor.

    Parameters
    ----------
    image : numpy.ndarray, optional
        2-D image to display.
    parent : QWidget, optional
        Qt parent.
    """

    def __init__(self, image=None, parent=None):
        super().__init__(parent)

        # Embeddable editor: parent dialog owns the window and Save action.
        if stylesheet:
            self.setStyleSheet(stylesheet)

        self.viewer = MaskImageViewer(self)

        self.mask = None
        self.undo_stack = []
        self.redo_stack = []

        self.current_tool = None
        self.start_xy = None
        self.current_xy = None
        # self.polygon_points = []
        # self.last_pencil_xy = None
        self.polygon_points = []
        self.last_pencil_xy = None

        # Polygon vertex editing
        self.dragging_vertex = None       # index of vertex being dragged
        self.completed_polygon = None     # most recently completed polygon
        self.completed_polygon_base = None  # mask before that polygon was applied

        self.mask_artist = None
        self.low_threshold_artist = None
        self.high_threshold_artist = None
        self.low_threshold_mask = None
        self.high_threshold_mask = None
        self.preview_artist = None

        self.brush_size = 10
        self.mask_alpha = 0.35

        self._build_ui()
        self._connect_signals()

        if image is not None:
            self.set_image(image)

    # ------------------------------------------------------------------
    # UI
    # ------------------------------------------------------------------

    def _build_ui(self):
        root = QVBoxLayout(self)
        root.setContentsMargins(6, 6, 6, 6)
        root.setSpacing(6)

        # Main viewer.
        root.addWidget(self.viewer, 1)

        # Existing DisplayOptionsPanel from ImageViewerWidget.
        # It contains intensity, colormap, zoom and double-zoom controls.
        if self.viewer.display_panel is not None:
            root.addWidget(self.viewer.display_panel)

        # Mask controls.
        controls = QHBoxLayout()
        controls.setSpacing(5)

        controls.addWidget(QLabel("Mask:"))

        self.rectangleButton = QPushButton("Rectangle")
        self.ovalButton = QPushButton("Oval")
        self.polygonButton = QPushButton("Polygon")
        self.pencilButton = QPushButton("Pencil")
        self.eraserButton = QPushButton("Eraser")

        self.tool_buttons = [
            self.rectangleButton,
            self.ovalButton,
            self.polygonButton,
            self.pencilButton,
            self.eraserButton,
        ]

        for button in self.tool_buttons:
            button.setCheckable(True)
            controls.addWidget(button)

        controls.addWidget(QLabel("Width:"))

        self.brushSpinBox = QSpinBox()
        self.brushSpinBox.setRange(1, 200)
        self.brushSpinBox.setValue(self.brush_size)
        self.brushSpinBox.setSuffix(" px")
        controls.addWidget(self.brushSpinBox)

        self.undoButton = QPushButton("Undo")
        self.redoButton = QPushButton("Redo")
        self.clearButton = QPushButton("Clear")
        # Saving is handled by the containing Set Image Mask dialog.
        self.saveButton = QPushButton("Save Mask")
        self.saveButton.setVisible(False)
        self.loadButton = QPushButton("Load Mask")

        controls.addWidget(self.undoButton)
        controls.addWidget(self.redoButton)
        controls.addWidget(self.clearButton)
        controls.addWidget(self.loadButton)

        root.addLayout(controls)

        self.statusLabel = QLabel(
            "Select a drawing tool. Use the existing viewer controls for zoom/intensity."
        )
        root.addWidget(self.statusLabel)

    def _connect_signals(self):
        self.rectangleButton.clicked.connect(
            lambda: self._select_tool("rectangle")
        )
        self.ovalButton.clicked.connect(lambda: self._select_tool("oval"))
        self.polygonButton.clicked.connect(lambda: self._select_tool("polygon"))
        self.pencilButton.clicked.connect(lambda: self._select_tool("pencil"))
        self.eraserButton.clicked.connect(lambda: self._select_tool("eraser"))

        self.undoButton.clicked.connect(self.undo)
        self.redoButton.clicked.connect(self.redo)
        self.clearButton.clicked.connect(self.clear_mask)
        self.loadButton.clicked.connect(self.load_mask)

        self.brushSpinBox.valueChanged.connect(self._brush_size_changed)

        # ImageViewerWidget emits these raw Matplotlib events before its
        # internal pan/tool handling.
        self.viewer.mousePressed.connect(self._mouse_pressed)
        self.viewer.mouseMoved.connect(self._mouse_moved)
        self.viewer.mouseReleased.connect(self._mouse_released)

        # display_image() and display settings changes can recreate the
        # axes image, so redraw our mask overlay afterward.
        self.viewer.displayOptionsChanged.connect(
            lambda _options: self._redraw_mask_overlay()
        )

    # ------------------------------------------------------------------
    # Image / mask setup
    # ------------------------------------------------------------------

    def set_image(self, image):
        image = np.asarray(image)

        if image.ndim != 2:
            raise ValueError(
                f"MaskTool expects a 2-D image, got shape {image.shape}"
            )

        self.viewer.display_image(image)

        self.mask = np.zeros(image.shape, dtype=bool)
        self.undo_stack.clear()
        self.redo_stack.clear()
        self.polygon_points = []

        self._redraw_mask_overlay()

        self.statusLabel.setText(
            f"Image: {image.shape[1]} x {image.shape[0]} pixels"
        )

    def set_threshold_masks(self, low_mask=None, high_mask=None):
        """Update live threshold overlays without changing the editable mask.

        Threshold masks use the dialog convention: 1=keep, 0=masked.
        """
        shape = self.mask.shape if self.mask is not None else None
        prepared = []
        for threshold_mask in (low_mask, high_mask):
            if threshold_mask is None:
                prepared.append(None)
                continue
            arr = np.asarray(threshold_mask)
            if shape is None or arr.shape != shape:
                raise ValueError(
                    f"Threshold mask shape {arr.shape} does not match image shape {shape}."
                )
            prepared.append(arr.astype(bool, copy=True))
        self.low_threshold_mask, self.high_threshold_mask = prepared
        self._redraw_mask_overlay()

    def get_mask(self):
        """Return the current boolean mask."""
        if self.mask is None:
            return None
        return self.mask.copy()

    def set_mask(self, mask):
        """Load an existing boolean/0-1 mask into the editor."""
        if self.mask is None:
            return
        mask = np.asarray(mask).astype(bool)
        if mask.shape != self.mask.shape:
            raise ValueError(
                f"Mask shape {mask.shape} does not match image shape {self.mask.shape}."
            )
        self.mask = mask.copy()
        self.undo_stack.clear()
        self.redo_stack.clear()
        self._clear_preview()
        self._redraw_mask_overlay()

    def has_mask(self):
        return self.mask is not None and bool(np.any(self.mask))

    # ------------------------------------------------------------------
    # Tool selection
    # ------------------------------------------------------------------

    def _select_tool(self, tool_name):
        # Clicking an already selected tool toggles it off.
        if self.current_tool == tool_name:
            self._deactivate_tool()
            return

        self._clear_preview()

        for button in self.tool_buttons:
            button.setChecked(False)

        button_map = {
            "rectangle": self.rectangleButton,
            "oval": self.ovalButton,
            "polygon": self.polygonButton,
            "pencil": self.pencilButton,
            "eraser": self.eraserButton,
        }

        button_map[tool_name].setChecked(True)

        self.current_tool = tool_name
        self.viewer.mask_drawing_enabled = True
        self.start_xy = None
        self.current_xy = None
        self.polygon_points = []
        self.last_pencil_xy = None

        if tool_name == "polygon":
            self.statusLabel.setText(
                "Polygon: click points; double-click or right-click to finish."
            )
        else:
            self.statusLabel.setText(
                f"{tool_name.capitalize()}: drag on the image."
            )

    def _deactivate_tool(self):
        self._clear_preview()
        self.current_tool = None
        self.viewer.mask_drawing_enabled = False
        self.start_xy = None
        self.current_xy = None
        self.polygon_points = []
        self.last_pencil_xy = None

        for button in self.tool_buttons:
            button.setChecked(False)

        self.statusLabel.setText("Drawing disabled.")

    # ------------------------------------------------------------------
    # Mouse handling
    # ------------------------------------------------------------------

    @staticmethod
    def _event_xy(event):
        if event.xdata is None or event.ydata is None:
            return None

        return float(event.xdata), float(event.ydata)

    def _inside_image(self, x, y):
        if self.mask is None:
            return False

        h, w = self.mask.shape
        return 0 <= x < w and 0 <= y < h

    def _clip_xy(self, x, y):
        if self.mask is None:
            return x, y

        h, w = self.mask.shape
        return (
            float(np.clip(x, 0, w - 1)),
            float(np.clip(y, 0, h - 1)),
        )

    def _find_polygon_vertex(self, event, points):
        """Return the index of a nearby vertex, or None."""
        if not points or event.x is None or event.y is None:
            return None

        ax = self.viewer.axes
        click_xy = np.array([event.x, event.y], dtype=float)

        # Compare in screen pixels so hit-testing works at different zooms.
        vertices_px = ax.transData.transform(np.asarray(points, dtype=float))
        distances = np.linalg.norm(vertices_px - click_xy, axis=1)

        idx = int(np.argmin(distances))
        return idx if distances[idx] <= 9 else None


    def _redraw_edited_polygon(self):
        """Rebuild the mask using the edited polygon vertices."""
        if self.completed_polygon_base is None:
            return

        self.mask = self.completed_polygon_base.copy()
        self.polygon_points = list(self.completed_polygon)

        self._apply_polygon()

        self.polygon_points = []
        self._redraw_mask_overlay()

        # Show the edited polygon's vertices again.
        self.polygon_points = list(self.completed_polygon)
        self._draw_polygon_preview()
        self.polygon_points = []
    
    def _mouse_pressed(self, event):
        if self.mask is None or self.current_tool is None:
            return

        xy = self._event_xy(event)
        if xy is None:
            return

        x, y = xy

        # # Polygon uses clicks rather than drag.
        # if self.current_tool == "polygon":
        #     if event.button == 3:
        #         self._finish_polygon()
        #         return

        #     if event.button == 1:
        #         self.polygon_points.append((x, y))
        #         self._draw_polygon_preview()
        #     return

        # Polygon supports both vertex dragging and point placement.
        if self.current_tool == "polygon":

            if event.button == 3:
                self._finish_polygon()
                return

            if event.button != 1:
                return

            # 1) Drag a vertex of the polygon currently being drawn.
            idx = self._find_polygon_vertex(event, self.polygon_points)
            if idx is not None:
                self.dragging_vertex = ("drawing", idx)
                return

            # 2) Drag a vertex of the most recently completed polygon.
            if self.completed_polygon is not None:
                idx = self._find_polygon_vertex(
                    event, self.completed_polygon
                )
                if idx is not None:
                    self._push_undo()
                    self.dragging_vertex = ("completed", idx)
                    return

            # 3) Otherwise, add a new polygon point.
            self.polygon_points.append((x, y))
            self._draw_polygon_preview()
            return

        if event.button != 1:
            return

        if not self._inside_image(x, y):
            return

        self._push_undo()

        self.start_xy = (x, y)
        self.current_xy = (x, y)

        if self.current_tool in ("pencil", "eraser"):
            self.last_pencil_xy = (x, y)
            self._draw_brush_segment(
                x,
                y,
                x,
                y,
                erase=self.current_tool == "eraser",
            )

    def _mouse_moved(self, event):
        if self.mask is None or self.current_tool is None:
            return

        xy = self._event_xy(event)
        if xy is None:
            return

        x, y = xy

        # if self.current_tool == "polygon":
        #     if self.polygon_points:
        #         self.current_xy = (x, y)
        #         self._draw_polygon_preview()
        #     return

        if self.current_tool == "polygon":

            # Dragging a vertex while drawing a polygon.
            if self.dragging_vertex is not None:
                mode, idx = self.dragging_vertex
                new_xy = self._clip_xy(x, y)

                if mode == "drawing":
                    self.polygon_points[idx] = new_xy
                    self._draw_polygon_preview()

                elif mode == "completed":
                    self.completed_polygon[idx] = new_xy
                    self._redraw_edited_polygon()

                return

            # Normal polygon preview while adding points.
            if self.polygon_points:
                self.current_xy = (x, y)
                self._draw_polygon_preview()

            return

        if self.start_xy is None:
            return

        self.current_xy = self._clip_xy(x, y)

        if self.current_tool in ("pencil", "eraser"):
            if self.last_pencil_xy is not None:
                self._draw_brush_segment(
                    self.last_pencil_xy[0],
                    self.last_pencil_xy[1],
                    self.current_xy[0],
                    self.current_xy[1],
                    erase=self.current_tool == "eraser",
                )

            self.last_pencil_xy = self.current_xy
            self._redraw_mask_overlay()
        else:
            self._draw_shape_preview()

    def _mouse_released(self, event):
        # Finish vertex dragging on mouse release.
        if self.current_tool == "polygon" and self.dragging_vertex is not None:
            self.dragging_vertex = None
            self.current_xy = None
            return
        if self.mask is None or self.current_tool is None:
            return

        if self.current_tool == "polygon":
            # Right-click is handled on press.
            return

        if event.button != 1 or self.start_xy is None:
            return

        xy = self._event_xy(event)
        if xy is not None:
            self.current_xy = self._clip_xy(*xy)

        if self.current_tool == "rectangle":
            self._apply_rectangle(self.start_xy, self.current_xy)

        elif self.current_tool == "oval":
            self._apply_oval(self.start_xy, self.current_xy)

        # Pencil/eraser are modified continuously during movement.

        self._clear_preview()
        self._redraw_mask_overlay()

        self.start_xy = None
        self.current_xy = None
        self.last_pencil_xy = None

    # ------------------------------------------------------------------
    # Raster drawing
    # ------------------------------------------------------------------

    def _push_undo(self):
        if self.mask is None:
            return

        self.undo_stack.append(self.mask.copy())

        # Avoid an unbounded memory footprint.
        if len(self.undo_stack) > 30:
            self.undo_stack.pop(0)

        self.redo_stack.clear()

    def _apply_rectangle(self, p1, p2):
        if self.mask is None:
            return

        x1, y1 = p1
        x2, y2 = p2

        xmin = max(0, int(np.floor(min(x1, x2))))
        xmax = min(self.mask.shape[1] - 1, int(np.ceil(max(x1, x2))))
        ymin = max(0, int(np.floor(min(y1, y2))))
        ymax = min(self.mask.shape[0] - 1, int(np.ceil(max(y1, y2))))

        if xmin <= xmax and ymin <= ymax:
            self.mask[ymin : ymax + 1, xmin : xmax + 1] = True

    def _apply_oval(self, p1, p2):
        if self.mask is None:
            return

        x1, y1 = p1
        x2, y2 = p2

        xmin = max(0, int(np.floor(min(x1, x2))))
        xmax = min(self.mask.shape[1] - 1, int(np.ceil(max(x1, x2))))
        ymin = max(0, int(np.floor(min(y1, y2))))
        ymax = min(self.mask.shape[0] - 1, int(np.ceil(max(y1, y2))))

        if xmin > xmax or ymin > ymax:
            return

        yy, xx = np.ogrid[ymin : ymax + 1, xmin : xmax + 1]

        cx = (x1 + x2) / 2.0
        cy = (y1 + y2) / 2.0
        rx = max(abs(x2 - x1) / 2.0, 0.5)
        ry = max(abs(y2 - y1) / 2.0, 0.5)

        ellipse = ((xx - cx) / rx) ** 2 + ((yy - cy) / ry) ** 2 <= 1.0
        self.mask[ymin : ymax + 1, xmin : xmax + 1] |= ellipse

    def _apply_polygon(self):
        if self.mask is None or len(self.polygon_points) < 3:
            return

        points = np.asarray(self.polygon_points, dtype=float)

        xmin = max(0, int(np.floor(points[:, 0].min())))
        xmax = min(self.mask.shape[1] - 1, int(np.ceil(points[:, 0].max())))
        ymin = max(0, int(np.floor(points[:, 1].min())))
        ymax = min(self.mask.shape[0] - 1, int(np.ceil(points[:, 1].max())))

        if xmin > xmax or ymin > ymax:
            return

        yy, xx = np.mgrid[ymin : ymax + 1, xmin : xmax + 1]

        coords = np.column_stack((xx.ravel(), yy.ravel()))
        inside = MplPath(points).contains_points(coords)

        region = inside.reshape(xx.shape)
        self.mask[ymin : ymax + 1, xmin : xmax + 1] |= region

    def _draw_brush_segment(self, x1, y1, x2, y2, erase=False):
        if self.mask is None:
            return

        h, w = self.mask.shape

        radius = max(0.5, self.brush_size / 2.0)

        length = max(abs(x2 - x1), abs(y2 - y1))
        steps = max(1, int(np.ceil(length)))

        for t in np.linspace(0.0, 1.0, steps + 1):
            x = x1 + t * (x2 - x1)
            y = y1 + t * (y2 - y1)

            xmin = max(0, int(np.floor(x - radius)))
            xmax = min(w - 1, int(np.ceil(x + radius)))
            ymin = max(0, int(np.floor(y - radius)))
            ymax = min(h - 1, int(np.ceil(y + radius)))

            if xmin > xmax or ymin > ymax:
                continue

            yy, xx = np.ogrid[ymin : ymax + 1, xmin : xmax + 1]
            disk = (xx - x) ** 2 + (yy - y) ** 2 <= radius ** 2

            if erase:
                self.mask[ymin : ymax + 1, xmin : xmax + 1][disk] = False
            else:
                self.mask[ymin : ymax + 1, xmin : xmax + 1][disk] = True

    # ------------------------------------------------------------------
    # Polygon / shape preview
    # ------------------------------------------------------------------

    def _draw_shape_preview(self):
        self._clear_preview()

        if self.start_xy is None or self.current_xy is None:
            return

        x1, y1 = self.start_xy
        x2, y2 = self.current_xy

        ax = self.viewer.axes

        if self.current_tool == "rectangle":
            from matplotlib.patches import Rectangle

            patch = Rectangle(
                (min(x1, x2), min(y1, y2)),
                abs(x2 - x1),
                abs(y2 - y1),
                fill=False,
                linewidth=1.5,
                edgecolor="red",
            )
            ax.add_patch(patch)
            self.preview_artist = patch

        elif self.current_tool == "oval":
            from matplotlib.patches import Ellipse

            patch = Ellipse(
                ((x1 + x2) / 2.0, (y1 + y2) / 2.0),
                abs(x2 - x1),
                abs(y2 - y1),
                fill=False,
                linewidth=1.5,
                edgecolor="red",
            )
            ax.add_patch(patch)
            self.preview_artist = patch

        self.viewer.canvas.draw_idle()

    def _draw_polygon_preview(self):
        self._clear_preview()

        if not self.polygon_points:
            return

        points = list(self.polygon_points)

        if self.current_xy is not None:
            points.append(self.current_xy)

        if len(points) < 2:
            return

        ax = self.viewer.axes
        xs = [p[0] for p in points]
        ys = [p[1] for p in points]

        (line,) = ax.plot(
            xs,
            ys,
            linewidth=1.5,
            color="red",
            marker="o",
            markersize=3,
        )

        self.preview_artist = line
        self.viewer.canvas.draw_idle()

    def _finish_polygon(self):
        if self.current_tool != "polygon":
            return

        if len(self.polygon_points) < 3:
            self.polygon_points = []
            self._clear_preview()
            return

        # self._push_undo()
        # self._apply_polygon()

        # self.polygon_points = []
        # self.current_xy = None
# Save the mask before applying this polygon.
        self._push_undo()
        self.completed_polygon_base = self.mask.copy()
        self.completed_polygon = list(self.polygon_points)

        self._apply_polygon()

        self.polygon_points = []
        self.current_xy = None

        self._clear_preview()
        self._redraw_mask_overlay()

        self.statusLabel.setText(
            "Polygon applied. Continue drawing or select another tool."
        )

    def _clear_preview(self):
        if self.preview_artist is not None:
            try:
                self.preview_artist.remove()
            except (ValueError, AttributeError):
                pass

        self.preview_artist = None
        self.viewer.canvas.draw_idle()

    # ------------------------------------------------------------------
    # Mask display
    # ------------------------------------------------------------------

    def _redraw_mask_overlay(self):
        if self.mask is None:
            return

        if self.mask_artist is not None:
            try:
                self.mask_artist.remove()
            except (ValueError, AttributeError):
                pass
            self.mask_artist = None

        if np.any(self.mask):
            # Transparent for False, visible red overlay for True.
            cmap = ListedColormap(
                [
                    (0.0, 0.0, 0.0, 0.0),
                    (1.0, 0.0, 0.0, self.mask_alpha),
                ]
            )
            self.mask_artist = self.viewer.axes.imshow(
                self.mask,
                cmap=cmap,
                interpolation="nearest",
                origin="upper",
                zorder=10,
            )

        # Threshold overlays are independent of the user's editable red mask.
        for attr in ("low_threshold_artist", "high_threshold_artist"):
            artist = getattr(self, attr, None)
            if artist is not None:
                try:
                    artist.remove()
                except (ValueError, AttributeError):
                    pass
                setattr(self, attr, None)

        for mask, color, attr in (
            (self.low_threshold_mask, (0.0, 1.0, 0.0, 0.38), "low_threshold_artist"),
            (self.high_threshold_mask, (0.0, 0.35, 1.0, 0.38), "high_threshold_artist"),
        ):
            if mask is None:
                continue
            rgba = np.zeros((*mask.shape, 4), dtype=float)
            rgba[~mask] = color
            if np.any(~mask):
                setattr(self, attr, self.viewer.axes.imshow(
                    rgba, interpolation="nearest", origin="upper", zorder=8
                ))

        self.viewer.canvas.draw_idle()

    # ------------------------------------------------------------------
    # Undo / redo / clear
    # ------------------------------------------------------------------

    def undo(self):
        if self.mask is None or not self.undo_stack:
            return

        self.redo_stack.append(self.mask.copy())
        self.mask = self.undo_stack.pop()

        self._clear_preview()
        self._redraw_mask_overlay()

    def redo(self):
        if self.mask is None or not self.redo_stack:
            return

        self.undo_stack.append(self.mask.copy())
        self.mask = self.redo_stack.pop()

        self._clear_preview()
        self._redraw_mask_overlay()

    def clear_mask(self):
        if self.mask is None:
            return

        if not np.any(self.mask):
            return

        self._push_undo()
        self.mask[:] = False

        self._clear_preview()
        self._redraw_mask_overlay()

    def _brush_size_changed(self, value):
        self.brush_size = int(value)

    # ------------------------------------------------------------------
    # Save / load
    # ------------------------------------------------------------------

    def save_mask(self):
        if self.mask is None:
            QMessageBox.warning(self, "Mask Tool", "No image is loaded.")
            return

        path, _ = QFileDialog.getSaveFileName(
            self,
            "Save Mask",
            "",
            "NumPy Mask (*.npy);;NumPy Archive (*.npz)",
        )

        if not path:
            return

        try:
            if path.lower().endswith(".npz"):
                np.savez_compressed(path, mask=self.mask.astype(np.uint8))
            else:
                if not path.lower().endswith(".npy"):
                    path += ".npy"
                np.save(path, self.mask.astype(np.uint8))

            self.statusLabel.setText(f"Mask saved: {path}")

        except Exception as exc:
            QMessageBox.critical(
                self,
                "Save Mask",
                f"Could not save mask:\n{exc}",
            )

    def load_mask(self):
        if self.mask is None:
            QMessageBox.warning(self, "Mask Tool", "No image is loaded.")
            return

        path, _ = QFileDialog.getOpenFileName(
            self,
            "Load Mask",
            "",
            "NumPy Mask (*.npy *.npz)",
        )

        if not path:
            return

        try:
            if path.lower().endswith(".npz"):
                data = np.load(path)
                if "mask" not in data:
                    raise ValueError("NPZ file does not contain a 'mask' array.")
                loaded = data["mask"]
            else:
                loaded = np.load(path)

            loaded = np.asarray(loaded).astype(bool)

            if loaded.shape != self.mask.shape:
                raise ValueError(
                    f"Mask shape {loaded.shape} does not match "
                    f"image shape {self.mask.shape}."
                )

            self._push_undo()
            self.mask = loaded.copy()

            self._clear_preview()
            self._redraw_mask_overlay()

            self.statusLabel.setText(f"Mask loaded: {path}")

        except Exception as exc:
            QMessageBox.critical(
                self,
                "Load Mask",
                f"Could not load mask:\n{exc}",
            )


# ----------------------------------------------------------------------
# Optional standalone test
# ----------------------------------------------------------------------

if __name__ == "__main__":
    import sys

    app = QApplication(sys.argv)

    # Optional test image.
    # Replace this with your MuscleX image when integrating.
    test_image = np.random.random((512, 512))

    window = MaskTool(image=test_image)
    window.show()

    sys.exit(app.exec())