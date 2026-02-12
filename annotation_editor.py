"""Interactive GUI for visualizing and correcting YOLO polygon annotations.

Usage:
    python annotation_editor.py --pred-dir /path/to/predictions

Controls:
    - Left/Right arrows or A/D: Navigate samples
    - Click on polygon: Select it (highlighted in white)
    - Delete/Backspace: Remove selected polygon
    - N: Start drawing new polygon (click points, Enter to finish)
    - Escape: Cancel drawing
    - S: Save current annotations
    - 1-4: Set class for new polygon (0=Whole Root, 1=Aerenchyma, 2=Outer Endo, 3=Inner Endo)
"""

import argparse
import copy
import sys
from pathlib import Path
from typing import List, Optional, Tuple

import cv2
import numpy as np
from PyQt5.QtCore import Qt, QPoint, QPointF, QRectF
from PyQt5.QtGui import (
    QImage, QPixmap, QPainter, QPen, QColor, QBrush,
    QPolygonF, QFont, QKeySequence, QTransform
)
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QLabel, QPushButton, QComboBox, QMessageBox, QShortcut,
    QSplitter, QFrame, QScrollArea, QStatusBar, QGroupBox,
    QRadioButton, QButtonGroup, QSizePolicy, QCheckBox
)

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from src.config import (
    ANNOTATED_CLASSES, CLASS_COLORS_RGB, DATA_DIR, IMAGE_DIR, ANNOTATION_DIR
)
from src.dataset import SampleRegistry
from src.preprocessing import load_sample_normalized, to_uint8
from src.annotation_utils import parse_yolo_annotations, polygon_to_mask


# Colors for visualization (QColor format)
CLASS_QCOLORS = {
    0: QColor(0, 0, 255, 180),      # Whole Root - Blue
    1: QColor(255, 255, 0, 180),    # Aerenchyma - Yellow
    2: QColor(0, 255, 0, 180),      # Outer Endodermis - Green
    3: QColor(255, 0, 0, 180),      # Inner Endodermis - Red
}

SELECTED_COLOR = QColor(255, 255, 255, 220)  # White for selected polygon


class PolygonCanvas(QWidget):
    """Canvas widget that displays an image with polygon overlays, zoom/pan, and interaction."""

    def __init__(self, title: str, parent=None, editable: bool = False):
        super().__init__(parent)
        self.title = title
        self.editable = editable
        self.base_image: Optional[np.ndarray] = None
        self.polygons: List[dict] = []  # [{class_id, polygon (Nx2 array)}]
        self.selected_idx: Optional[int] = None
        self.drawing_mode = False
        self.drawing_points: List[Tuple[int, int]] = []
        self.drawing_class = 1  # Default class for new polygons
        self.hidden_classes: set = set()  # class IDs to hide

        # Vertex editing state
        self.editing_mode = False
        self._dragging_vertex: Optional[int] = None  # index into selected polygon

        # Zoom / pan state (in image coordinates)
        self.zoom_level = 1.0
        self.pan_offset = [0.0, 0.0]  # image-space offset of the viewport center
        self._panning = False
        self._pan_start: Optional[QPoint] = None

        self.setMinimumSize(300, 300)
        self.setStyleSheet("border: 1px solid #555; background-color: #1a1a1a;")
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.setMouseTracking(True)

    # ── Zoom / pan helpers ────────────────────────────────────────────────

    def reset_view(self):
        """Reset zoom and pan to show the full image."""
        self.zoom_level = 1.0
        self.pan_offset = [0.0, 0.0]

    def _base_scale(self) -> float:
        """Scale factor to fit the full image in the widget (zoom=1)."""
        if self.base_image is None:
            return 1.0
        h, w = self.base_image.shape[:2]
        ws = (self.width() - 4) / w
        hs = (self.height() - 4) / h
        return min(ws, hs)

    def _effective_scale(self) -> float:
        return self._base_scale() * self.zoom_level

    def image_to_widget(self, ix: float, iy: float) -> QPointF:
        """Convert image coords → widget coords."""
        s = self._effective_scale()
        if self.base_image is None:
            return QPointF(0, 0)
        h, w = self.base_image.shape[:2]
        cx = self.width() / 2.0
        cy = self.height() / 2.0
        wx = cx + (ix - w / 2.0 - self.pan_offset[0]) * s
        wy = cy + (iy - h / 2.0 - self.pan_offset[1]) * s
        return QPointF(wx, wy)

    def widget_to_image(self, wx: float, wy: float) -> Tuple[float, float]:
        """Convert widget coords → image coords."""
        s = self._effective_scale()
        if self.base_image is None or s == 0:
            return (0.0, 0.0)
        h, w = self.base_image.shape[:2]
        cx = self.width() / 2.0
        cy = self.height() / 2.0
        ix = (wx - cx) / s + w / 2.0 + self.pan_offset[0]
        iy = (wy - cy) / s + h / 2.0 + self.pan_offset[1]
        return (ix, iy)

    def set_view(self, zoom: float, pan: List[float]):
        """Set zoom/pan from an external source (for synchronization)."""
        self.zoom_level = zoom
        self.pan_offset = list(pan)
        self.update()

    # ── Data setters ──────────────────────────────────────────────────────

    def set_image(self, img: np.ndarray):
        """Set the base image (H, W, 3) uint8 RGB."""
        self.base_image = img.copy()
        self._qimage = QImage(
            self.base_image.data, img.shape[1], img.shape[0],
            3 * img.shape[1], QImage.Format_RGB888
        )
        self.update()

    def set_polygons(self, polygons: List[dict]):
        """Set the list of polygons to display."""
        self.polygons = polygons
        self.selected_idx = None
        self.update()

    def update_display(self):
        self.update()

    # ── Paint ─────────────────────────────────────────────────────────────

    def paintEvent(self, event):
        """Draw image + polygons with current zoom/pan."""
        if self.base_image is None:
            return

        h, w = self.base_image.shape[:2]
        s = self._effective_scale()

        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)
        painter.setRenderHint(QPainter.SmoothPixmapTransform)

        # Fill background
        painter.fillRect(self.rect(), QColor(26, 26, 26))

        # Compute transform: image → widget
        origin = self.image_to_widget(0, 0)
        painter.translate(origin)
        painter.scale(s, s)

        # Draw image
        painter.drawImage(QRectF(0, 0, w, h), self._qimage)

        # Draw polygons
        for idx, poly_data in enumerate(self.polygons):
            class_id = poly_data["class_id"]
            polygon = poly_data["polygon"]

            # Skip hidden classes (but always show selected)
            if class_id in self.hidden_classes and idx != self.selected_idx:
                continue

            if idx == self.selected_idx:
                color = SELECTED_COLOR
                pen_width = max(1, int(3 / s))
            else:
                color = CLASS_QCOLORS.get(class_id, QColor(128, 128, 128, 180))
                pen_width = max(1, int(2 / s))

            qpoly = QPolygonF()
            for pt in polygon:
                qpoly.append(QPointF(float(pt[0]), float(pt[1])))

            painter.setPen(QPen(color, pen_width))
            fill_color = QColor(color)
            fill_color.setAlpha(60)
            painter.setBrush(QBrush(fill_color))
            painter.drawPolygon(qpoly)

        # Constant screen-pixel sizes for handles/dots (converted to image space)
        handle_screen_px = 5.0
        handle_r = handle_screen_px / s
        handle_pen = 1.5 / s

        # Draw vertex handles in edit mode
        if self.editing_mode and self.selected_idx is not None and 0 <= self.selected_idx < len(self.polygons):
            polygon = self.polygons[self.selected_idx]["polygon"]
            painter.setPen(QPen(QColor(255, 0, 255), handle_pen))
            painter.setBrush(QBrush(QColor(255, 0, 255, 160)))
            for pt in polygon:
                painter.drawEllipse(QPointF(float(pt[0]), float(pt[1])), handle_r, handle_r)

        # Draw points being drawn
        if self.drawing_mode and self.drawing_points:
            painter.setPen(QPen(QColor(255, 0, 255), handle_pen))
            painter.setBrush(QBrush(QColor(255, 0, 255, 80)))

            for pt in self.drawing_points:
                painter.drawEllipse(QPointF(pt[0], pt[1]), handle_r, handle_r)

            if len(self.drawing_points) > 1:
                for i in range(len(self.drawing_points) - 1):
                    p1 = self.drawing_points[i]
                    p2 = self.drawing_points[i + 1]
                    painter.drawLine(QPointF(p1[0], p1[1]), QPointF(p2[0], p2[1]))

        painter.resetTransform()

        # Draw title (top-left, in widget space)
        painter.setPen(QPen(Qt.white))
        painter.setFont(QFont("Arial", 12, QFont.Bold))
        painter.drawText(10, 25, self.title)

        # Draw selected polygon label (top-center, in widget space)
        if self.selected_idx is not None and 0 <= self.selected_idx < len(self.polygons):
            poly_data = self.polygons[self.selected_idx]
            class_id = poly_data["class_id"]
            class_name = ANNOTATED_CLASSES.get(class_id, f"Class {class_id}")
            color = CLASS_COLORS_RGB.get(class_id, (128, 128, 128))
            painter.setPen(QPen(QColor(color[0], color[1], color[2])))
            painter.setFont(QFont("Arial", 14, QFont.Bold))
            fm = painter.fontMetrics()
            text_w = fm.horizontalAdvance(class_name)
            painter.drawText((self.width() - text_w) // 2, 25, class_name)

        painter.end()

    def resizeEvent(self, event):
        super().resizeEvent(event)
        self.update()

    # ── Mouse events ──────────────────────────────────────────────────────

    def wheelEvent(self, event):
        """Zoom in/out centered on mouse position."""
        if self.base_image is None:
            return

        # Get image coords under mouse before zoom
        mx, my = event.pos().x(), event.pos().y()
        ix, iy = self.widget_to_image(mx, my)

        # Zoom
        factor = 1.15 if event.angleDelta().y() > 0 else 1.0 / 1.15
        new_zoom = max(0.5, min(self.zoom_level * factor, 40.0))
        self.zoom_level = new_zoom

        # Adjust pan so the image point under mouse stays under mouse
        ix2, iy2 = self.widget_to_image(mx, my)
        self.pan_offset[0] -= (ix2 - ix)
        self.pan_offset[1] -= (iy2 - iy)

        self.update()
        self._sync_view()

    def mousePressEvent(self, event):
        if self.base_image is None:
            return

        if event.button() == Qt.MiddleButton or event.button() == Qt.RightButton:
            self._panning = True
            self._pan_start = event.pos()
            self.setCursor(Qt.ClosedHandCursor)
            return

        if event.button() != Qt.LeftButton:
            return

        ix, iy = self.widget_to_image(event.pos().x(), event.pos().y())
        img_x, img_y = int(ix), int(iy)
        h, w = self.base_image.shape[:2]

        if not (0 <= img_x < w and 0 <= img_y < h):
            return

        # Vertex editing mode: try to grab a handle
        if self.editable and self.editing_mode and self.selected_idx is not None:
            polygon = self.polygons[self.selected_idx]["polygon"]
            s = self._effective_scale()
            grab_radius = 8.0 / s  # constant screen-pixel grab area
            for vi, pt in enumerate(polygon):
                if abs(ix - pt[0]) < grab_radius and abs(iy - pt[1]) < grab_radius:
                    self._dragging_vertex = vi
                    # Push undo snapshot before the drag mutates anything
                    parent = self.get_editor_parent()
                    if parent:
                        parent.push_undo()
                    return
            # Clicked away from any handle — deselect
            self._dragging_vertex = None
            return

        if self.editable and self.drawing_mode:
            self.drawing_points.append((img_x, img_y))
            self.update()
        elif self.editable:
            self.selected_idx = self.find_polygon_at(img_x, img_y)
            self.update()

            if self.selected_idx is not None:
                parent = self.get_editor_parent()
                if parent:
                    poly = self.polygons[self.selected_idx]
                    cname = ANNOTATED_CLASSES.get(poly['class_id'], 'Unknown')
                    parent.update_status(f"Selected polygon {self.selected_idx} ({cname})")

    def mouseMoveEvent(self, event):
        # Vertex dragging
        if self._dragging_vertex is not None and self.editing_mode and self.selected_idx is not None:
            ix, iy = self.widget_to_image(event.pos().x(), event.pos().y())
            h, w = self.base_image.shape[:2]
            ix = max(0, min(ix, w - 1))
            iy = max(0, min(iy, h - 1))
            self.polygons[self.selected_idx]["polygon"][self._dragging_vertex] = [ix, iy]
            self.update()
            return

        if self._panning and self._pan_start is not None:
            s = self._effective_scale()
            if s == 0:
                return
            dx = (event.pos().x() - self._pan_start.x()) / s
            dy = (event.pos().y() - self._pan_start.y()) / s
            self.pan_offset[0] -= dx
            self.pan_offset[1] -= dy
            self._pan_start = event.pos()
            self.update()
            self._sync_view()

    def mouseReleaseEvent(self, event):
        if self._dragging_vertex is not None and event.button() == Qt.LeftButton:
            self._dragging_vertex = None
            # Mark modified
            parent = self.get_editor_parent()
            if parent:
                parent.modified = True
            return

        if event.button() in (Qt.MiddleButton, Qt.RightButton) and self._panning:
            self._panning = False
            self._pan_start = None
            self.setCursor(Qt.ArrowCursor)

    def find_polygon_at(self, x: int, y: int) -> Optional[int]:
        """Find the smallest-area polygon containing (x, y). Returns index or None."""
        best_idx = None
        best_area = float("inf")

        for idx, poly_data in enumerate(self.polygons):
            polygon = poly_data["polygon"]
            # cv2.pointPolygonTest: >0 means inside
            pts = polygon.reshape((-1, 1, 2)).astype(np.float32)
            dist = cv2.pointPolygonTest(pts, (float(x), float(y)), False)
            if dist >= 0:
                area = cv2.contourArea(pts)
                if area < best_area:
                    best_area = area
                    best_idx = idx

        return best_idx

    def _sync_view(self):
        """Notify parent to synchronize zoom/pan across all canvases."""
        parent = self.get_editor_parent()
        if parent:
            parent.sync_view(self)

    def get_editor_parent(self):
        """Get the AnnotationEditor parent window."""
        parent = self.parent()
        while parent is not None:
            if isinstance(parent, AnnotationEditor):
                return parent
            parent = parent.parent()
        return None

    def start_drawing(self, class_id: int):
        """Start drawing a new polygon."""
        self.drawing_mode = True
        self.drawing_class = class_id
        self.drawing_points = []
        self.selected_idx = None
        self.update()

    def finish_drawing(self) -> Optional[dict]:
        """Finish drawing and return the new polygon dict, or None if cancelled."""
        if len(self.drawing_points) >= 3:
            polygon = np.array(self.drawing_points, dtype=np.float32)
            result = {"class_id": self.drawing_class, "polygon": polygon}
            self.drawing_mode = False
            self.drawing_points = []
            return result
        self.cancel_drawing()
        return None

    def cancel_drawing(self):
        """Cancel current drawing."""
        self.drawing_mode = False
        self.drawing_points = []
        self.update()

    def delete_selected(self) -> Optional[dict]:
        """Delete the selected polygon and return it."""
        if self.selected_idx is not None and 0 <= self.selected_idx < len(self.polygons):
            removed = self.polygons.pop(self.selected_idx)
            self.selected_idx = None
            self.update()
            return removed
        return None


class AnnotationEditor(QMainWindow):
    """Main window for annotation editing."""

    def __init__(self, pred_dir: Optional[Path] = None):
        super().__init__()
        self.pred_dir = pred_dir
        self.registry = SampleRegistry()
        self.samples = self.registry.samples
        self.current_idx = 0
        self.modified = False
        self._undo_stack: List[List[dict]] = []
        self._redo_stack: List[List[dict]] = []

        self.setWindowTitle("Plant Root Annotation Editor")
        self.setMinimumSize(1400, 800)
        self.setup_ui()
        self.setup_shortcuts()

        if self.samples:
            self.load_sample(0)
        else:
            QMessageBox.warning(self, "No Samples", "No annotated samples found!")

    def setup_ui(self):
        central = QWidget()
        self.setCentralWidget(central)
        main_layout = QVBoxLayout(central)

        # Top controls
        controls = QHBoxLayout()

        # Navigation
        nav_group = QGroupBox("Navigation")
        nav_layout = QHBoxLayout(nav_group)

        self.prev_btn = QPushButton("< Prev (A)")
        self.prev_btn.clicked.connect(self.prev_sample)
        nav_layout.addWidget(self.prev_btn)

        self.sample_combo = QComboBox()
        self.sample_combo.setMinimumWidth(300)
        self.sample_combo.setMaxVisibleItems(20)
        self.sample_combo.view().setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOn)
        for i, s in enumerate(self.samples):
            self.sample_combo.addItem(f"{i+1}. {s.uid}")
        self.sample_combo.currentIndexChanged.connect(self.on_combo_change)
        nav_layout.addWidget(self.sample_combo)

        self.next_btn = QPushButton("Next (D) >")
        self.next_btn.clicked.connect(self.next_sample)
        nav_layout.addWidget(self.next_btn)

        self.sample_label = QLabel(f"1 / {len(self.samples)}")
        nav_layout.addWidget(self.sample_label)

        controls.addWidget(nav_group)

        # Class selection for drawing
        class_group = QGroupBox("New Polygon Class")
        class_layout = QHBoxLayout(class_group)
        self.class_buttons = QButtonGroup(self)

        for cid, cname in ANNOTATED_CLASSES.items():
            rb = QRadioButton(f"{cid}: {cname}")
            if cid == 1:  # Default to Aerenchyma
                rb.setChecked(True)
            self.class_buttons.addButton(rb, cid)
            class_layout.addWidget(rb)

        controls.addWidget(class_group)

        # Action buttons
        action_group = QGroupBox("Actions")
        action_layout = QHBoxLayout(action_group)

        self.draw_btn = QPushButton("Draw New (N)")
        self.draw_btn.clicked.connect(self.toggle_drawing)
        action_layout.addWidget(self.draw_btn)

        self.edit_btn = QPushButton("Edit (E)")
        self.edit_btn.clicked.connect(self.toggle_editing)
        action_layout.addWidget(self.edit_btn)

        self.delete_btn = QPushButton("Delete Selected (Del)")
        self.delete_btn.clicked.connect(self.delete_selected)
        action_layout.addWidget(self.delete_btn)

        self.save_btn = QPushButton("Save (S)")
        self.save_btn.clicked.connect(self.save_annotations)
        action_layout.addWidget(self.save_btn)

        self.copy_pred_btn = QPushButton("Copy Pred->GT (C)")
        self.copy_pred_btn.clicked.connect(self.copy_predictions_to_gt)
        action_layout.addWidget(self.copy_pred_btn)

        controls.addWidget(action_group)

        # Class visibility toggles
        vis_group = QGroupBox("Visibility")
        vis_layout = QHBoxLayout(vis_group)
        self.vis_checks = {}
        for cid, cname in ANNOTATED_CLASSES.items():
            color = CLASS_COLORS_RGB.get(cid, (128, 128, 128))
            cb = QCheckBox(cname)
            cb.setChecked(True)
            cb.setStyleSheet(f"color: rgb({color[0]},{color[1]},{color[2]}); font-weight: bold;")
            cb.toggled.connect(lambda checked, c=cid: self.toggle_class_visibility(c, checked))
            vis_layout.addWidget(cb)
            self.vis_checks[cid] = cb

        controls.addWidget(vis_group)
        controls.addStretch()

        main_layout.addLayout(controls)

        # Image panels
        splitter = QSplitter(Qt.Horizontal)

        # Original image
        self.original_canvas = PolygonCanvas("Original Image", editable=False)
        splitter.addWidget(self.original_canvas)

        # Ground truth (editable)
        self.gt_canvas = PolygonCanvas("Ground Truth (Click to Edit)", editable=True)
        splitter.addWidget(self.gt_canvas)

        # Prediction
        self.pred_canvas = PolygonCanvas("Prediction", editable=False)
        splitter.addWidget(self.pred_canvas)

        # Equal panel sizes, prevent user from resizing unevenly
        splitter.setSizes([1, 1, 1])
        splitter.setChildrenCollapsible(False)
        main_layout.addWidget(splitter, stretch=1)

        # Status bar
        self.status_bar = QStatusBar()
        self.setStatusBar(self.status_bar)
        self.update_status("Ready. Use A/D to navigate, N to draw, Del to delete.")

        # Legend removed — visibility toggles in the toolbar serve as the legend

    def setup_shortcuts(self):
        # Navigation
        QShortcut(QKeySequence(Qt.Key_A), self, self.prev_sample)
        QShortcut(QKeySequence(Qt.Key_Left), self, self.prev_sample)
        QShortcut(QKeySequence(Qt.Key_D), self, self.next_sample)
        QShortcut(QKeySequence(Qt.Key_Right), self, self.next_sample)

        # Actions
        QShortcut(QKeySequence(Qt.Key_N), self, self.toggle_drawing)
        QShortcut(QKeySequence(Qt.Key_E), self, self.toggle_editing)
        QShortcut(QKeySequence(Qt.Key_Delete), self, self.delete_selected)
        QShortcut(QKeySequence(Qt.Key_Backspace), self, self.delete_selected)
        QShortcut(QKeySequence(Qt.Key_S), self, self.save_annotations)
        QShortcut(QKeySequence(Qt.Key_Escape), self, self.escape_action)
        QShortcut(QKeySequence(Qt.Key_Return), self, self.finish_drawing)
        QShortcut(QKeySequence(Qt.Key_Enter), self, self.finish_drawing)

        # Class shortcuts (1-4)
        for i in range(4):
            QShortcut(QKeySequence(Qt.Key_1 + i), self, lambda idx=i: self.set_drawing_class(idx))

        # Copy predictions to GT
        QShortcut(QKeySequence(Qt.Key_C), self, self.copy_predictions_to_gt)

        # Undo / Redo
        QShortcut(QKeySequence("Ctrl+Z"), self, self.undo)
        QShortcut(QKeySequence("Ctrl+Shift+Z"), self, self.redo)

    def load_sample(self, idx: int):
        """Load a sample and display it."""
        if self.modified:
            reply = QMessageBox.question(
                self, "Unsaved Changes",
                "You have unsaved changes. Save before switching?",
                QMessageBox.Save | QMessageBox.Discard | QMessageBox.Cancel
            )
            if reply == QMessageBox.Save:
                self.save_annotations()
            elif reply == QMessageBox.Cancel:
                self.sample_combo.blockSignals(True)
                self.sample_combo.setCurrentIndex(self.current_idx)
                self.sample_combo.blockSignals(False)
                return

        self.current_idx = idx
        self.modified = False
        self._undo_stack.clear()
        self._redo_stack.clear()
        sample = self.samples[idx]

        # Reset zoom/pan for all canvases
        for canvas in (self.original_canvas, self.gt_canvas, self.pred_canvas):
            canvas.reset_view()

        # Load and display original image
        img = load_sample_normalized(sample)
        img_uint8 = to_uint8(img)
        h, w = img.shape[:2]

        self.original_canvas.set_image(img_uint8)
        self.gt_canvas.set_image(img_uint8)
        self.pred_canvas.set_image(img_uint8)

        # Load ground truth annotations
        gt_polygons = parse_yolo_annotations(sample.annotation_path, w, h)
        self.gt_canvas.set_polygons(gt_polygons)
        self.original_canvas.set_polygons([])  # No polygons on original

        # Load predictions if available
        pred_polygons = []
        if self.pred_dir:
            pred_file = self.pred_dir / f"{sample.uid}.txt"
            if pred_file.exists():
                pred_polygons = parse_yolo_annotations(pred_file, w, h)
        self.pred_canvas.set_polygons(pred_polygons)

        # Update UI
        self.sample_combo.blockSignals(True)
        self.sample_combo.setCurrentIndex(idx)
        self.sample_combo.blockSignals(False)
        self.sample_label.setText(f"{idx + 1} / {len(self.samples)}")
        self.update_status(f"Loaded: {sample.uid}")

    def on_combo_change(self, idx: int):
        if idx != self.current_idx:
            self.load_sample(idx)

    def prev_sample(self):
        if self.current_idx > 0:
            self.load_sample(self.current_idx - 1)

    def next_sample(self):
        if self.current_idx < len(self.samples) - 1:
            self.load_sample(self.current_idx + 1)

    def set_drawing_class(self, class_id: int):
        """Set the class for new polygons."""
        btn = self.class_buttons.button(class_id)
        if btn:
            btn.setChecked(True)
        self.update_status(f"Drawing class set to {class_id}: {ANNOTATED_CLASSES.get(class_id, 'Unknown')}")

    def toggle_drawing(self):
        """Toggle drawing mode on/off."""
        if self.gt_canvas.drawing_mode:
            # Cancel drawing and revert button
            self.gt_canvas.cancel_drawing()
            self._exit_editing()
            self.draw_btn.setStyleSheet("")
            self.update_status("Drawing cancelled")
        else:
            # Enter drawing mode
            self._exit_editing()
            class_id = self.class_buttons.checkedId()
            self.gt_canvas.start_drawing(class_id)
            self.draw_btn.setStyleSheet("background-color: #cc3333; color: white;")
            self.update_status(f"Drawing mode: Click to add points for class {class_id}. Press Enter to finish, Escape to cancel.")

    def _update_draw_btn(self):
        """Reset draw button style when drawing ends via Enter/Escape."""
        if not self.gt_canvas.drawing_mode:
            self.draw_btn.setStyleSheet("")

    def finish_drawing(self):
        """Finish drawing the current polygon."""
        if self.gt_canvas.drawing_mode:
            result = self.gt_canvas.finish_drawing()
            self._update_draw_btn()
            if result:
                self.push_undo()
                self.gt_canvas.polygons.append(result)
                self.gt_canvas.update_display()
                self.modified = True
                self.update_status(f"Added new polygon (class {result['class_id']})")
            else:
                self.update_status("Drawing cancelled (need at least 3 points)")

    def cancel_drawing(self):
        """Cancel current drawing."""
        if self.gt_canvas.drawing_mode:
            self.gt_canvas.cancel_drawing()
            self._update_draw_btn()
            self.update_status("Drawing cancelled")

    def toggle_editing(self):
        """Toggle vertex editing mode on/off."""
        if self.gt_canvas.editing_mode:
            self._exit_editing()
        else:
            if self.gt_canvas.selected_idx is None:
                self.update_status("Select a polygon first, then press E to edit vertices")
                return
            # Exit drawing if active
            if self.gt_canvas.drawing_mode:
                self.gt_canvas.cancel_drawing()
                self.draw_btn.setStyleSheet("")
            self.gt_canvas.editing_mode = True
            self.edit_btn.setStyleSheet("background-color: #cc3333; color: white;")
            self.gt_canvas.update()
            self.update_status("Edit mode: Drag vertices to reshape. Press E or Escape to exit.")

    def _exit_editing(self):
        """Exit vertex editing mode."""
        self.gt_canvas.editing_mode = False
        self.gt_canvas._dragging_vertex = None
        self.edit_btn.setStyleSheet("")
        self.gt_canvas.update()

    def escape_action(self):
        """Handle Escape: exit drawing, editing, or deselect."""
        if self.gt_canvas.drawing_mode:
            self.cancel_drawing()
        elif self.gt_canvas.editing_mode:
            self._exit_editing()
            self.update_status("Exited edit mode")
        else:
            self.gt_canvas.selected_idx = None
            self.gt_canvas.update()

    def push_undo(self):
        """Snapshot current GT polygons onto undo stack."""
        self._undo_stack.append(copy.deepcopy(self.gt_canvas.polygons))
        self._redo_stack.clear()

    def undo(self):
        """Restore the previous polygon state, preserving current mode."""
        if not self._undo_stack:
            self.update_status("Nothing to undo")
            return
        was_editing = self.gt_canvas.editing_mode
        selected_idx = self.gt_canvas.selected_idx
        # Save current state for redo
        self._redo_stack.append(copy.deepcopy(self.gt_canvas.polygons))
        self.gt_canvas.polygons = self._undo_stack.pop()
        self.gt_canvas._dragging_vertex = None
        # Keep editing mode and selection if the index is still valid
        if was_editing and selected_idx is not None and selected_idx < len(self.gt_canvas.polygons):
            self.gt_canvas.selected_idx = selected_idx
        else:
            self.gt_canvas.selected_idx = None
            if was_editing:
                self._exit_editing()
        self.gt_canvas.update()
        self.modified = bool(self._undo_stack)
        self.update_status("Undo")

    def redo(self):
        """Re-apply the last undone action, preserving current mode."""
        if not self._redo_stack:
            self.update_status("Nothing to redo")
            return
        was_editing = self.gt_canvas.editing_mode
        selected_idx = self.gt_canvas.selected_idx
        self._undo_stack.append(copy.deepcopy(self.gt_canvas.polygons))
        self.gt_canvas.polygons = self._redo_stack.pop()
        self.gt_canvas._dragging_vertex = None
        if was_editing and selected_idx is not None and selected_idx < len(self.gt_canvas.polygons):
            self.gt_canvas.selected_idx = selected_idx
        else:
            self.gt_canvas.selected_idx = None
            if was_editing:
                self._exit_editing()
        self.gt_canvas.update()
        self.modified = True
        self.update_status("Redo")

    def toggle_class_visibility(self, class_id: int, visible: bool):
        """Show or hide polygons of a given class on all canvases."""
        for canvas in (self.original_canvas, self.gt_canvas, self.pred_canvas):
            if visible:
                canvas.hidden_classes.discard(class_id)
            else:
                canvas.hidden_classes.add(class_id)
            canvas.update()

    def delete_selected(self):
        """Delete the currently selected polygon."""
        if self.gt_canvas.selected_idx is not None:
            self.push_undo()
        removed = self.gt_canvas.delete_selected()
        if removed:
            self.modified = True
            self.update_status(f"Deleted polygon (class {removed['class_id']})")
        else:
            self.update_status("No polygon selected to delete")

    def save_annotations(self):
        """Save current polygons back to annotation file."""
        if not self.samples:
            return

        sample = self.samples[self.current_idx]
        img = load_sample_normalized(sample)
        h, w = img.shape[:2]

        # Convert polygons back to YOLO format
        lines = []
        for poly_data in self.gt_canvas.polygons:
            class_id = poly_data["class_id"]
            polygon = poly_data["polygon"]

            # Normalize coordinates
            coords = []
            for pt in polygon:
                coords.append(f"{pt[0] / w:.6f}")
                coords.append(f"{pt[1] / h:.6f}")

            line = f"{class_id} " + " ".join(coords)
            lines.append(line)

        # Write to file
        with open(sample.annotation_path, "w") as f:
            f.write("\n".join(lines))

        self.modified = False
        self.update_status(f"Saved {len(lines)} polygons to {sample.annotation_path.name}")

    def sync_view(self, source: PolygonCanvas):
        """Synchronize zoom/pan from source canvas to all other canvases."""
        zoom = source.zoom_level
        pan = source.pan_offset
        for canvas in (self.original_canvas, self.gt_canvas, self.pred_canvas):
            if canvas is not source:
                canvas.set_view(zoom, pan)

    def copy_predictions_to_gt(self):
        """Copy all predictions to ground truth."""
        if not self.pred_canvas.polygons:
            self.update_status("No predictions to copy")
            return

        reply = QMessageBox.question(
            self, "Copy Predictions",
            f"Replace GT with {len(self.pred_canvas.polygons)} prediction polygons?",
            QMessageBox.Yes | QMessageBox.No
        )
        if reply == QMessageBox.Yes:
            self.push_undo()
            self.gt_canvas.polygons = copy.deepcopy(self.pred_canvas.polygons)
            self.gt_canvas.selected_idx = None
            self.gt_canvas.update_display()
            self.modified = True
            self.update_status(f"Copied {len(self.pred_canvas.polygons)} predictions to GT")

    def update_status(self, msg: str):
        """Update status bar message."""
        self.status_bar.showMessage(msg)

    def closeEvent(self, event):
        """Handle window close."""
        if self.modified:
            reply = QMessageBox.question(
                self, "Unsaved Changes",
                "You have unsaved changes. Save before closing?",
                QMessageBox.Save | QMessageBox.Discard | QMessageBox.Cancel
            )
            if reply == QMessageBox.Save:
                self.save_annotations()
                event.accept()
            elif reply == QMessageBox.Discard:
                event.accept()
            else:
                event.ignore()
        else:
            event.accept()


def main():
    parser = argparse.ArgumentParser(
        description="Interactive annotation editor for YOLO polygon annotations",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Controls:
  A / Left Arrow    Previous sample
  D / Right Arrow   Next sample
  N                 Start drawing new polygon
  Enter             Finish drawing
  Escape            Cancel drawing
  Delete/Backspace  Delete selected polygon
  S                 Save changes
  C                 Copy predictions to ground truth
  1-4               Set class for new polygon

Classes:
  0: Whole Root (Blue)
  1: Aerenchyma (Yellow)
  2: Outer Endodermis (Green)
  3: Inner Endodermis (Red)
"""
    )
    parser.add_argument(
        "--pred-dir", type=str, default=None,
        help="Directory containing prediction .txt files (YOLO format)"
    )
    parser.add_argument(
        "--data-dir", type=str, default=None,
        help="Override data directory (default: project data/)"
    )
    args = parser.parse_args()

    # Override paths if specified
    if args.data_dir:
        import src.config as config
        data_path = Path(args.data_dir)
        config.DATA_DIR = data_path
        config.IMAGE_DIR = data_path / "image"
        config.ANNOTATION_DIR = data_path / "annotation"

    pred_dir = Path(args.pred_dir) if args.pred_dir else None

    app = QApplication(sys.argv)
    app.setStyle("Fusion")

    # Dark theme
    palette = app.palette()
    palette.setColor(palette.Window, QColor(53, 53, 53))
    palette.setColor(palette.WindowText, Qt.white)
    palette.setColor(palette.Base, QColor(25, 25, 25))
    palette.setColor(palette.AlternateBase, QColor(53, 53, 53))
    palette.setColor(palette.ToolTipBase, Qt.white)
    palette.setColor(palette.ToolTipText, Qt.white)
    palette.setColor(palette.Text, Qt.white)
    palette.setColor(palette.Button, QColor(53, 53, 53))
    palette.setColor(palette.ButtonText, Qt.white)
    palette.setColor(palette.BrightText, Qt.red)
    palette.setColor(palette.Highlight, QColor(42, 130, 218))
    palette.setColor(palette.HighlightedText, Qt.black)
    app.setPalette(palette)

    editor = AnnotationEditor(pred_dir=pred_dir)
    editor.show()

    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
