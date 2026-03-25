"""Interactive GUI for visualizing and correcting YOLO polygon annotations.

Usage:
    python polygon_editor.py [--data-dir /path/to/data]

Modes:
    1. Correct GT:    images + annotation/ + prediction/ — edit GT with predictions as reference
    2. Correct Pred:  images + prediction/               — edit predictions, save to annotation/
    3. Create GT:     images only                        — draw annotations from scratch

Controls:
    - Left/Right arrows or A/D: Navigate samples
    - Click on polygon: Select it (highlighted in white)
    - N: Start drawing new polygon with nodes (click points, connect with lines)
    - B: Enter brush mode (erase by default, Shift+paint to add area)
        - Scroll wheel: zoom in/out (normal zoom)
        - Ctrl+scroll: change brush size
    - V: Enter vertex (node) editing mode (drag, add, delete vertices)
    - Enter: Edit selected polygon with brush / confirm edits + auto-save
    - Enter: Confirm drawing/edits/brush
    - Escape: Cancel drawing/edits/brush (reverts changes)
    - R: Split selected ring polygon into outer + inner (endo/exo)
    - Shift+click (edit mode): Insert a free vertex at cursor position
    - Drag on empty space (edit mode): Rectangle-select multiple vertices
    - Shift+drag: Add to existing vertex selection
    - Delete/Backspace: Remove selected vertex(es) (edit mode) or polygon
    - S: Save current annotations to file
    - Tab: Copy selected prediction polygon to editable panel
    - C: Copy ALL predictions to editable panel
    - 0-5: Set class for new polygon
"""

import argparse
import copy
import json
import math
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple

import cv2
import numpy as np
from PyQt5.QtCore import Qt, QPoint, QPointF, QRectF, QTimer, QEvent
from PyQt5.QtGui import (
    QImage, QPixmap, QPainter, QPen, QColor, QBrush,
    QPolygonF, QFont, QKeySequence, QTransform, QIcon
)
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QLabel, QPushButton, QComboBox, QMessageBox, QShortcut,
    QSplitter, QFrame, QScrollArea, QStatusBar, QGroupBox,
    QRadioButton, QButtonGroup, QSizePolicy, QCheckBox, QStackedWidget,
    QFileDialog, QLineEdit, QSlider, QStyle, QProgressDialog,
    QToolButton, QMenu, QAction, QActionGroup, QToolTip,
    QDialog, QTextBrowser
)

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from src.config import ANNOTATED_CLASSES, CLASS_COLORS_RGB, SampleRecord
from src.preprocessing import load_sample_normalized, to_uint8
from src.annotation_utils import parse_yolo_annotations, polygon_to_mask


def _extract_ring_contours(mask: np.ndarray) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
    """Extract outer and inner contours from a ring-shaped mask.

    Applies morphological closing first to fix small gaps in broken rings.

    Returns:
        (outer_polygon, inner_polygon) — each Nx2 int32 or None if not found.
    """
    # Close small gaps in the ring before extracting contours
    h, w = mask.shape[:2]
    kern_size = max(7, int(min(h, w) * 0.015) | 1)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kern_size, kern_size))
    closed = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)

    # Find the outer boundary
    contours_ext, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours_ext:
        return None, None
    outer_contour = max(contours_ext, key=cv2.contourArea)
    if len(outer_contour) < 3:
        return None, None
    outer_poly = outer_contour.reshape(-1, 2).astype(np.int32)

    # Fill the outer contour to find the hole (inner boundary)
    filled_solid = np.zeros_like(closed)
    cv2.fillPoly(filled_solid, [outer_contour], 1)
    hole = filled_solid & (~closed.astype(bool)).astype(np.uint8)

    # Find the largest hole contour = inner boundary
    hole_contours, _ = cv2.findContours(hole, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not hole_contours:
        return outer_poly, None
    inner_contour = max(hole_contours, key=cv2.contourArea)
    if len(inner_contour) < 3:
        return outer_poly, None
    inner_poly = inner_contour.reshape(-1, 2).astype(np.int32)

    return outer_poly, inner_poly


# Target class → annotation class mapping for ring structures
# Target 2 (endodermis ring) → annotation 2 (outer endo) + 3 (inner endo)
# Target 4 (exodermis ring) → annotation 4 (outer exo) + 5 (inner exo)
_RING_CLASS_MAP = {
    2: (2, 3),   # endodermis: outer=2, inner=3
    4: (4, 5),   # exodermis: outer=4, inner=5
}


def parse_npz_predictions(path: Path, img_w: int = 0, img_h: int = 0) -> List[dict]:
    """Parse an .npz prediction file containing binary masks into polygon dicts.

    Predictions use target class IDs (0-4). Ring classes (endodermis=2, exodermis=4)
    are split back into outer/inner annotation class pairs (2→2+3, 4→4+5).

    Args:
        path: Path to .npz file with 'masks', 'labels', and optionally 'scores'.
        img_w, img_h: Unused, kept for API compatibility.

    Returns:
        List of dicts with keys: class_id (int), polygon (Nx2 int32 array).
    """
    data = np.load(str(path), allow_pickle=True)
    masks = data['masks']      # (N, H, W) uint8
    labels = data['labels']    # (N,)
    annotations = []
    for i in range(len(labels)):
        label = int(labels[i])
        mask = masks[i].astype(np.uint8)

        if label in _RING_CLASS_MAP:
            # Ring mask: split into outer + inner annotation polygons
            outer_cls, inner_cls = _RING_CLASS_MAP[label]
            outer_poly, inner_poly = _extract_ring_contours(mask)
            if outer_poly is not None:
                annotations.append({"class_id": outer_cls, "polygon": outer_poly})
            if inner_poly is not None:
                annotations.append({"class_id": inner_cls, "polygon": inner_poly})
        else:
            # Non-ring class: use largest external contour
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if not contours:
                continue
            contour = max(contours, key=cv2.contourArea)
            if len(contour) < 3:
                continue
            polygon = contour.reshape(-1, 2).astype(np.int32)
            annotations.append({"class_id": label, "polygon": polygon})
    return annotations


# ── QC checks ─────────────────────────────────────────────────────────────

def run_qc_checks(polygons: List[dict], img_h: int = 1024, img_w: int = 1024) -> List[str]:
    """Run anatomical constraint checks on annotation polygons.

    Args:
        polygons: List of dicts with 'class_id' and 'polygon' (Nx2 int32 arrays).
        img_h, img_w: Rasterization dimensions.

    Returns:
        List of violation description strings. Empty list means all checks passed.
    """
    violations = []

    # Group polygons by class
    by_class: dict = {}
    for p in polygons:
        cid = p["class_id"]
        by_class.setdefault(cid, []).append(p)

    # Helper: rasterize a single polygon to mask
    def _mask(poly_dict):
        pts = poly_dict["polygon"].astype(np.float32)
        return polygon_to_mask(pts, img_h, img_w)

    # Helper: check if mask_inner is contained within mask_outer (2% tolerance)
    def _is_contained(mask_inner, mask_outer, tolerance=0.02):
        outside = int((mask_inner & ~mask_outer).sum())
        total = int(mask_inner.sum())
        if total == 0:
            return True
        return outside <= tolerance * total

    # Cache masks
    mask_cache = {}
    def _get_mask(idx):
        if idx not in mask_cache:
            mask_cache[idx] = _mask(polygons[idx])
        return mask_cache[idx]

    # Get single-instance masks by class (for whole root, endo, exo)
    def _class_mask(cid):
        """Union mask for all polygons of a given class."""
        if cid not in by_class:
            return None
        masks = [_mask(p) for p in by_class[cid]]
        result = masks[0]
        for m in masks[1:]:
            result = result | m
        return result

    # Helper: check no boundary-crossing intersection between two masks
    def _no_boundary_crossing(mask_a, mask_b, name_a, name_b):
        overlap = int((mask_a & mask_b).sum())
        if overlap == 0:
            return  # no overlap, OK
        sa, sb = int(mask_a.sum()), int(mask_b.sum())
        a_in_b = (sa - overlap) <= 0.02 * sa if sa > 0 else True
        b_in_a = (sb - overlap) <= 0.02 * sb if sb > 0 else True
        if not a_in_b and not b_in_a:
            violations.append(
                f"Boundary intersection between {name_a} and {name_b}")

    # Check 1: All polygons within whole root (class 0) — skip if missing
    root_mask = _class_mask(0)
    if root_mask is not None:
        for p in polygons:
            if p["class_id"] == 0:
                continue
            m = _mask(p)
            if not _is_contained(m, root_mask):
                cname = CLASS_SHORT_NAMES.get(p["class_id"], f"Class {p['class_id']}")
                violations.append(f"{cname} polygon extends outside Whole Root boundary")

    # Check 2: Inner exodermis (5) within outer exodermis (4) — skip if either missing
    if 4 in by_class and 5 in by_class:
        outer_exo = _class_mask(4)
        inner_exo = _class_mask(5)
        if not _is_contained(inner_exo, outer_exo):
            violations.append("Inner Exodermis (I.Exo) extends outside Outer Exodermis (O.Exo)")
        _no_boundary_crossing(outer_exo, inner_exo, "O.Exo", "I.Exo")

    # Check 3: Outer endodermis (2) within inner exodermis (5) — skip if either missing
    if 2 in by_class and 5 in by_class:
        outer_endo = _class_mask(2)
        inner_exo = _class_mask(5)
        if not _is_contained(outer_endo, inner_exo):
            violations.append("Outer Endodermis (O.Endo) extends outside Inner Exodermis (I.Exo)")
        _no_boundary_crossing(outer_endo, inner_exo, "O.Endo", "I.Exo")

    # Check 4: Inner endodermis (3) within outer endodermis (2) — skip if either missing
    if 2 in by_class and 3 in by_class:
        outer_endo = _class_mask(2)
        inner_endo = _class_mask(3)
        if not _is_contained(inner_endo, outer_endo):
            violations.append("Inner Endodermis (I.Endo) extends outside Outer Endodermis (O.Endo)")
        _no_boundary_crossing(outer_endo, inner_endo, "O.Endo", "I.Endo")

    # Check 5: Aerenchyma in cortex only — skip if class 1 missing
    if 1 in by_class:
        outer_endo = _class_mask(2)
        # Outer boundary of cortex
        if 5 in by_class:
            cortex_outer = _class_mask(5)  # inner exodermis
        elif root_mask is not None:
            cortex_outer = root_mask
        else:
            cortex_outer = None

        if cortex_outer is not None and outer_endo is not None:
            cortex_mask = cortex_outer & ~outer_endo
            for p in by_class[1]:
                m = _mask(p)
                if not _is_contained(m, cortex_mask, tolerance=0.02):
                    violations.append("Aerenchyma polygon extends outside cortex region")
                    break  # report once

    # Check 6: Required classes — must have root, O.Endo, I.Endo, O.Exo, I.Exo
    required = {0: "Whole Root", 2: "O.Endo", 3: "I.Endo", 4: "O.Exo", 5: "I.Exo"}
    for cid, name in required.items():
        if cid not in by_class:
            violations.append(f"Missing required class: {name}")

    return violations


# ── Editor modes ──────────────────────────────────────────────────────────
MODE_CORRECT_GT = "Correct GT with Pred"
MODE_CORRECT_PRED = "Correct Predictions"
MODE_CREATE_GT = "Create GT"


# Colors for visualization (QColor format)
CLASS_QCOLORS = {
    0: QColor(0, 0, 255, 180),      # Whole Root - Blue
    1: QColor(255, 255, 0, 180),    # Aerenchyma - Yellow
    2: QColor(0, 255, 0, 180),      # Outer Endodermis - Green
    3: QColor(255, 0, 0, 180),      # Inner Endodermis - Red
    4: QColor(255, 128, 0, 180),    # Outer Exodermis - Orange
    5: QColor(128, 0, 255, 180),    # Inner Exodermis - Purple
}

SELECTED_COLOR = QColor(255, 255, 255, 220)  # White for selected polygon

CLASS_SHORT_NAMES = {
    0: "Root", 1: "Aer", 2: "O.Endo", 3: "I.Endo", 4: "O.Exo", 5: "I.Exo",
}


# ── Lightweight sample discovery ──────────────────────────────────────────

def discover_samples(image_dir: Path, annotation_dir: Optional[Path] = None,
                     prediction_dir: Optional[Path] = None,
                     require_annotation: bool = False,
                     require_prediction: bool = False) -> List[SampleRecord]:
    """Walk image_dir to find samples, optionally requiring annotation/prediction files.

    Supports both structured (Species/Microscope/Experiment/Sample) and generic
    (any depth) directory layouts.  For generic layouts the sample folder name is
    used as sample_name and species/microscope/experiment are set to "_".
    """
    if not image_dir.exists():
        return []

    ann_files = set()
    if annotation_dir and annotation_dir.exists():
        ann_files = {f.name for f in annotation_dir.iterdir() if f.is_file()}

    pred_files = set()
    if prediction_dir and prediction_dir.exists():
        pred_files = {f.name for f in prediction_dir.iterdir() if f.is_file()}

    samples = []
    for root, dirs, files in os.walk(image_dir):
        if not dirs and files:
            tif_files = [f for f in files if f.lower().endswith((".tif", ".tiff"))]
            if not tif_files:
                continue

            # Check that the three required channels exist
            channel_map = {}
            for f in tif_files:
                ch = f.rsplit("_", 1)[-1].split(".")[0].upper()
                if ch in ("DAPI", "FITC", "TRITC"):
                    channel_map[ch] = f
            if not all(ch in channel_map for ch in ("DAPI", "FITC", "TRITC")):
                continue

            full_path = Path(root)
            try:
                rel_path = full_path.relative_to(image_dir)
            except ValueError:
                continue
            parts = rel_path.parts

            if len(parts) == 4:
                # Structured: Species/Microscope/Experiment/SampleName
                species, microscope, experiment, sample_name = parts
            else:
                # Generic: use folder name as sample_name
                sample_name = full_path.name
                species = microscope = experiment = "_"

            uid = f"{species}_{microscope}_{experiment}_{sample_name}"
            ann_name = uid + ".txt"

            # For generic layout, also check using just the sample folder name
            ann_name_short = sample_name + ".txt"

            has_ann = ann_name in ann_files or ann_name_short in ann_files
            # Check for both .txt and .npz prediction files
            pred_name_npz = uid + ".npz"
            pred_name_npz_short = sample_name + ".npz"
            has_pred = (ann_name in pred_files or ann_name_short in pred_files
                        or pred_name_npz in pred_files or pred_name_npz_short in pred_files)

            if require_annotation and not has_ann:
                continue
            if require_prediction and not has_pred:
                continue

            # Resolve actual annotation path
            if annotation_dir:
                if ann_name in ann_files:
                    ann_path = annotation_dir / ann_name
                elif ann_name_short in ann_files:
                    ann_path = annotation_dir / ann_name_short
                else:
                    ann_path = annotation_dir / ann_name_short
            else:
                ann_path = Path(ann_name_short)

            samples.append(SampleRecord(
                species=species, microscope=microscope,
                experiment=experiment, sample_name=sample_name,
                image_dir=full_path, annotation_path=ann_path,
            ))

    samples.sort(key=lambda s: s.uid)
    return samples


class PolygonCanvas(QWidget):
    """Canvas widget that displays an image with polygon overlays, zoom/pan, and interaction."""

    def __init__(self, title: str, parent=None, editable: bool = False, selectable: bool = False):
        super().__init__(parent)
        self.title = title
        self.editable = editable
        self.selectable = selectable
        self.base_image: Optional[np.ndarray] = None
        self.polygons: List[dict] = []
        self.selected_idx: Optional[int] = None
        self.drawing_mode = False
        self.drawing_points: List[Tuple[int, int]] = []
        self.drawing_class = 1
        self.hidden_classes: set = set()

        # Vertex editing state
        self.editing_mode = False
        self._dragging_vertex: Optional[int] = None
        self._selected_vertex: Optional[int] = None
        self._selected_vertices: set = set()  # multi-select set
        self._drag_started = False
        self._rubber_band_start: Optional[Tuple[float, float]] = None
        self._rubber_band_end: Optional[Tuple[float, float]] = None

        # Brush editing state
        self.brush_mode = False
        self.brush_radius = 20  # pixels in image space
        self._brush_mask: Optional[np.ndarray] = None  # (H, W) uint8
        self._brush_class_id: int = 0
        self._brush_painting = False
        self._brush_erasing = False
        self._brush_cursor_pos: Optional[Tuple[float, float]] = None
        self._brush_orig_idx: Optional[int] = None  # index of polygon being edited

        # Edge hover state for adding nodes
        self._hover_edge_idx: Optional[int] = None
        self._hover_midpoint: Optional[Tuple[float, float]] = None
        self._hover_timer = QTimer(self)
        self._hover_timer.setSingleShot(True)
        self._hover_timer.setInterval(10)
        self._hover_timer.timeout.connect(self._on_hover_timeout)
        self._pending_hover_edge: Optional[int] = None
        self._pending_hover_mid: Optional[Tuple[float, float]] = None

        # Zoom / pan state
        self.zoom_level = 1.0
        self.pan_offset = [0.0, 0.0]
        self._panning = False
        self._pan_start: Optional[QPoint] = None

        self.setMinimumSize(300, 300)
        self.setStyleSheet("border: 1px solid #555; background-color: #1a1a1a;")
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.setMouseTracking(True)

    # ── Edge hover helpers ────────────────────────────────────────────────

    @staticmethod
    def _project_onto_segment(px, py, ax, ay, bx, by) -> Tuple[float, float, float]:
        """Return (distance, proj_x, proj_y) from point to segment."""
        dx, dy = bx - ax, by - ay
        len_sq = dx * dx + dy * dy
        if len_sq == 0:
            return ((px - ax) ** 2 + (py - ay) ** 2) ** 0.5, ax, ay
        t = max(0, min(1, ((px - ax) * dx + (py - ay) * dy) / len_sq))
        proj_x = ax + t * dx
        proj_y = ay + t * dy
        return ((px - proj_x) ** 2 + (py - proj_y) ** 2) ** 0.5, proj_x, proj_y

    def _point_to_segment_dist(self, px, py, ax, ay, bx, by) -> float:
        return self._project_onto_segment(px, py, ax, ay, bx, by)[0]

    def _find_nearest_edge(self, ix, iy) -> Optional[Tuple[int, float, float]]:
        if self.selected_idx is None or not (0 <= self.selected_idx < len(self.polygons)):
            return None
        polygon = self.polygons[self.selected_idx]["polygon"]
        if len(polygon) < 3:
            return None
        s = self._effective_scale()
        threshold = 8.0 / s
        best_edge = None
        best_dist = float("inf")
        n = len(polygon)
        for i in range(n):
            j = (i + 1) % n
            ax, ay = float(polygon[i][0]), float(polygon[i][1])
            bx, by = float(polygon[j][0]), float(polygon[j][1])
            dist, proj_x, proj_y = self._project_onto_segment(ix, iy, ax, ay, bx, by)
            if dist < threshold and dist < best_dist:
                best_dist = dist
                best_edge = (i, proj_x, proj_y)
        return best_edge

    def _on_hover_timeout(self):
        if self._pending_hover_edge is not None:
            self._hover_edge_idx = self._pending_hover_edge
            self._hover_midpoint = self._pending_hover_mid
            self.update()

    def _clear_hover(self):
        self._hover_timer.stop()
        self._pending_hover_edge = None
        self._pending_hover_mid = None
        if self._hover_edge_idx is not None:
            self._hover_edge_idx = None
            self._hover_midpoint = None
            self.update()

    # ── Zoom / pan helpers ────────────────────────────────────────────────

    def reset_view(self):
        self.zoom_level = 1.0
        self.pan_offset = [0.0, 0.0]

    def _base_scale(self) -> float:
        if self.base_image is None:
            return 1.0
        h, w = self.base_image.shape[:2]
        ws = (self.width() - 4) / w
        hs = (self.height() - 4) / h
        return min(ws, hs)

    def _effective_scale(self) -> float:
        return self._base_scale() * self.zoom_level

    def image_to_widget(self, ix: float, iy: float) -> QPointF:
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
        self.zoom_level = zoom
        self.pan_offset = list(pan)
        self.update()

    # ── Data setters ──────────────────────────────────────────────────────

    def set_image(self, img: np.ndarray):
        self.base_image = img.copy()
        self._qimage = QImage(
            self.base_image.data, img.shape[1], img.shape[0],
            3 * img.shape[1], QImage.Format_RGB888
        )
        self.update()

    def set_polygons(self, polygons: List[dict]):
        self.polygons = polygons
        self.selected_idx = None
        self.update()

    def update_display(self):
        self.update()

    # ── Paint ─────────────────────────────────────────────────────────────

    def paintEvent(self, event):
        if self.base_image is None:
            return
        h, w = self.base_image.shape[:2]
        s = self._effective_scale()

        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)
        painter.setRenderHint(QPainter.SmoothPixmapTransform)
        painter.fillRect(self.rect(), QColor(26, 26, 26))

        origin = self.image_to_widget(0, 0)
        painter.translate(origin)
        painter.scale(s, s)
        painter.drawImage(QRectF(0, 0, w, h), self._qimage)

        for idx, poly_data in enumerate(self.polygons):
            class_id = poly_data["class_id"]
            polygon = poly_data["polygon"]
            if class_id in self.hidden_classes and idx != self.selected_idx:
                continue
            if idx == self.selected_idx:
                color = SELECTED_COLOR
                pen_width = max(2, int(5 / s))
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

        handle_screen_px = 5.0
        handle_r = handle_screen_px / s
        handle_pen = 1.5 / s

        if self.editing_mode and self.selected_idx is not None and 0 <= self.selected_idx < len(self.polygons):
            polygon = self.polygons[self.selected_idx]["polygon"]
            for vi, pt in enumerate(polygon):
                if vi == self._selected_vertex or vi in self._selected_vertices:
                    painter.setPen(QPen(QColor(0, 255, 255), handle_pen * 1.5))
                    painter.setBrush(QBrush(QColor(0, 255, 255, 200)))
                    painter.drawEllipse(QPointF(float(pt[0]), float(pt[1])), handle_r * 1.4, handle_r * 1.4)
                else:
                    painter.setPen(QPen(QColor(255, 0, 255), handle_pen))
                    painter.setBrush(QBrush(QColor(255, 0, 255, 160)))
                    painter.drawEllipse(QPointF(float(pt[0]), float(pt[1])), handle_r, handle_r)

            # Draw rubber band selection rectangle
            if self._rubber_band_start is not None and self._rubber_band_end is not None:
                rx0, ry0 = self._rubber_band_start
                rx1, ry1 = self._rubber_band_end
                rect = QRectF(min(rx0, rx1), min(ry0, ry1), abs(rx1 - rx0), abs(ry1 - ry0))
                painter.setPen(QPen(QColor(0, 255, 255, 200), handle_pen, Qt.DashLine))
                painter.setBrush(QBrush(QColor(0, 255, 255, 40)))
                painter.drawRect(rect)

        if self.editing_mode and self._hover_midpoint is not None and self.selected_idx is not None:
            mx, my = self._hover_midpoint
            plus_r = handle_r * 1.6
            painter.setPen(QPen(QColor(0, 255, 0), handle_pen * 2))
            painter.setBrush(QBrush(QColor(0, 255, 0, 100)))
            painter.drawEllipse(QPointF(mx, my), plus_r, plus_r)
            arm = plus_r * 0.6
            painter.drawLine(QPointF(mx - arm, my), QPointF(mx + arm, my))
            painter.drawLine(QPointF(mx, my - arm), QPointF(mx, my + arm))

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

        # Brush mode: mask overlay + cursor
        if self.brush_mode and self._brush_mask is not None:
            mask_h, mask_w = self._brush_mask.shape
            color = CLASS_QCOLORS.get(self._brush_class_id, QColor(128, 128, 128, 180))
            overlay = np.zeros((mask_h, mask_w, 4), dtype=np.uint8)
            overlay[self._brush_mask > 0] = [
                color.blue(), color.green(), color.red(), 100]
            qimg = QImage(overlay.data, mask_w, mask_h,
                          4 * mask_w, QImage.Format_ARGB32)
            painter.drawImage(QRectF(0, 0, mask_w, mask_h), qimg)

        painter.resetTransform()

        # Brush cursor (drawn in widget space, after resetTransform)
        if self.brush_mode and self._brush_cursor_pos is not None:
            bx, by = self._brush_cursor_pos
            wp = self.image_to_widget(bx, by)
            radius_w = self.brush_radius * self._effective_scale()
            cursor_color = QColor(255, 0, 0, 200) if self._brush_erasing else QColor(255, 255, 255, 200)
            painter.setPen(QPen(cursor_color, 1.5, Qt.DashLine))
            painter.setBrush(Qt.NoBrush)
            painter.drawEllipse(wp, radius_w, radius_w)
            # Brush size text
            painter.setPen(QPen(Qt.white))
            painter.setFont(QFont("Arial", 10))
            painter.drawText(int(wp.x() + radius_w + 5), int(wp.y() - 5),
                             f"{self.brush_radius}px")

        painter.setPen(QPen(Qt.white))
        painter.setFont(QFont("Arial", 12, QFont.Bold))
        painter.drawText(10, 25, self.title)

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
        if self.base_image is None:
            return
        # In brush mode: Ctrl+scroll = resize brush, scroll = zoom (default)
        if self.brush_mode and (event.modifiers() & Qt.ControlModifier):
            delta = 3 if event.angleDelta().y() > 0 else -3
            self.brush_radius = max(2, min(self.brush_radius + delta, 200))
            mx, my = event.pos().x(), event.pos().y()
            self._brush_cursor_pos = self.widget_to_image(mx, my)
            self.update()
            return
        mx, my = event.pos().x(), event.pos().y()
        ix, iy = self.widget_to_image(mx, my)
        factor = 1.15 if event.angleDelta().y() > 0 else 1.0 / 1.15
        self.zoom_level = max(0.5, min(self.zoom_level * factor, 40.0))
        ix2, iy2 = self.widget_to_image(mx, my)
        self.pan_offset[0] -= (ix2 - ix)
        self.pan_offset[1] -= (iy2 - iy)
        self.update()
        self._sync_view()

    def mousePressEvent(self, event):
        if self.base_image is None:
            return
        if event.button() in (Qt.MiddleButton, Qt.RightButton):
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

        # Brush mode: when editing existing polygon, erase by default (Shift = paint)
        # When drawing new polygon (no original), paint by default (Shift = erase)
        if self.editable and self.brush_mode:
            shift_held = bool(event.modifiers() & Qt.ShiftModifier)
            if self._brush_orig_idx is None:
                # New polygon: default = paint, Shift = erase
                self._brush_erasing = shift_held
            else:
                # Editing existing: default = erase, Shift = paint
                self._brush_erasing = not shift_held
            self._brush_painting = True
            self._apply_brush_stroke(ix, iy)
            self.update()
            return

        if self.editable and self.editing_mode and self.selected_idx is not None:
            polygon = self.polygons[self.selected_idx]["polygon"]
            s = self._effective_scale()
            grab_radius = 8.0 / s
            for vi, pt in enumerate(polygon):
                if abs(ix - pt[0]) < grab_radius and abs(iy - pt[1]) < grab_radius:
                    self._dragging_vertex = vi
                    self._selected_vertex = vi
                    self._selected_vertices.clear()
                    self._drag_started = False
                    self._clear_hover()
                    parent = self.get_editor_parent()
                    if parent:
                        parent.push_undo()
                    self.update()
                    return

            if self._hover_midpoint is not None and self._hover_edge_idx is not None:
                mx, my = self._hover_midpoint
                plus_grab = 10.0 / s
                if abs(ix - mx) < plus_grab and abs(iy - my) < plus_grab:
                    parent = self.get_editor_parent()
                    if parent:
                        parent.push_undo()
                    edge_idx = self._hover_edge_idx
                    new_pt = [mx, my]
                    self.polygons[self.selected_idx]["polygon"] = np.insert(
                        polygon, edge_idx + 1, new_pt, axis=0
                    )
                    self._selected_vertex = edge_idx + 1
                    self._dragging_vertex = edge_idx + 1
                    self._drag_started = False
                    self._clear_hover()
                    if parent:
                        parent.update_status(f"Added vertex ({len(self.polygons[self.selected_idx]['polygon'])} vertices)")
                    self.update()
                    return

            # Shift+click in edit mode: insert a free vertex at cursor position
            if event.modifiers() & Qt.ShiftModifier:
                insert_after = self._selected_vertex if self._selected_vertex is not None else len(polygon) - 1
                parent = self.get_editor_parent()
                if parent:
                    parent.push_undo()
                new_pt = [ix, iy]
                self.polygons[self.selected_idx]["polygon"] = np.insert(
                    polygon, insert_after + 1, new_pt, axis=0
                )
                self._selected_vertex = insert_after + 1
                self._selected_vertices.clear()
                self._dragging_vertex = insert_after + 1
                self._drag_started = False
                self._clear_hover()
                if parent:
                    parent.update_status(
                        f"Inserted free vertex at ({int(ix)}, {int(iy)}) — "
                        f"{len(self.polygons[self.selected_idx]['polygon'])} vertices")
                self.update()
                return

            # Start rubber band selection for multi-vertex select
            self._rubber_band_start = (ix, iy)
            self._rubber_band_end = (ix, iy)
            self._dragging_vertex = None
            self._selected_vertex = None
            self._selected_vertices.clear()
            self._clear_hover()
            self.update()
            return

        if self.editable and self.drawing_mode:
            self.drawing_points.append((img_x, img_y))
            self.update()
        elif self.editable or self.selectable:
            self.selected_idx = self.find_polygon_at(img_x, img_y)
            self.update()
            if self.selected_idx is not None:
                parent = self.get_editor_parent()
                if parent:
                    poly = self.polygons[self.selected_idx]
                    cname = ANNOTATED_CLASSES.get(poly['class_id'], 'Unknown')
                    parent.update_status(f"Selected polygon {self.selected_idx} ({cname})")

    def mouseMoveEvent(self, event):
        # Brush mode: update cursor + paint
        if self.brush_mode and self.base_image is not None:
            ix, iy = self.widget_to_image(event.pos().x(), event.pos().y())
            self._brush_cursor_pos = (ix, iy)
            if self._brush_painting:
                shift_held = bool(event.modifiers() & Qt.ShiftModifier)
                if self._brush_orig_idx is None:
                    self._brush_erasing = shift_held
                else:
                    self._brush_erasing = not shift_held
                self._apply_brush_stroke(ix, iy)
            self.update()
            # Still allow panning with middle/right button
            if self._panning and self._pan_start is not None:
                s = self._effective_scale()
                if s > 0:
                    dx = (event.pos().x() - self._pan_start.x()) / s
                    dy = (event.pos().y() - self._pan_start.y()) / s
                    self.pan_offset[0] -= dx
                    self.pan_offset[1] -= dy
                    self._pan_start = event.pos()
                    self._sync_view()
            return

        # Rubber band selection drag
        if self._rubber_band_start is not None and self.editing_mode:
            ix, iy = self.widget_to_image(event.pos().x(), event.pos().y())
            self._rubber_band_end = (ix, iy)
            self.update()
            return

        if self._dragging_vertex is not None and self.editing_mode and self.selected_idx is not None:
            self._drag_started = True
            ix, iy = self.widget_to_image(event.pos().x(), event.pos().y())
            h, w = self.base_image.shape[:2]
            ix = max(0, min(ix, w - 1))
            iy = max(0, min(iy, h - 1))
            self.polygons[self.selected_idx]["polygon"][self._dragging_vertex] = [ix, iy]
            self.update()
            return

        if (self.editing_mode and self.selected_idx is not None
                and self._dragging_vertex is None and self.base_image is not None):
            ix, iy = self.widget_to_image(event.pos().x(), event.pos().y())
            result = self._find_nearest_edge(ix, iy)
            if result is not None:
                edge_idx, proj_x, proj_y = result
                if edge_idx != self._pending_hover_edge:
                    # New edge — use timer for debounce
                    self._hover_timer.stop()
                    self._hover_edge_idx = None
                    self._hover_midpoint = None
                    self._pending_hover_edge = edge_idx
                    self._pending_hover_mid = (proj_x, proj_y)
                    self._hover_timer.start()
                    self.update()
                else:
                    # Same edge — update position to follow mouse
                    self._pending_hover_mid = (proj_x, proj_y)
                    if self._hover_edge_idx is not None:
                        self._hover_midpoint = (proj_x, proj_y)
                    self.update()
            else:
                self._clear_hover()

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
        # Brush mode: stop painting
        if self.brush_mode and event.button() == Qt.LeftButton:
            self._brush_painting = False
            return

        # Finalize rubber band selection
        if self._rubber_band_start is not None and event.button() == Qt.LeftButton:
            if (self._rubber_band_end is not None and self.selected_idx is not None
                    and 0 <= self.selected_idx < len(self.polygons)):
                rx0, ry0 = self._rubber_band_start
                rx1, ry1 = self._rubber_band_end
                x_min, x_max = min(rx0, rx1), max(rx0, rx1)
                y_min, y_max = min(ry0, ry1), max(ry0, ry1)
                # Only count as drag-select if the rectangle is non-trivial
                s = self._effective_scale()
                min_drag = 4.0 / s if s > 0 else 4.0
                if abs(rx1 - rx0) > min_drag or abs(ry1 - ry0) > min_drag:
                    polygon = self.polygons[self.selected_idx]["polygon"]
                    shift = event.modifiers() & Qt.ShiftModifier
                    if not shift:
                        self._selected_vertices.clear()
                    for vi, pt in enumerate(polygon):
                        if x_min <= pt[0] <= x_max and y_min <= pt[1] <= y_max:
                            self._selected_vertices.add(vi)
                    if self._selected_vertices:
                        self._selected_vertex = max(self._selected_vertices)
                        parent = self.get_editor_parent()
                        if parent:
                            parent.update_status(
                                f"Selected {len(self._selected_vertices)} vertices (Delete to remove)")
                else:
                    # Tiny drag = click on empty space, deselect
                    self._selected_vertex = None
                    self._selected_vertices.clear()
            self._rubber_band_start = None
            self._rubber_band_end = None
            self.update()
            return

        if self._dragging_vertex is not None and event.button() == Qt.LeftButton:
            if not self._drag_started:
                parent = self.get_editor_parent()
                if parent and parent._undo_stack:
                    parent._undo_stack.pop()
            self._dragging_vertex = None
            return
        if event.button() in (Qt.MiddleButton, Qt.RightButton) and self._panning:
            self._panning = False
            self._pan_start = None
            self.setCursor(Qt.ArrowCursor)

    def find_polygon_at(self, x: int, y: int) -> Optional[int]:
        best_idx = None
        best_area = float("inf")
        for idx, poly_data in enumerate(self.polygons):
            if poly_data["class_id"] in self.hidden_classes:
                continue
            polygon = poly_data["polygon"]
            pts = polygon.reshape((-1, 1, 2)).astype(np.float32)
            dist = cv2.pointPolygonTest(pts, (float(x), float(y)), False)
            if dist >= 0:
                area = cv2.contourArea(pts)
                if area < best_area:
                    best_area = area
                    best_idx = idx
        return best_idx

    def _sync_view(self):
        parent = self.get_editor_parent()
        if parent:
            parent.sync_view(self)

    def get_editor_parent(self):
        parent = self.parent()
        while parent is not None:
            if isinstance(parent, AnnotationEditor):
                return parent
            parent = parent.parent()
        return None

    def start_drawing(self, class_id: int):
        self.drawing_mode = True
        self.drawing_class = class_id
        self.drawing_points = []
        self.selected_idx = None
        self.update()

    def finish_drawing(self) -> Optional[dict]:
        if len(self.drawing_points) >= 3:
            polygon = np.array(self.drawing_points, dtype=np.float32)
            result = {"class_id": self.drawing_class, "polygon": polygon}
            self.drawing_mode = False
            self.drawing_points = []
            return result
        self.cancel_drawing()
        return None

    def cancel_drawing(self):
        self.drawing_mode = False
        self.drawing_points = []
        self.update()

    def delete_selected_vertex(self) -> bool:
        if self.selected_idx is None or not (0 <= self.selected_idx < len(self.polygons)):
            return False
        polygon = self.polygons[self.selected_idx]["polygon"]

        # Collect all indices to delete (multi-select or single)
        to_delete = set(self._selected_vertices)
        if self._selected_vertex is not None:
            to_delete.add(self._selected_vertex)
        if not to_delete:
            return False

        # Ensure at least 3 vertices remain
        remaining = len(polygon) - len(to_delete)
        if remaining < 3:
            return False

        # Delete in reverse order to keep indices valid
        self.polygons[self.selected_idx]["polygon"] = np.delete(
            polygon, sorted(to_delete), axis=0
        )
        self._selected_vertex = None
        self._selected_vertices.clear()
        self.update()
        return True

    def delete_selected(self) -> Optional[dict]:
        if self.selected_idx is not None and 0 <= self.selected_idx < len(self.polygons):
            removed = self.polygons.pop(self.selected_idx)
            self.selected_idx = None
            self._selected_vertex = None
            self._selected_vertices.clear()
            self.update()
            return removed
        return None

    # ── Brush mode ────────────────────────────────────────────────────────

    def start_brush_mode(self, polygon_idx: Optional[int]):
        """Enter brush mode. Rasterizes selected polygon to mask, or starts blank."""
        if self.base_image is None:
            return
        h, w = self.base_image.shape[:2]
        self._brush_mask = np.zeros((h, w), dtype=np.uint8)
        if polygon_idx is not None and 0 <= polygon_idx < len(self.polygons):
            poly = self.polygons[polygon_idx]
            self._brush_class_id = poly["class_id"]
            self._brush_orig_idx = polygon_idx
            pts = poly["polygon"].reshape((-1, 1, 2)).astype(np.int32)
            cv2.fillPoly(self._brush_mask, [pts], 1)
        else:
            self._brush_orig_idx = None
        self.brush_mode = True
        self.update()

    def finish_brush_mode(self) -> Optional[dict]:
        """Convert brush mask to polygon and exit brush mode."""
        if self._brush_mask is None or self._brush_mask.sum() == 0:
            self.cancel_brush_mode()
            return None
        contours, _ = cv2.findContours(
            self._brush_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_TC89_L1)
        if not contours:
            self.cancel_brush_mode()
            return None
        largest = max(contours, key=cv2.contourArea)
        if len(largest) < 3:
            self.cancel_brush_mode()
            return None
        polygon = largest.reshape(-1, 2).astype(np.float32)
        result = {"class_id": self._brush_class_id, "polygon": polygon}
        self._reset_brush_state()
        return result

    def cancel_brush_mode(self):
        self._reset_brush_state()
        self.update()

    def _reset_brush_state(self):
        self.brush_mode = False
        self._brush_mask = None
        self._brush_painting = False
        self._brush_erasing = False
        self._brush_cursor_pos = None
        self._brush_orig_idx = None

    def _apply_brush_stroke(self, ix: float, iy: float):
        if self._brush_mask is None:
            return
        cx, cy = int(round(ix)), int(round(iy))
        value = 0 if self._brush_erasing else 1
        cv2.circle(self._brush_mask, (cx, cy), self.brush_radius, int(value), -1)


class AnnotationEditor(QMainWindow):
    """Main window for annotation editing."""

    def __init__(self, data_dir: Optional[Path] = None):
        super().__init__()
        self.data_dir: Optional[Path] = data_dir
        self.editor_mode: str = MODE_CREATE_GT
        self._info_dialog: Optional[QDialog] = None
        self._all_samples: List[SampleRecord] = []  # unfiltered
        self.samples: List[SampleRecord] = []

        self.current_idx = 0
        self.modified = False
        self._undo_stack: List[List[dict]] = []
        self._redo_stack: List[List[dict]] = []
        self._mode_entry_snapshot: Optional[List[dict]] = None
        self._undo_depth_at_mode_entry: int = 0
        self._max_undo = 50
        self._editing_polygon_idx: Optional[int] = None  # tracks polygon being edited
        self._deleting_mode: bool = False  # True when in delete confirmation modal

        # QC status persistence: {"sample_uid": "passed"|"failed"}, absent = to_be_qc
        self._qc_status: dict = {}

        # Display-only brightness/gamma (do not affect saved data)
        self._raw_image: Optional[np.ndarray] = None
        self._brightness = 0       # -100 to +100
        self._gamma = 100          # 10 to 300 (displayed as 0.1 to 3.0)

        self.setWindowTitle("Plant Roots Polygon Editor")
        self.setMinimumSize(1400, 800)
        self.setup_ui()
        self.setup_shortcuts()
        QApplication.instance().installEventFilter(self)

        if self.data_dir:
            self._reload_samples()

    # ── Directory helpers ─────────────────────────────────────────────────

    def _image_dir(self) -> Optional[Path]:
        return self.data_dir / "image" if self.data_dir else None

    def _annotation_dir(self) -> Optional[Path]:
        return self.data_dir / "annotation" if self.data_dir else None

    def _prediction_dir(self) -> Optional[Path]:
        return self.data_dir / "prediction" if self.data_dir else None

    def _validate_dirs_for_mode(self, mode: str) -> Optional[str]:
        """Return error message if directories are missing, else None."""
        if not self.data_dir:
            return "No data directory set. Browse for a data folder first."
        img = self._image_dir()
        if not img or not img.exists():
            return f"image/ directory not found in {self.data_dir}"
        if mode == MODE_CORRECT_GT:
            if not self._annotation_dir().exists():
                return f"annotation/ directory not found in {self.data_dir}"
            if not self._prediction_dir().exists():
                return f"prediction/ directory not found in {self.data_dir}"
        elif mode == MODE_CORRECT_PRED:
            if not self._prediction_dir().exists():
                return f"prediction/ directory not found in {self.data_dir}"
        # MODE_CREATE_GT only needs image/
        return None

    # ── Sample loading ────────────────────────────────────────────────────

    def _reload_samples(self):
        """Discover samples based on current mode and data directory."""
        err = self._validate_dirs_for_mode(self.editor_mode)
        if err:
            self._all_samples = []
            self.samples = []
            self._update_sample_combo()
            self.update_status(err)
            return

        # Load QC status for this data directory
        self._load_qc_status()

        img_dir = self._image_dir()
        ann_dir = self._annotation_dir()
        pred_dir = self._prediction_dir()

        if self.editor_mode == MODE_CORRECT_GT:
            self._all_samples = discover_samples(
                img_dir, ann_dir, pred_dir,
                require_annotation=True, require_prediction=True)
        elif self.editor_mode == MODE_CORRECT_PRED:
            self._all_samples = discover_samples(
                img_dir, ann_dir, pred_dir,
                require_prediction=True)
        else:  # MODE_CREATE_GT
            self._all_samples = discover_samples(img_dir, ann_dir)

        self._apply_filter()

    # ── QC persistence ───────────────────────────────────────────────────

    def _qc_status_path(self) -> Optional[Path]:
        if self.data_dir:
            return self.data_dir / "qc_status.json"
        return None

    def _load_qc_status(self):
        p = self._qc_status_path()
        if p and p.exists():
            try:
                with open(p, 'r') as f:
                    self._qc_status = json.load(f)
            except (json.JSONDecodeError, OSError):
                self._qc_status = {}
        else:
            self._qc_status = {}

    def _save_qc_status(self):
        p = self._qc_status_path()
        if p:
            with open(p, 'w') as f:
                json.dump(self._qc_status, f, indent=2)

    def _get_sample_qc_status(self, sample: SampleRecord) -> str:
        entry = self._qc_status.get(sample.uid, "to_be_qc")
        if isinstance(entry, dict):
            return entry.get("status", "to_be_qc")
        return entry  # backward compat: bare string

    def _get_sample_qc_violations(self, sample: SampleRecord) -> list:
        entry = self._qc_status.get(sample.uid)
        if isinstance(entry, dict):
            return entry.get("violations", [])
        return []

    def _set_sample_qc_status(self, sample: SampleRecord, status: str,
                              violations: list = None):
        self._qc_status[sample.uid] = {
            "status": status,
            "violations": violations or [],
        }

    def _mark_modified(self):
        """Mark current sample as modified and reset its QC status to TBD."""
        if self.modified:
            return  # already marked
        self.modified = True
        if self.samples and 0 <= self.current_idx < len(self.samples):
            sample = self.samples[self.current_idx]
            if self._get_sample_qc_status(sample) != "to_be_qc":
                self._qc_status.pop(sample.uid, None)
                self._save_qc_status()
                self._update_qc_label()

    def _update_qc_label(self):
        """Update the QC status overlay and stats label."""
        _base_style = (
            "background-color: rgba(0, 0, 0, 160); padding: 4px 10px;"
            " border-bottom-left-radius: 6px; font-weight: bold; font-size: 13px;")
        if not self.samples:
            self.qc_label.setText("")
            self.qc_label.hide()
            self.qc_stats_label.setText("")
            return
        sample = self.samples[self.current_idx]
        status = self._get_sample_qc_status(sample)
        if status == "passed":
            self.qc_label.setText("QC Passed")
            self.qc_label.setStyleSheet(_base_style + " color: #00cc00;")
            self.qc_label.setToolTip("")
        elif status == "failed":
            self.qc_label.setText("QC Failed")
            self.qc_label.setStyleSheet(_base_style + " color: #ff3333;")
            violations = self._get_sample_qc_violations(sample)
            if violations:
                self.qc_label.setToolTip("\n".join(f"• {v}" for v in violations))
            else:
                self.qc_label.setToolTip("No details available — re-run QC")
        else:
            self.qc_label.setText("To Be QC")
            self.qc_label.setStyleSheet(_base_style + " color: #888888;")
            self.qc_label.setToolTip("")
        self.qc_label.adjustSize()
        self.qc_label.show()
        self._position_qc_label()
        # Update stats across currently displayed (filtered) samples
        n_passed = n_failed = n_pending = 0
        for s in self.samples:
            st = self._get_sample_qc_status(s)
            if st == "passed":
                n_passed += 1
            elif st == "failed":
                n_failed += 1
            else:
                n_pending += 1
        self.qc_stats_label.setText(
            f'<span style="color:#00cc00;">P:{n_passed}</span>  '
            f'<span style="color:#ff4444;">F:{n_failed}</span>  '
            f'<span style="color:#aaaaaa;">TBD:{n_pending}</span>')

    def _position_qc_label(self):
        """Position the QC overlay label at the top-right of the image panel."""
        if not hasattr(self, '_panel_widget'):
            return
        pw = self._panel_widget
        lw = self.qc_label.width()
        self.qc_label.move(pw.width() - lw - 30, 4)

    def resizeEvent(self, event):
        super().resizeEvent(event)
        self._position_qc_label()

    def _on_qc_current(self):
        """Run QC checks on the current sample."""
        if not self.samples:
            return
        sample = self.samples[self.current_idx]
        polys = self.edit_canvas.polygons
        if not polys:
            QMessageBox.information(self, "QC", "No polygons to check.")
            return

        # Use canvas image dimensions if available, else 1024x1024
        if self.edit_canvas.base_image is not None:
            h, w = self.edit_canvas.base_image.shape[:2]
        else:
            h, w = 1024, 1024

        violations = run_qc_checks(polys, h, w)
        if violations:
            self._set_sample_qc_status(sample, "failed", violations)
            msg = "QC FAILED:\n\n" + "\n".join(f"  - {v}" for v in violations)
        else:
            self._set_sample_qc_status(sample, "passed")
            msg = "QC PASSED — no violations found."

        self._save_qc_status()
        self._update_qc_label()
        QMessageBox.information(self, "QC Result", msg)

    def _on_auto_qc_toggled(self, checked: bool):
        if checked:
            self._schedule_auto_qc()

    def _schedule_auto_qc(self):
        """Schedule auto QC via a deferred timer if the Auto checkbox is checked."""
        if not self.qc_auto_cb.isChecked():
            return
        QTimer.singleShot(0, self._run_auto_qc)

    def _run_auto_qc(self):
        """Run QC silently on the current sample (no message box)."""
        if not self.qc_auto_cb.isChecked():
            return
        if not self.samples or not (0 <= self.current_idx < len(self.samples)):
            return
        if self._in_modal_mode():
            return
        sample = self.samples[self.current_idx]
        polys = self.edit_canvas.polygons
        if not polys:
            # No polygons — clear QC status back to TBD
            self._qc_status.pop(sample.uid, None)
            self._save_qc_status()
            self._update_qc_label()
            return

        if self.edit_canvas.base_image is not None:
            h, w = self.edit_canvas.base_image.shape[:2]
        else:
            h, w = 1024, 1024

        violations = run_qc_checks(polys, h, w)
        if violations:
            self._set_sample_qc_status(sample, "failed", violations)
        else:
            self._set_sample_qc_status(sample, "passed")
        self._save_qc_status()
        self._update_qc_label()

    def _on_qc_all(self):
        """Run QC on currently filtered samples with a progress dialog."""
        if not self.samples:
            QMessageBox.information(self, "QC All", "No samples to check.")
            return

        progress = QProgressDialog(
            f"Running QC on {len(self.samples)} filtered samples...",
            "Cancel", 0, len(self.samples), self)
        progress.setWindowTitle("QC Filtered Samples")
        progress.setMinimumDuration(0)
        progress.setWindowModality(Qt.WindowModal)

        passed = 0
        failed = 0
        errors = 0

        ann_dir = self._annotation_dir()
        for i, sample in enumerate(self.samples):
            if progress.wasCanceled():
                break
            progress.setValue(i)
            QApplication.processEvents()

            # Load annotation polygons (no image loading needed)
            stem = self._file_stem(sample)
            ann_file = ann_dir / f"{stem}.txt" if ann_dir else sample.annotation_path
            if not ann_file or not ann_file.exists():
                # No annotation = skip
                continue

            try:
                polys = parse_yolo_annotations(ann_file, 1024, 1024)
            except Exception:
                errors += 1
                continue

            if not polys:
                continue

            violations = run_qc_checks(polys, 1024, 1024)
            if violations:
                self._set_sample_qc_status(sample, "failed", violations)
                failed += 1
            else:
                self._set_sample_qc_status(sample, "passed")
                passed += 1

        progress.setValue(len(self.samples))
        self._save_qc_status()
        self._update_qc_label()

        msg = f"QC Complete:\n\n  Passed: {passed}\n  Failed: {failed}"
        if errors:
            msg += f"\n  Errors: {errors}"
        QMessageBox.information(self, "QC All Results", msg)

    def _on_filter_changed(self):
        """Handle filter dropdown changes."""
        sender = self.sender()
        filter_type = self.filter_type_combo.currentText()
        is_none = (filter_type == "None")
        is_qc = (filter_type == "By QC")

        # Only repopulate the class combo when the filter TYPE changes
        if sender is self.filter_type_combo:
            self.filter_class_combo.blockSignals(True)
            self.filter_class_combo.clear()
            if is_qc:
                self.filter_class_combo.addItem("None", None)
                self.filter_class_combo.addItem("To Be QC", "to_be_qc")
                self.filter_class_combo.addItem("QC Passed", "passed")
                self.filter_class_combo.addItem("QC Failed", "failed")
            else:
                self.filter_class_combo.addItem("None", None)
                for cid, cname in ANNOTATED_CLASSES.items():
                    self.filter_class_combo.addItem(f"{cid}: {cname}", cid)
            self.filter_class_combo.blockSignals(False)

            self.filter_class_combo.setEnabled(not is_none)
            if is_none:
                self.filter_class_combo.setCurrentIndex(0)

        self._apply_filter()

    def _apply_filter(self):
        """Filter samples based on current filter settings and refresh the view."""
        filter_type = self.filter_type_combo.currentText()
        filter_data = self.filter_class_combo.currentData()

        if filter_type == "None" or filter_data is None:
            self.samples = list(self._all_samples)
        elif filter_type == "By QC":
            # filter_data is a QC status string: "to_be_qc", "passed", "failed"
            self.samples = []
            for sample in self._all_samples:
                status = self._get_sample_qc_status(sample)
                if status == filter_data:
                    self.samples.append(sample)
        else:
            class_id = filter_data
            self.samples = []
            for sample in self._all_samples:
                ann_path = sample.annotation_path
                has_class = False
                if ann_path and ann_path.exists():
                    try:
                        with open(ann_path, 'r') as f:
                            for line in f:
                                parts = line.strip().split()
                                if parts and int(parts[0]) == class_id:
                                    has_class = True
                                    break
                    except (ValueError, IndexError):
                        pass
                if filter_type == "By Containing" and has_class:
                    self.samples.append(sample)
                elif filter_type == "By Missing" and not has_class:
                    self.samples.append(sample)

        self._update_sample_combo()
        self.current_idx = 0
        self.modified = False
        self._undo_stack.clear()
        self._redo_stack.clear()

        if self.samples:
            self.load_sample(0)
            self.update_status(f"{len(self.samples)} images")
        else:
            # Clear canvases and raw image
            self._raw_image = None
            for c in self._all_canvases():
                c.base_image = None
                c.set_polygons([])
                c.update()
            self.update_status("0 images")

    @staticmethod
    def _display_name(sample: SampleRecord) -> str:
        """Return a display name: full uid for structured, sample_name for generic."""
        if sample.species == "_":
            return sample.sample_name
        return sample.uid

    @staticmethod
    def _file_stem(sample: SampleRecord) -> str:
        """Return the file stem used for annotation/prediction .txt files."""
        if sample.species == "_":
            return sample.sample_name
        return sample.uid

    def _update_sample_combo(self):
        self.sample_combo.blockSignals(True)
        self.sample_combo.clear()
        for i, s in enumerate(self.samples):
            self.sample_combo.addItem(f"{i+1}. {self._display_name(s)}")
        self.sample_combo.blockSignals(False)
        self.sample_label.setText(f"{len(self.samples)} images")

    def _all_canvases(self):
        canvases = [self.original_canvas, self.edit_canvas]
        if hasattr(self, 'ref_canvas'):
            canvases.append(self.ref_canvas)
        return canvases

    # ── UI setup ──────────────────────────────────────────────────────────

    def setup_ui(self):
        central = QWidget()
        self.setCentralWidget(central)
        main_layout = QVBoxLayout(central)

        # ── Row 1: Data + Navigation + Settings gear ────────────────────
        row1 = QHBoxLayout()

        # Data folder (QGroupBox)
        data_group = QGroupBox("Data")
        data_layout = QHBoxLayout(data_group)
        data_layout.setContentsMargins(4, 2, 4, 2)
        self.data_dir_entry = QLineEdit()
        self.data_dir_entry.setPlaceholderText("Browse for data folder...")
        if self.data_dir:
            self.data_dir_entry.setText(str(self.data_dir))
        self.data_dir_entry.returnPressed.connect(self._on_data_dir_changed)
        data_layout.addWidget(self.data_dir_entry)
        self.browse_btn = QPushButton("...")
        self.browse_btn.setFixedWidth(30)
        self.browse_btn.clicked.connect(self._browse_data_dir)
        data_layout.addWidget(self.browse_btn)
        row1.addWidget(data_group)

        # Navigation (filter + sample selector)
        nav_group = QGroupBox("Navigation")
        nav_layout = QHBoxLayout(nav_group)
        nav_layout.setContentsMargins(4, 2, 4, 2)
        nav_layout.addWidget(QLabel("Filter:"))
        self.filter_type_combo = QComboBox()
        self.filter_type_combo.addItems(["None", "By Missing", "By Containing", "By QC"])
        self.filter_type_combo.setMinimumWidth(110)
        self.filter_type_combo.currentIndexChanged.connect(self._on_filter_changed)
        nav_layout.addWidget(self.filter_type_combo)
        self.filter_class_combo = QComboBox()
        self.filter_class_combo.setMinimumWidth(130)
        self.filter_class_combo.addItem("None", None)
        for cid, cname in ANNOTATED_CLASSES.items():
            self.filter_class_combo.addItem(f"{cid}: {cname}", cid)
        self.filter_class_combo.setEnabled(False)
        self.filter_class_combo.currentIndexChanged.connect(self._on_filter_changed)
        nav_layout.addWidget(self.filter_class_combo)
        self.sample_combo = QComboBox()
        self.sample_combo.setMinimumWidth(300)
        self.sample_combo.setMaxVisibleItems(20)
        self.sample_combo.view().setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOn)
        self.sample_combo.currentIndexChanged.connect(self.on_combo_change)
        nav_layout.addWidget(self.sample_combo)
        self.sample_label = QLabel("0 images")
        nav_layout.addWidget(self.sample_label)
        row1.addWidget(nav_group)

        row1.addStretch()

        # Settings gear button (right side) — editor mode selector
        self.mode_label = QLabel(self.editor_mode)
        self.mode_label.setStyleSheet("color: #aaaaaa; font-style: italic;")
        row1.addWidget(self.mode_label)
        self.settings_btn = QToolButton()
        self.settings_btn.setIcon(self._make_gear_icon())
        self.settings_btn.setToolTip("Settings")
        self.settings_btn.setPopupMode(QToolButton.InstantPopup)
        self.settings_btn.setFixedSize(28, 28)
        self.settings_btn.setStyleSheet(
            "QToolButton { border: none; }"
            "QToolButton::menu-indicator { image: none; }")
        settings_menu = QMenu(self)
        self._mode_action_group = QActionGroup(self)
        self._mode_action_group.setExclusive(True)
        for mode in [MODE_CREATE_GT, MODE_CORRECT_GT, MODE_CORRECT_PRED]:
            action = QAction(mode, self)
            action.setCheckable(True)
            if mode == self.editor_mode:
                action.setChecked(True)
            action.triggered.connect(lambda checked, m=mode: self._on_mode_changed(m))
            self._mode_action_group.addAction(action)
            settings_menu.addAction(action)
        self.settings_btn.setMenu(settings_menu)
        row1.addWidget(self.settings_btn)

        # Info button — quick reference guide
        self.info_btn = QToolButton()
        self.info_btn.setIcon(
            self.style().standardIcon(QStyle.SP_MessageBoxInformation))
        self.info_btn.setFixedSize(28, 28)
        self.info_btn.setStyleSheet(
            "QToolButton { border: none; }"
            "QToolButton::menu-indicator { image: none; }")
        self.info_btn.setToolTip("Guide")
        self.info_btn.clicked.connect(self._show_info_guide)
        row1.addWidget(self.info_btn)

        # Instant tooltips for settings and info buttons
        self.settings_btn.installEventFilter(self)
        self.info_btn.installEventFilter(self)

        main_layout.addLayout(row1)

        # ── Row 2: Actions + QC ──────────────────────────────────────────
        controls = QHBoxLayout()

        # Class radio buttons (created here, added to modal page below)
        self.class_buttons = QButtonGroup(self)
        self._class_radios = []
        for cid, cname in ANNOTATED_CLASSES.items():
            rb = QRadioButton(f"{cid}: {CLASS_SHORT_NAMES[cid]}")
            rb.setToolTip(cname)
            if cid == 1:
                rb.setChecked(True)
            self.class_buttons.addButton(rb, cid)
            self._class_radios.append(rb)
        self.class_buttons.buttonToggled.connect(self._update_class_radio_styles)
        self._update_class_radio_styles()

        # Action buttons — stacked: page 0 = main, page 1 = modal
        action_group = QGroupBox("Actions")
        action_outer = QHBoxLayout(action_group)
        action_outer.setContentsMargins(4, 2, 4, 2)
        self.action_stack = QStackedWidget()
        action_outer.addWidget(self.action_stack)

        # Page 0: main actions
        main_page = QWidget()
        main_layout_inner = QHBoxLayout(main_page)
        main_layout_inner.setContentsMargins(0, 0, 0, 0)

        # Tool mode radio: Node vs Brush
        main_layout_inner.addWidget(QLabel("Mode:"))
        self.tool_mode_group = QButtonGroup(self)
        self.node_radio = QRadioButton("Node")
        self.node_radio.setChecked(True)
        self.node_radio.setToolTip("Use node/vertex tool for drawing and editing")
        self.brush_radio = QRadioButton("Brush")
        self.brush_radio.setToolTip("Use brush tool for drawing and editing")
        self.tool_mode_group.addButton(self.node_radio, 0)
        self.tool_mode_group.addButton(self.brush_radio, 1)
        main_layout_inner.addWidget(self.node_radio)
        main_layout_inner.addWidget(self.brush_radio)

        self.draw_btn = QPushButton("Draw (N)")
        self.draw_btn.clicked.connect(self._on_draw)
        main_layout_inner.addWidget(self.draw_btn)
        self.edit_btn = QPushButton("Edit (Enter)")
        self.edit_btn.clicked.connect(self._on_edit)
        main_layout_inner.addWidget(self.edit_btn)
        self.delete_btn = QPushButton("Delete (Del)")
        self.delete_btn.clicked.connect(self.delete_selected)
        main_layout_inner.addWidget(self.delete_btn)
        self.copy_pred_btn = QPushButton("Copy All Pred->GT")
        self.copy_pred_btn.clicked.connect(self.copy_all_ref_to_edit)
        main_layout_inner.addWidget(self.copy_pred_btn)
        self.copy_sel_btn = QPushButton("Copy Selected (Ctrl+C)")
        self.copy_sel_btn.clicked.connect(self.copy_selected_ref_to_edit)
        main_layout_inner.addWidget(self.copy_sel_btn)
        self.action_stack.addWidget(main_page)

        # Page 1: modal cancel/confirm + class radio buttons
        modal_page = QWidget()
        modal_layout = QHBoxLayout(modal_page)
        modal_layout.setContentsMargins(0, 0, 0, 0)
        self.modal_label = QLabel("")
        self.modal_label.setStyleSheet("font-weight: bold; color: #ffaa00;")
        modal_layout.addWidget(self.modal_label)
        # Class radio buttons (shown in modal for drawing/brush)
        for rb in self._class_radios:
            modal_layout.addWidget(rb)
        self.modal_undo_btn = QPushButton("Undo (Ctrl+Z)")
        self.modal_undo_btn.setStyleSheet("background-color: #336699; color: white; font-weight: bold;")
        self.modal_undo_btn.clicked.connect(self.undo)
        modal_layout.addWidget(self.modal_undo_btn)
        self.modal_cancel_btn = QPushButton("Cancel (Esc)")
        self.modal_cancel_btn.setStyleSheet("background-color: #993333; color: white; font-weight: bold;")
        self.modal_cancel_btn.clicked.connect(self.modal_cancel)
        modal_layout.addWidget(self.modal_cancel_btn)
        self.modal_confirm_btn = QPushButton("Confirm (Enter)")
        self.modal_confirm_btn.setStyleSheet("background-color: #339933; color: white; font-weight: bold;")
        self.modal_confirm_btn.clicked.connect(self.modal_confirm)
        modal_layout.addWidget(self.modal_confirm_btn)
        self.action_stack.addWidget(modal_page)

        # QC section
        qc_group = QGroupBox("Quality Check")
        qc_layout = QHBoxLayout(qc_group)
        qc_layout.setContentsMargins(4, 2, 4, 2)
        self.qc_btn = QPushButton("QC Current")
        self.qc_btn.setToolTip("Run QC checks on current image")
        self.qc_btn.clicked.connect(self._on_qc_current)
        qc_layout.addWidget(self.qc_btn)
        self.qc_all_btn = QPushButton("QC All")
        self.qc_all_btn.setToolTip("Run QC checks on all filtered images")
        self.qc_all_btn.clicked.connect(self._on_qc_all)
        qc_layout.addWidget(self.qc_all_btn)
        self.qc_auto_cb = QCheckBox("Auto")
        self.qc_auto_cb.setToolTip("Automatically run QC after each edit")
        self.qc_auto_cb.setChecked(True)
        self.qc_auto_cb.toggled.connect(self._on_auto_qc_toggled)
        qc_layout.addWidget(self.qc_auto_cb)
        self.qc_stats_label = QLabel("")
        self.qc_stats_label.setStyleSheet("color: #aaaaaa;")
        qc_layout.addWidget(self.qc_stats_label)
        controls.addWidget(qc_group)

        controls.addWidget(action_group)

        controls.addStretch()
        main_layout.addLayout(controls)

        # ── Image panels ─────────────────────────────────────────────────
        self.prev_btn = QPushButton("<")
        self.prev_btn.setFixedWidth(24)
        self.prev_btn.setMinimumHeight(120)
        self.prev_btn.setToolTip("Previous sample")
        self.prev_btn.setStyleSheet("font-size: 18px; font-weight: bold;")
        self.prev_btn.clicked.connect(self.prev_sample)

        self.splitter = QSplitter(Qt.Horizontal)
        self.original_canvas = PolygonCanvas("Original Image", editable=False)
        self.splitter.addWidget(self.original_canvas)
        self.edit_canvas = PolygonCanvas("Editable", editable=True)
        self.splitter.addWidget(self.edit_canvas)
        self.ref_canvas = PolygonCanvas("Reference", editable=False, selectable=True)
        self.splitter.addWidget(self.ref_canvas)
        self.splitter.setSizes([1, 1, 1])
        self.splitter.setChildrenCollapsible(False)

        self.next_btn = QPushButton(">")
        self.next_btn.setFixedWidth(24)
        self.next_btn.setMinimumHeight(120)
        self.next_btn.setToolTip("Next sample")
        self.next_btn.setStyleSheet("font-size: 18px; font-weight: bold;")
        self.next_btn.clicked.connect(self.next_sample)

        panel_widget = QWidget()
        panel_row = QHBoxLayout(panel_widget)
        panel_row.setContentsMargins(0, 0, 0, 0)
        panel_row.addWidget(self.prev_btn)
        panel_row.addWidget(self.splitter, stretch=1)
        panel_row.addWidget(self.next_btn)
        main_layout.addWidget(panel_widget, stretch=1)

        # QC status overlay — top-right corner of image panel
        self.qc_label = QLabel("", panel_widget)
        self.qc_label.setStyleSheet(
            "background-color: rgba(0, 0, 0, 160); padding: 4px 10px;"
            " border-bottom-left-radius: 6px; font-weight: bold; font-size: 13px;")
        self.qc_label.installEventFilter(self)
        self.qc_label.adjustSize()
        self.qc_label.hide()
        self.qc_label.raise_()
        self._panel_widget = panel_widget

        # Status bar: [status msg] ... Visibility ... | Brightness | Gamma | Reset | Home
        self.status_bar = QStatusBar()

        # Visibility checkboxes with short labels
        vis_label = QLabel("Visibility:")
        self.status_bar.addPermanentWidget(vis_label)
        self.vis_checks = {}
        for cid, cname in ANNOTATED_CLASSES.items():
            color = CLASS_COLORS_RGB.get(cid, (128, 128, 128))
            cb = QCheckBox(CLASS_SHORT_NAMES[cid])
            cb.setChecked(True)
            cb.setStyleSheet(f"color: rgb({color[0]},{color[1]},{color[2]}); font-weight: bold;")
            cb.setToolTip(cname)
            cb.toggled.connect(lambda checked, c=cid: self._on_visibility_toggled(c, checked))
            self.status_bar.addPermanentWidget(cb)
            self.vis_checks[cid] = cb

        # Separator
        sep = QFrame()
        sep.setFrameShape(QFrame.VLine)
        sep.setFixedWidth(2)
        self.status_bar.addPermanentWidget(sep)

        # Brightness slider (compact)
        self.status_bar.addPermanentWidget(QLabel("Brightness:"))
        self.brightness_slider = QSlider(Qt.Horizontal)
        self.brightness_slider.setRange(-100, 100)
        self.brightness_slider.setValue(0)
        self.brightness_slider.setFixedWidth(80)
        self.brightness_slider.setToolTip("Adjust brightness (for display)")
        self.brightness_slider.valueChanged.connect(self._on_brightness_changed)
        self.status_bar.addPermanentWidget(self.brightness_slider)
        self.brightness_label = QLabel("0")
        self.brightness_label.setFixedWidth(25)
        self.status_bar.addPermanentWidget(self.brightness_label)

        # Gamma slider (compact)
        self.status_bar.addPermanentWidget(QLabel(" γ:"))
        self.gamma_slider = QSlider(Qt.Horizontal)
        self.gamma_slider.setRange(10, 300)
        self.gamma_slider.setValue(100)
        self.gamma_slider.setFixedWidth(80)
        self.gamma_slider.setToolTip("Adjust contrast (for display)")
        self.gamma_slider.valueChanged.connect(self._on_gamma_changed)
        self.status_bar.addPermanentWidget(self.gamma_slider)
        self.gamma_label = QLabel("1.0")
        self.gamma_label.setFixedWidth(25)
        self.status_bar.addPermanentWidget(self.gamma_label)

        # Reset button for sliders
        reset_display_btn = QPushButton("Reset")
        reset_display_btn.setFixedWidth(50)
        reset_display_btn.setToolTip("Reset brightness and gamma to defaults")
        reset_display_btn.clicked.connect(self._reset_display_adjustments)
        self.status_bar.addPermanentWidget(reset_display_btn)

        sep2 = QFrame()
        sep2.setFrameShape(QFrame.VLine)
        sep2.setFixedWidth(2)
        self.status_bar.addPermanentWidget(sep2)

        home_btn = QPushButton("Home")
        home_btn.setFixedWidth(60)
        home_btn.setToolTip("Reset zoom and center all panels (H)")
        home_btn.clicked.connect(self.home_view)
        self.status_bar.addPermanentWidget(home_btn)
        self.setStatusBar(self.status_bar)
        self.update_status("Select a data folder and mode to begin.")

        # Alias for backward compat (internal references use gt_canvas)
        self.gt_canvas = self.edit_canvas
        self.pred_canvas = self.ref_canvas

        # Apply initial mode layout
        self._apply_mode_layout()

    def _apply_mode_layout(self):
        """Configure panels and labels for the current editor mode."""
        mode = self.editor_mode
        if mode == MODE_CORRECT_GT:
            self.edit_canvas.title = "Ground Truth (Editable)"
            self.ref_canvas.title = "Prediction (Reference)"
            self.ref_canvas.setVisible(True)
            self.copy_pred_btn.setVisible(True)
            self.copy_pred_btn.setText("Copy All Pred->GT")
            self.copy_sel_btn.setVisible(True)
            self.splitter.setSizes([1, 1, 1])
        elif mode == MODE_CORRECT_PRED:
            self.edit_canvas.title = "Prediction (Editable)"
            self.ref_canvas.title = ""
            self.ref_canvas.setVisible(False)
            self.copy_pred_btn.setVisible(False)
            self.copy_sel_btn.setVisible(False)
            self.splitter.setSizes([1, 1])
        else:  # MODE_CREATE_GT
            self.edit_canvas.title = "Ground Truth (Editable)"
            self.ref_canvas.title = ""
            self.ref_canvas.setVisible(False)
            self.copy_pred_btn.setVisible(False)
            self.copy_sel_btn.setVisible(False)
            self.splitter.setSizes([1, 1])
        # Refresh titles
        self.edit_canvas.update()
        self.ref_canvas.update()

    # ── Display brightness / gamma (display only) ────────────────────────

    def _on_brightness_changed(self, value: int):
        self._brightness = value
        self.brightness_label.setText(str(value))
        self._apply_display_adjustments()

    def _on_gamma_changed(self, value: int):
        self._gamma = value
        self.gamma_label.setText(f"{value / 100:.1f}")
        self._apply_display_adjustments()

    def _reset_display_adjustments(self):
        self.brightness_slider.setValue(0)
        self.gamma_slider.setValue(100)

    def _apply_display_adjustments(self):
        """Apply brightness and gamma to the raw image for display only."""
        if self._raw_image is None:
            return
        img = self._raw_image.astype(np.float32)

        # Brightness: shift pixel values
        if self._brightness != 0:
            img = img + self._brightness * 255.0 / 100.0

        # Gamma correction: gamma < 1 brightens midtones, gamma > 1 darkens
        gamma = self._gamma / 100.0
        if gamma != 1.0:
            img = np.clip(img, 0, 255) / 255.0
            img = np.power(img, gamma) * 255.0

        display_img = np.clip(img, 0, 255).astype(np.uint8)

        # Update all canvases with adjusted image
        for canvas in self._all_canvases():
            canvas.set_image(display_img)

    def _in_modal_mode(self) -> bool:
        """Return True if in any modal mode (drawing, editing, brushing, deleting)."""
        return (self._deleting_mode or self.edit_canvas.drawing_mode
                or self.edit_canvas.editing_mode or self.edit_canvas.brush_mode)

    def eventFilter(self, obj, event):
        """Intercept events: arrow keys for navigation, instant tooltip for QC label."""
        if obj is self.qc_label and event.type() == event.Enter:
            tip = self.qc_label.toolTip()
            if tip:
                QToolTip.showText(self.qc_label.mapToGlobal(QPoint(0, self.qc_label.height())), tip, self.qc_label)
                return True
        if event.type() == event.KeyPress:
            if event.key() == Qt.Key_Left:
                if not self._in_modal_mode():
                    self.prev_sample()
                return True
            if event.key() == Qt.Key_Right:
                if not self._in_modal_mode():
                    self.next_sample()
                return True
            if event.key() in (Qt.Key_Up, Qt.Key_Down):
                return True
        return super().eventFilter(obj, event)

    def setup_shortcuts(self):
        QShortcut(QKeySequence(Qt.Key_N), self, self._on_draw)
        QShortcut(QKeySequence(Qt.Key_Delete), self, self.delete_selected)
        QShortcut(QKeySequence(Qt.Key_Backspace), self, self.delete_selected)
        QShortcut(QKeySequence(Qt.Key_Escape), self, self.escape_action)
        QShortcut(QKeySequence(Qt.Key_Return), self, self.enter_action)
        QShortcut(QKeySequence(Qt.Key_Enter), self, self.enter_action)
        QShortcut(QKeySequence(Qt.Key_Space), self, self.enter_action)
        QShortcut(QKeySequence(Qt.Key_R), self, self.split_ring)
        QShortcut(QKeySequence("Ctrl+C"), self, self.copy_selected_ref_to_edit)
        QShortcut(QKeySequence("Ctrl+Z"), self, self.undo)
        QShortcut(QKeySequence("Ctrl+S"), self, self.save_annotations)
        QShortcut(QKeySequence("Ctrl+Shift+Z"), self, self.redo)

    # ── Sample loading ────────────────────────────────────────────────────

    def load_sample(self, idx: int):
        if self.modified:
            reply = QMessageBox.question(
                self, "Unsaved Changes",
                "You have unsaved changes. Save before switching?",
                QMessageBox.Save | QMessageBox.Discard | QMessageBox.Cancel)
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
        # Exit any active modal mode
        if self.edit_canvas.brush_mode:
            self.edit_canvas.cancel_brush_mode()
        if self.edit_canvas.drawing_mode:
            self.edit_canvas.cancel_drawing()
        if self.edit_canvas.editing_mode:
            self._exit_editing_raw()
        self._exit_modal()
        sample = self.samples[idx]

        for canvas in self._all_canvases():
            canvas.reset_view()

        img = load_sample_normalized(sample)
        img_uint8 = to_uint8(img)
        h, w = img.shape[:2]

        # Store raw image for display adjustments (brightness/gamma)
        self._raw_image = img_uint8.copy()
        self._apply_display_adjustments()
        self.original_canvas.set_polygons([])

        ann_dir = self._annotation_dir()
        pred_dir = self._prediction_dir()
        stem = self._file_stem(sample)
        ann_file = ann_dir / f"{stem}.txt" if ann_dir else None

        # Resolve prediction file: prefer .txt, fall back to .npz
        pred_file = None
        if pred_dir:
            txt_path = pred_dir / f"{stem}.txt"
            npz_path = pred_dir / f"{stem}.npz"
            if txt_path.exists():
                pred_file = txt_path
            elif npz_path.exists():
                pred_file = npz_path

        def _load_predictions(path, w, h):
            if path is None or not path.exists():
                return []
            if path.suffix == ".npz":
                return parse_npz_predictions(path, w, h)
            return parse_yolo_annotations(path, w, h)

        if self.editor_mode == MODE_CORRECT_GT:
            # Edit: GT annotations, Reference: predictions only
            gt_polys = parse_yolo_annotations(ann_file, w, h) if ann_file and ann_file.exists() else []
            pred_polys = _load_predictions(pred_file, w, h)
            self.edit_canvas.set_polygons(gt_polys)
            self.ref_canvas.set_polygons(pred_polys)
        elif self.editor_mode == MODE_CORRECT_PRED:
            # Edit: predictions (to be corrected), no reference
            pred_polys = _load_predictions(pred_file, w, h)
            self.edit_canvas.set_polygons(pred_polys)
            self.ref_canvas.set_polygons([])
        else:  # MODE_CREATE_GT
            # Edit: existing annotations if any, else empty
            gt_polys = parse_yolo_annotations(ann_file, w, h) if ann_file and ann_file.exists() else []
            self.edit_canvas.set_polygons(gt_polys)
            self.ref_canvas.set_polygons([])

        self.sample_combo.blockSignals(True)
        self.sample_combo.setCurrentIndex(idx)
        self.sample_combo.blockSignals(False)
        self.sample_label.setText(f"{len(self.samples)} images")
        self._update_qc_label()
        self._schedule_auto_qc()
        self.update_status(f"Loaded: {self._display_name(sample)}")

    def on_combo_change(self, idx: int):
        if idx != self.current_idx and 0 <= idx < len(self.samples):
            self.load_sample(idx)

    def prev_sample(self):
        if self._in_modal_mode():
            return
        if self.current_idx > 0:
            self.load_sample(self.current_idx - 1)

    def next_sample(self):
        if self._in_modal_mode():
            return
        if self.current_idx < len(self.samples) - 1:
            self.load_sample(self.current_idx + 1)

    # ── Folder / mode callbacks ───────────────────────────────────────────

    def _browse_data_dir(self):
        current = self.data_dir_entry.text() or ""
        folder = QFileDialog.getExistingDirectory(self, "Select Data Directory", current)
        if folder:
            self.data_dir_entry.setText(folder)
            self._on_data_dir_changed()

    def _on_data_dir_changed(self):
        text = self.data_dir_entry.text().strip()
        self.data_dir = Path(text) if text else None
        self._reload_samples()

    @staticmethod
    def _make_gear_icon(size: int = 64) -> QIcon:
        """Draw a gear icon using QPainter."""
        pixmap = QPixmap(size, size)
        pixmap.fill(Qt.transparent)
        p = QPainter(pixmap)
        p.setRenderHint(QPainter.Antialiasing)
        p.setPen(Qt.NoPen)
        p.setBrush(QBrush(QColor(200, 200, 200)))
        cx, cy = size / 2, size / 2
        r_outer = size * 0.44
        r_inner = size * 0.30
        r_center = size * 0.15
        n_teeth = 8
        points = []
        for i in range(n_teeth * 2):
            angle = math.pi * 2 * i / (n_teeth * 2) - math.pi / 2
            r = r_outer if i % 2 == 0 else r_inner
            points.append(QPointF(cx + r * math.cos(angle),
                                  cy + r * math.sin(angle)))
        p.drawPolygon(QPolygonF(points))
        # Cut out center hole
        p.setCompositionMode(QPainter.CompositionMode_Clear)
        p.drawEllipse(QPointF(cx, cy), r_center, r_center)
        p.end()
        return QIcon(pixmap)

    def eventFilter(self, obj, event):
        if event.type() == QEvent.Enter and hasattr(obj, 'toolTip') and obj.toolTip():
            QToolTip.showText(obj.mapToGlobal(QPoint(0, obj.height())), obj.toolTip(), obj)
        return super().eventFilter(obj, event)

    def _show_info_guide(self):
        """Show a non-modal quick-reference guide dialog (singleton)."""
        # If already open, just raise it
        if self._info_dialog is not None and self._info_dialog.isVisible():
            self._info_dialog.raise_()
            self._info_dialog.activateWindow()
            return

        dlg = QDialog(self)
        dlg.setWindowTitle("Quick Guide")
        dlg.setAttribute(Qt.WA_DeleteOnClose)
        dlg.finished.connect(lambda: setattr(self, '_info_dialog', None))
        dlg.resize(520, 620)

        layout = QVBoxLayout(dlg)
        layout.setContentsMargins(0, 0, 0, 0)
        browser = QTextBrowser()
        browser.setOpenExternalLinks(False)
        browser.setHtml(
            "<style>"
            "body { font-size: 14px; color: #e0e0e0; background: #353535;"
            " margin: 12px 16px; }"
            "table { border-collapse: collapse; margin: 6px 0 14px 0; width: 100%; }"
            "th { padding: 5px 12px; text-align: left; "
            "background: #2a82da; color: #ffffff; font-size: 13px; }"
            "td { padding: 5px 12px; background: #2e2e2e; color: #e0e0e0; font-size: 13px; }"
            "h2 { color: #ffffff; font-size: 18px; margin-top: 0; }"
            "h3 { color: #2a82da; margin: 14px 0 6px 0; font-size: 15px; }"
            "kbd { background: #2e2e2e; color: #e0e0e0; padding: 2px 6px; "
            "border-radius: 3px; font-family: monospace; font-size: 12px; }"
            "</style>"

            "<h2>Quick Guide</h2>"

            "<h3>Editor Modes</h3>"
            "<table>"
            "<tr><th>Mode</th><th>Purpose</th></tr>"
            "<tr><td><b>Create GT</b></td>"
            "<td>Draw annotations from scratch on raw images.<br>"
            "Only the image folder is needed.</td></tr>"
            "<tr><td><b>Correct GT with Pred</b></td>"
            "<td>Edit existing ground-truth annotations using<br>"
            "model predictions as a visual reference.<br>"
            "Requires image + annotation + prediction folders.</td></tr>"
            "<tr><td><b>Correct Predictions</b></td>"
            "<td>Edit model predictions and save them<br>"
            "for downstream analysis.<br>"
            "Requires image + prediction folders.</td></tr>"
            "</table>"

            "<h3>QC Rules</h3>"
            "<table>"
            "<tr><th>#</th><th>Check</th></tr>"
            "<tr><td>1</td><td>All polygons must be inside Whole Root</td></tr>"
            "<tr><td>2</td><td>I.Exo must be inside O.Exo</td></tr>"
            "<tr><td>3</td><td>O.Endo must be inside I.Exo</td></tr>"
            "<tr><td>4</td><td>I.Endo must be inside O.Endo</td></tr>"
            "<tr><td>5</td><td>Aerenchyma must be in cortex only<br>"
            "<span style='color:#888'>(between root boundary &amp; O.Endo)</span></td></tr>"
            "<tr><td>6</td><td>Required classes: Root, O.Endo, I.Endo,<br>"
            "O.Exo, I.Exo</td></tr>"
            "</table>"

            "<h3>Keyboard Shortcuts</h3>"
            "<table>"
            "<tr><th>Key</th><th>Action</th></tr>"
            "<tr><td><kbd>\u2190</kbd> / <kbd>\u2192</kbd></td>"
            "<td>Previous / Next sample</td></tr>"
            "<tr><td><kbd>N</kbd></td>"
            "<td>Draw new polygon (Node or Brush)</td></tr>"
            "<tr><td><kbd>Enter</kbd> / <kbd>Space</kbd></td>"
            "<td>Confirm action, or edit selected polygon</td></tr>"
            "<tr><td><kbd>Esc</kbd></td>"
            "<td>Cancel action, or deselect polygon</td></tr>"
            "<tr><td><kbd>Ctrl+S</kbd></td><td>Save annotations</td></tr>"
            "<tr><td><kbd>R</kbd></td>"
            "<td>Split selected ring polygon (endo/exo)</td></tr>"
            "<tr><td><kbd>Del</kbd> / <kbd>Backspace</kbd></td>"
            "<td>Delete selected polygon or vertices</td></tr>"
            "<tr><td><kbd>Ctrl+Z</kbd></td><td>Undo</td></tr>"
            "<tr><td><kbd>C</kbd></td>"
            "<td>Copy all prediction polygons to GT (Correct GT mode)</td></tr>"
            "<tr><td><kbd>Ctrl+C</kbd></td>"
            "<td>Copy selected prediction polygon to GT (Correct GT mode)</td></tr>"
            "<tr><td><kbd>Ctrl+Scroll</kbd></td>"
            "<td>Change brush size</td></tr>"
            "<tr><td><kbd>Scroll</kbd></td><td>Zoom</td></tr>"
            "</table>"

            "<h3>Mouse (Node Mode)</h3>"
            "<table>"
            "<tr><th>Input</th><th>Action</th></tr>"
            "<tr><td><kbd>Left Click</kbd></td>"
            "<td>Place a node (drawing)</td></tr>"
            "<tr><td><kbd>Left Click</kbd></td>"
            "<td>Select a single node (editing)</td></tr>"
            "<tr><td><kbd>Left Drag</kbd></td>"
            "<td>Select multiple nodes (editing)</td></tr>"
            "<tr><td><kbd>Right Click</kbd> / <kbd>Middle Click</kbd></td>"
            "<td>Pan / move the image</td></tr>"
            "</table>"

            "<h3>Mouse (Brush Mode)</h3>"
            "<table>"
            "<tr><th>Input</th><th>Action</th></tr>"
            "<tr><td><kbd>Left Click</kbd></td>"
            "<td>Erase area</td></tr>"
            "<tr><td><kbd>Shift + Left Click</kbd></td>"
            "<td>Paint / add area</td></tr>"
            "<tr><td><kbd>Right Click</kbd> / <kbd>Middle Click</kbd></td>"
            "<td>Pan / move the image</td></tr>"
            "</table>"
        )
        layout.addWidget(browser)

        self._info_dialog = dlg
        dlg.show()

    def _on_mode_changed(self, mode_text: str):
        if mode_text == self.editor_mode:
            return
        # Validate before switching
        err = self._validate_dirs_for_mode(mode_text)
        if err:
            QMessageBox.warning(self, "Missing Directories", err)
            # Revert the action group check to current mode
            for action in self._mode_action_group.actions():
                if action.text() == self.editor_mode:
                    action.setChecked(True)
            return
        self.editor_mode = mode_text
        self.mode_label.setText(mode_text)
        self._apply_mode_layout()
        self._reload_samples()

    # ── Styling helpers ───────────────────────────────────────────────────

    def set_drawing_class(self, class_id: int):
        btn = self.class_buttons.button(class_id)
        if btn:
            btn.setChecked(True)
        self.update_status(f"Drawing class: {class_id}: {ANNOTATED_CLASSES.get(class_id, 'Unknown')}")

    def _update_class_radio_styles(self):
        for cid, _ in ANNOTATED_CLASSES.items():
            btn = self.class_buttons.button(cid)
            if btn is None:
                continue
            color = CLASS_COLORS_RGB.get(cid, (128, 128, 128))
            if btn.isChecked():
                btn.setStyleSheet(
                    f"QRadioButton {{ color: rgb({color[0]},{color[1]},{color[2]}); font-weight: bold; }}")
            else:
                btn.setStyleSheet("QRadioButton { color: rgb(180,180,180); font-weight: normal; }")
        # Update polygon class if editing an existing polygon
        if self._editing_polygon_idx is not None:
            new_class = self.class_buttons.checkedId()
            idx = self._editing_polygon_idx
            if 0 <= idx < len(self.edit_canvas.polygons) and new_class >= 0:
                self.edit_canvas.polygons[idx]["class_id"] = new_class
                # Also update brush class if in brush mode
                if self.edit_canvas.brush_mode:
                    self.edit_canvas._brush_class_id = new_class
                self.edit_canvas.update()

    def _on_visibility_toggled(self, class_id: int, checked: bool):
        color = CLASS_COLORS_RGB.get(class_id, (128, 128, 128))
        cb = self.vis_checks.get(class_id)
        if cb:
            if checked:
                cb.setStyleSheet(f"color: rgb({color[0]},{color[1]},{color[2]}); font-weight: bold;")
            else:
                r, g, b = color
                cb.setStyleSheet(f"color: rgb({r//3},{g//3},{b//3}); font-weight: normal;")
        self.toggle_class_visibility(class_id, checked)

    def toggle_class_visibility(self, class_id: int, visible: bool):
        for canvas in self._all_canvases():
            if visible:
                canvas.hidden_classes.discard(class_id)
            else:
                canvas.hidden_classes.add(class_id)
                if (canvas.selected_idx is not None
                        and 0 <= canvas.selected_idx < len(canvas.polygons)
                        and canvas.polygons[canvas.selected_idx]["class_id"] == class_id):
                    canvas.selected_idx = None
                    canvas._selected_vertex = None
                    canvas._selected_vertices.clear()
            canvas.update()

    # ── Tool dispatchers (Node / Brush radio) ───────────────────────────

    def _on_draw(self):
        """Draw new polygon using the selected tool (Node or Brush)."""
        if self.brush_radio.isChecked():
            # Brush draw: deselect so brush starts a new polygon
            self.edit_canvas.selected_idx = None
            self.toggle_brush()
        else:
            self.toggle_drawing()

    def _on_edit(self):
        """Edit selected polygon using the selected tool (Node or Brush)."""
        if self.edit_canvas.selected_idx is None:
            self.update_status("Select a polygon first to edit")
            return
        # Set class radio to match the polygon being edited
        idx = self.edit_canvas.selected_idx
        poly = self.edit_canvas.polygons[idx]
        self._editing_polygon_idx = idx
        btn = self.class_buttons.button(poly["class_id"])
        if btn:
            btn.setChecked(True)
        if self.brush_radio.isChecked():
            self.toggle_brush()
        else:
            self.toggle_vertex_editing()

    # ── Modal (draw/edit) ─────────────────────────────────────────────────

    def _enter_modal(self, label: str, hide_class_radios: bool = False, hide_undo: bool = False):
        self.modal_label.setText(label)
        for rb in self._class_radios:
            rb.setVisible(not hide_class_radios)
        self.modal_undo_btn.setVisible(not hide_undo)
        self.action_stack.setCurrentIndex(1)
        # Disable navigation and controls during modal
        self._set_nav_controls_enabled(False)

    def _exit_modal(self):
        self.action_stack.setCurrentIndex(0)
        self._editing_polygon_idx = None
        self._deleting_mode = False
        # Restore class radios and undo button visibility
        for rb in self._class_radios:
            rb.setVisible(True)
        self.modal_undo_btn.setVisible(True)
        # Re-enable navigation and controls
        self._set_nav_controls_enabled(True)

    def _set_nav_controls_enabled(self, enabled: bool):
        """Enable or disable navigation, folder, mode, and filter controls."""
        self.prev_btn.setEnabled(enabled)
        self.next_btn.setEnabled(enabled)
        self.sample_combo.setEnabled(enabled)
        self.filter_type_combo.setEnabled(enabled)
        if enabled:
            # Respect filter state for class combo
            is_none = self.filter_type_combo.currentText() == "None"
            self.filter_class_combo.setEnabled(not is_none)
        else:
            self.filter_class_combo.setEnabled(False)
        self.data_dir_entry.setEnabled(enabled)
        self.browse_btn.setEnabled(enabled)
        self.settings_btn.setEnabled(enabled)

    def toggle_drawing(self):
        if self.edit_canvas.drawing_mode:
            self.modal_cancel()
            return
        if self.edit_canvas.editing_mode:
            self._exit_editing_raw()
        self._mode_entry_snapshot = copy.deepcopy(self.edit_canvas.polygons)
        self._undo_depth_at_mode_entry = len(self._undo_stack)
        class_id = self.class_buttons.checkedId()
        self.edit_canvas.start_drawing(class_id)
        self._enter_modal("Drawing")
        self.update_status(f"Drawing class {class_id}: Click to add points. Enter = confirm, Esc = cancel.")

    def toggle_editing(self):
        """Enter/button: enter brush mode on selected polygon (default editing mode)."""
        if self.edit_canvas.brush_mode:
            self.modal_confirm()
            return
        if self.edit_canvas.editing_mode:
            self.modal_confirm()
            return
        if self.edit_canvas.selected_idx is None:
            self.update_status("Select a polygon first to edit")
            return
        # Default to brush mode for editing
        self.toggle_brush()

    def toggle_vertex_editing(self):
        """V key: enter vertex (node) editing mode on selected polygon."""
        if self.edit_canvas.editing_mode:
            self.modal_confirm()
            return
        if self.edit_canvas.selected_idx is None:
            self.update_status("Select a polygon first, then press V to edit vertices")
            return
        if self.edit_canvas.drawing_mode:
            self.edit_canvas.cancel_drawing()
            self._exit_modal()
        if self.edit_canvas.brush_mode:
            self.edit_canvas.cancel_brush_mode()
            self._exit_modal()
        self._mode_entry_snapshot = copy.deepcopy(self.edit_canvas.polygons)
        self._undo_depth_at_mode_entry = len(self._undo_stack)
        self.edit_canvas.editing_mode = True
        self._editing_polygon_idx = self.edit_canvas.selected_idx
        # Sync class radio to polygon being edited
        poly = self.edit_canvas.polygons[self.edit_canvas.selected_idx]
        btn = self.class_buttons.button(poly["class_id"])
        if btn:
            btn.setChecked(True)
        self.edit_canvas.update()
        self._enter_modal("Editing vertices")
        self.update_status("Vertices: Drag to move, Shift+click=add, Del=remove. Enter=confirm, Esc=cancel.")

    def toggle_brush(self):
        if self.edit_canvas.brush_mode:
            self.modal_confirm()
            return
        # Exit other modes
        if self.edit_canvas.drawing_mode:
            self.edit_canvas.cancel_drawing()
            self._exit_modal()
        if self.edit_canvas.editing_mode:
            self._exit_editing_raw()
            self._exit_modal()
        self._mode_entry_snapshot = copy.deepcopy(self.edit_canvas.polygons)
        self._undo_depth_at_mode_entry = len(self._undo_stack)
        canvas = self.edit_canvas
        if canvas.selected_idx is not None and 0 <= canvas.selected_idx < len(canvas.polygons):
            self._editing_polygon_idx = canvas.selected_idx
            # Sync class radio to polygon being edited
            poly = canvas.polygons[canvas.selected_idx]
            btn = self.class_buttons.button(poly["class_id"])
            if btn:
                btn.setChecked(True)
            canvas.start_brush_mode(canvas.selected_idx)
        else:
            class_id = self.class_buttons.checkedId()
            canvas._brush_class_id = class_id
            canvas.start_brush_mode(None)
        self._enter_modal("Brush painting")
        self.update_status(
            "Brush: Erase by default, Shift+paint to add, "
            "Ctrl+scroll=brush size. Enter=confirm, Esc=cancel.")

    def _exit_editing_raw(self):
        self.edit_canvas.editing_mode = False
        self.edit_canvas._dragging_vertex = None
        self.edit_canvas._selected_vertex = None
        self.edit_canvas._selected_vertices.clear()
        self.edit_canvas._rubber_band_start = None
        self.edit_canvas._clear_hover()
        self.edit_canvas.update()

    def modal_cancel(self):
        if self._mode_entry_snapshot is not None:
            self.edit_canvas.polygons = self._mode_entry_snapshot
            self._mode_entry_snapshot = None
        depth = self._undo_depth_at_mode_entry
        del self._undo_stack[depth:]
        self._redo_stack.clear()
        if self.edit_canvas.drawing_mode:
            self.edit_canvas.cancel_drawing()
        if self.edit_canvas.editing_mode:
            self._exit_editing_raw()
        if self.edit_canvas.brush_mode:
            self.edit_canvas.cancel_brush_mode()
        self.edit_canvas.selected_idx = None
        self.edit_canvas.update_display()
        self._exit_modal()
        self.update_status("Cancelled — reverted to previous state")

    def modal_confirm(self):
        confirmed = False
        if self._deleting_mode:
            # Perform the actual deletion
            removed = self.edit_canvas.delete_selected()
            depth = self._undo_depth_at_mode_entry
            del self._undo_stack[depth:]
            if self._mode_entry_snapshot is not None:
                self._undo_stack.append(self._mode_entry_snapshot)
            self._redo_stack.clear()
            self._mode_entry_snapshot = None
            self._exit_modal()
            if removed:
                self._mark_modified()
                self.update_status(f"Deleted polygon (class {removed['class_id']})")
                confirmed = True
            else:
                self.update_status("Delete failed — no polygon selected")
            if confirmed:
                self.save_annotations()
                self._schedule_auto_qc()
            return
        if self.edit_canvas.drawing_mode:
            result = self.edit_canvas.finish_drawing()
            if result:
                result['class_id'] = self.class_buttons.checkedId()
                depth = self._undo_depth_at_mode_entry
                del self._undo_stack[depth:]
                if self._mode_entry_snapshot is not None:
                    self._undo_stack.append(self._mode_entry_snapshot)
                self._redo_stack.clear()
                self.edit_canvas.polygons.append(result)
                self.edit_canvas.update_display()
                self._mark_modified()
                self._mode_entry_snapshot = None
                self._exit_modal()
                confirmed = True
            else:
                depth = self._undo_depth_at_mode_entry
                del self._undo_stack[depth:]
                self._redo_stack.clear()
                if self._mode_entry_snapshot is not None:
                    self.edit_canvas.polygons = self._mode_entry_snapshot
                    self.edit_canvas.update_display()
                self._mode_entry_snapshot = None
                self._exit_modal()
                self.update_status("Drawing cancelled (need at least 3 points)")
                return
        elif self.edit_canvas.brush_mode:
            orig_idx = self.edit_canvas._brush_orig_idx
            result = self.edit_canvas.finish_brush_mode()
            depth = self._undo_depth_at_mode_entry
            del self._undo_stack[depth:]
            if self._mode_entry_snapshot is not None:
                self._undo_stack.append(self._mode_entry_snapshot)
            self._redo_stack.clear()
            if result is not None:
                # Only override class for new polygons; keep original class when editing
                if orig_idx is None:
                    result['class_id'] = self.class_buttons.checkedId()
                if orig_idx is not None and 0 <= orig_idx < len(self.edit_canvas.polygons):
                    self.edit_canvas.polygons[orig_idx] = result
                else:
                    self.edit_canvas.polygons.append(result)
                self.edit_canvas.update_display()
                self._mark_modified()
                confirmed = True
            else:
                if self._mode_entry_snapshot is not None:
                    self.edit_canvas.polygons = self._mode_entry_snapshot
                    self.edit_canvas.update_display()
                self.update_status("Brush cancelled (empty mask)")
            self._mode_entry_snapshot = None
            self._exit_modal()
            if not confirmed:
                return
        elif self.edit_canvas.editing_mode:
            self._exit_editing_raw()
            depth = self._undo_depth_at_mode_entry
            del self._undo_stack[depth:]
            if self._mode_entry_snapshot is not None:
                self._undo_stack.append(self._mode_entry_snapshot)
            self._redo_stack.clear()
            self._mark_modified()
            confirmed = True
            self._mode_entry_snapshot = None
            self._exit_modal()

        # Auto-save on confirm
        if confirmed:
            self.save_annotations()
            self._schedule_auto_qc()

    def enter_action(self):
        """Enter key: confirm modal if active, otherwise edit selected polygon."""
        if self._deleting_mode or self.edit_canvas.drawing_mode or self.edit_canvas.editing_mode or self.edit_canvas.brush_mode:
            self.modal_confirm()
        elif self.edit_canvas.selected_idx is not None:
            self._on_edit()

    def escape_action(self):
        if self._deleting_mode or self.edit_canvas.drawing_mode or self.edit_canvas.editing_mode or self.edit_canvas.brush_mode:
            self.modal_cancel()
        else:
            self.edit_canvas.selected_idx = None
            self.edit_canvas.update()

    # ── Undo / redo ───────────────────────────────────────────────────────

    def push_undo(self):
        self._undo_stack.append(copy.deepcopy(self.edit_canvas.polygons))
        if len(self._undo_stack) > self._max_undo:
            self._undo_stack = self._undo_stack[-self._max_undo:]
        self._redo_stack.clear()

    def undo(self):
        if self.edit_canvas.drawing_mode:
            if self.edit_canvas.drawing_points:
                self.edit_canvas.drawing_points.pop()
                self.edit_canvas.update()
                self.update_status(f"Undo last point ({len(self.edit_canvas.drawing_points)} remaining)")
            else:
                self.update_status("No points to undo")
            return
        if not self._undo_stack:
            self.update_status("Nothing to undo")
            return
        was_editing = self.edit_canvas.editing_mode
        selected_idx = self.edit_canvas.selected_idx
        self._redo_stack.append(copy.deepcopy(self.edit_canvas.polygons))
        self.edit_canvas.polygons = self._undo_stack.pop()
        self.edit_canvas._dragging_vertex = None
        self.edit_canvas._selected_vertex = None
        self.edit_canvas._selected_vertices.clear()
        self.edit_canvas._clear_hover()
        if was_editing and selected_idx is not None and selected_idx < len(self.edit_canvas.polygons):
            self.edit_canvas.selected_idx = selected_idx
        else:
            self.edit_canvas.selected_idx = None
            if was_editing:
                self._exit_editing_raw()
        self.edit_canvas.update()
        self.modified = bool(self._undo_stack)
        self.update_status("Undo")

    def redo(self):
        if not self._redo_stack:
            self.update_status("Nothing to redo")
            return
        was_editing = self.edit_canvas.editing_mode
        selected_idx = self.edit_canvas.selected_idx
        self._undo_stack.append(copy.deepcopy(self.edit_canvas.polygons))
        self.edit_canvas.polygons = self._redo_stack.pop()
        self.edit_canvas._dragging_vertex = None
        if was_editing and selected_idx is not None and selected_idx < len(self.edit_canvas.polygons):
            self.edit_canvas.selected_idx = selected_idx
        else:
            self.edit_canvas.selected_idx = None
            if was_editing:
                self._exit_editing_raw()
        self.edit_canvas.update()
        self._mark_modified()
        self.update_status("Redo")

    # ── Delete ────────────────────────────────────────────────────────────

    def delete_selected(self):
        has_vertex_selection = (self.edit_canvas._selected_vertex is not None
                                or len(self.edit_canvas._selected_vertices) > 0)
        if (self.edit_canvas.editing_mode and has_vertex_selection
                and self.edit_canvas.selected_idx is not None):
            polygon = self.edit_canvas.polygons[self.edit_canvas.selected_idx]["polygon"]
            # Count how many would be deleted
            to_delete = set(self.edit_canvas._selected_vertices)
            if self.edit_canvas._selected_vertex is not None:
                to_delete.add(self.edit_canvas._selected_vertex)
            remaining = len(polygon) - len(to_delete)
            if remaining < 3:
                self.push_undo()
                removed = self.edit_canvas.delete_selected()
                self._exit_editing_raw()
                if removed:
                    self._mark_modified()
                    self.update_status("Deleted polygon (< 3 vertices would remain)")
                return
            self.push_undo()
            if self.edit_canvas.delete_selected_vertex():
                self._mark_modified()
                n = len(self.edit_canvas.polygons[self.edit_canvas.selected_idx]['polygon'])
                self.update_status(f"Deleted {len(to_delete)} vertex(es) ({n} remaining)")
            return
        # Whole-polygon deletion: enter confirmation modal
        if self.edit_canvas.selected_idx is None:
            self.update_status("No polygon selected to delete")
            return
        self._deleting_mode = True
        self._mode_entry_snapshot = copy.deepcopy(self.edit_canvas.polygons)
        self._undo_depth_at_mode_entry = len(self._undo_stack)
        idx = self.edit_canvas.selected_idx
        class_id = self.edit_canvas.polygons[idx]['class_id']
        class_name = CLASS_SHORT_NAMES.get(class_id, f"class {class_id}")
        self._enter_modal(f"Delete {class_name}?", hide_class_radios=True, hide_undo=True)
        self.update_status(f"Confirm delete of {class_name} polygon, or cancel to keep it.")

    # ── Split Ring ────────────────────────────────────────────────────────

    # Annotation class pairs for ring structures
    _RING_PAIRS = {
        2: (2, 3),   # outer endo → outer endo (2) + inner endo (3)
        3: (2, 3),   # inner endo → same pair
        4: (4, 5),   # outer exo → outer exo (4) + inner exo (5)
        5: (4, 5),   # inner exo → same pair
    }

    def split_ring(self):
        """Split the selected polygon into outer + inner ring polygons.

        Works on any polygon that forms a ring/donut shape. The polygon is
        converted to a filled mask, morphological closing fixes small gaps,
        then outer and inner contours are extracted as separate polygons.

        The class assignment depends on the selected polygon's class:
        - Class 2 or 3 → outer endo (2) + inner endo (3)
        - Class 4 or 5 → outer exo (4) + inner exo (5)
        - Other classes → outer keeps original class, inner = original + 1
        """
        canvas = self.edit_canvas
        if canvas.selected_idx is None or canvas.base_image is None:
            self.update_status("Select a polygon first, then press R to split ring")
            return
        if not (0 <= canvas.selected_idx < len(canvas.polygons)):
            return

        poly_data = canvas.polygons[canvas.selected_idx]
        class_id = poly_data["class_id"]
        polygon = poly_data["polygon"]
        h, w = canvas.base_image.shape[:2]

        # Convert polygon to filled mask
        mask = np.zeros((h, w), dtype=np.uint8)
        pts = polygon.reshape(-1, 1, 2).astype(np.int32)
        cv2.fillPoly(mask, [pts], 1)

        # Extract ring contours (with morphological closing for gap repair)
        outer_poly, inner_poly = _extract_ring_contours(mask)

        if outer_poly is None:
            self.update_status("Could not extract ring contours from selected polygon")
            return
        if inner_poly is None:
            self.update_status("No inner ring found — polygon may be solid, not a ring. "
                               "Try closing gaps first (edit mode)")
            return

        # Determine class IDs for outer and inner
        if class_id in self._RING_PAIRS:
            outer_cls, inner_cls = self._RING_PAIRS[class_id]
        else:
            outer_cls = class_id
            inner_cls = class_id + 1

        # Replace selected polygon with outer + inner
        self.push_undo()
        idx = canvas.selected_idx
        canvas.polygons.pop(idx)
        canvas.polygons.insert(idx, {"class_id": outer_cls, "polygon": outer_poly})
        canvas.polygons.insert(idx + 1, {"class_id": inner_cls, "polygon": inner_poly})
        canvas.selected_idx = None
        canvas.update()
        self._mark_modified()
        self.update_status(
            f"Split ring → outer (class {outer_cls}, {len(outer_poly)} pts) "
            f"+ inner (class {inner_cls}, {len(inner_poly)} pts)")

    # ── Save ──────────────────────────────────────────────────────────────

    def save_annotations(self):
        if not self.samples:
            return
        sample = self.samples[self.current_idx]

        # Get image dimensions from the canvas (already loaded)
        if self.edit_canvas.base_image is None:
            return
        h, w = self.edit_canvas.base_image.shape[:2]

        lines = []
        for poly_data in self.edit_canvas.polygons:
            class_id = poly_data["class_id"]
            polygon = poly_data["polygon"]
            coords = []
            for pt in polygon:
                coords.append(f"{pt[0] / w:.6f}")
                coords.append(f"{pt[1] / h:.6f}")
            lines.append(f"{class_id} " + " ".join(coords))

        # Determine save directory
        ann_dir = self._annotation_dir()
        stem = self._file_stem(sample)
        if ann_dir:
            ann_dir.mkdir(parents=True, exist_ok=True)
            save_path = ann_dir / f"{stem}.txt"
        else:
            save_path = sample.annotation_path

        with open(save_path, "w") as f:
            f.write("\n".join(lines))

        self.modified = False

        # Reset QC status since annotations changed
        if sample.uid in self._qc_status:
            del self._qc_status[sample.uid]
            self._save_qc_status()
            self._update_qc_label()

        self.update_status(f"Saved {len(lines)} polygons to {save_path.name}")

    # ── Sync view ─────────────────────────────────────────────────────────

    def sync_view(self, source: PolygonCanvas):
        zoom = source.zoom_level
        pan = source.pan_offset
        for canvas in self._all_canvases():
            if canvas is not source:
                canvas.set_view(zoom, pan)

    # ── Copy from reference ───────────────────────────────────────────────

    def copy_selected_ref_to_edit(self):
        if not self.ref_canvas.isVisible():
            return
        if self.ref_canvas.selected_idx is None:
            self.update_status("No reference polygon selected. Click one in the Reference panel first.")
            return
        if not (0 <= self.ref_canvas.selected_idx < len(self.ref_canvas.polygons)):
            return
        self.push_undo()
        poly_copy = copy.deepcopy(self.ref_canvas.polygons[self.ref_canvas.selected_idx])
        self.edit_canvas.polygons.append(poly_copy)
        self.edit_canvas.selected_idx = len(self.edit_canvas.polygons) - 1
        self.edit_canvas.update_display()
        self._mark_modified()
        self.save_annotations()
        self._schedule_auto_qc()
        cname = ANNOTATED_CLASSES.get(poly_copy['class_id'], 'Unknown')
        self.update_status(f"Copied reference polygon ({cname}) — saved.")

    def copy_exodermis_ref_to_edit(self):
        """Copy all exodermis polygons (classes 4, 5) from reference to edit."""
        if not self.ref_canvas.isVisible() or not self.ref_canvas.polygons:
            self.update_status("No reference polygons to copy")
            return
        exo_polys = [p for p in self.ref_canvas.polygons if p['class_id'] in (4, 5)]
        if not exo_polys:
            self.update_status("No exodermis polygons (classes 4/5) in reference")
            return
        self.push_undo()
        for p in exo_polys:
            self.edit_canvas.polygons.append(copy.deepcopy(p))
        self.edit_canvas.selected_idx = len(self.edit_canvas.polygons) - 1
        self.edit_canvas.update_display()
        self._mark_modified()
        self.save_annotations()
        self._schedule_auto_qc()
        self.update_status(f"Copied {len(exo_polys)} exodermis polygon(s) from reference — saved.")

    def copy_all_ref_to_edit(self):
        if not self.ref_canvas.isVisible() or not self.ref_canvas.polygons:
            self.update_status("No reference polygons to copy")
            return
        reply = QMessageBox.question(
            self, "Copy Reference",
            f"Replace editable polygons with {len(self.ref_canvas.polygons)} reference polygons?",
            QMessageBox.Yes | QMessageBox.No)
        if reply == QMessageBox.Yes:
            self.push_undo()
            self.edit_canvas.polygons = copy.deepcopy(self.ref_canvas.polygons)
            self.edit_canvas.selected_idx = None
            self.edit_canvas.update_display()
            self._mark_modified()
            self.save_annotations()
            self._schedule_auto_qc()
            self.update_status(f"Copied {len(self.ref_canvas.polygons)} reference polygons — saved.")

    # ── Misc ──────────────────────────────────────────────────────────────

    def home_view(self):
        """Reset zoom, pan, and panel sizes to equal."""
        for canvas in self._all_canvases():
            canvas.reset_view()
            canvas.update()
        visible_count = sum(1 for c in self._all_canvases() if c.isVisible())
        if visible_count > 0:
            self.splitter.setSizes([1] * visible_count)
        self.update_status("View reset to home")

    def update_status(self, msg: str):
        self.status_bar.showMessage(msg)

    def closeEvent(self, event):
        if self.modified:
            reply = QMessageBox.question(
                self, "Unsaved Changes",
                "You have unsaved changes. Save before closing?",
                QMessageBox.Save | QMessageBox.Discard | QMessageBox.Cancel)
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
Modes:
  Correct GT        Edit ground truth with predictions as reference (3 panels)
  Correct Pred      Edit predictions and save as annotations (2 panels)
  Create GT         Draw annotations from scratch (2 panels)

Data folder structure:
  data_dir/
    image/           Required for all modes
    annotation/      Required for Correct GT; created on save for others
    prediction/      Required for Correct GT and Correct Pred

Controls:
  A / Left Arrow    Previous sample
  D / Right Arrow   Next sample
  N                 Start drawing new polygon
  E                 Enter vertex editing
  Enter / Space     Confirm drawing or edits
  Escape            Cancel drawing or edits (reverts changes)
  Delete/Backspace  Delete selected vertex (edit mode) or polygon
  S                 Save annotations to file
  Tab               Copy selected reference polygon to editable panel
  C                 Copy ALL reference polygons to editable panel
  Ctrl+Z / Ctrl+Shift+Z  Undo / Redo
  1-4               Set class for new polygon
"""
    )
    parser.add_argument(
        "--data-dir", type=str, default=None,
        help="Data directory containing image/, annotation/, prediction/ subdirectories"
    )
    args = parser.parse_args()

    data_dir = Path(args.data_dir) if args.data_dir else None

    app = QApplication(sys.argv)
    app.setStyle("Fusion")

    palette = app.palette()
    palette.setColor(palette.Window, QColor(53, 53, 53))
    palette.setColor(palette.WindowText, Qt.white)
    palette.setColor(palette.Base, QColor(25, 25, 25))
    palette.setColor(palette.AlternateBase, QColor(53, 53, 53))
    palette.setColor(palette.ToolTipBase, QColor(53, 53, 53))
    palette.setColor(palette.ToolTipText, Qt.white)
    palette.setColor(palette.Text, Qt.white)
    palette.setColor(palette.Button, QColor(53, 53, 53))
    palette.setColor(palette.ButtonText, Qt.white)
    palette.setColor(palette.BrightText, Qt.red)
    palette.setColor(palette.Highlight, QColor(42, 130, 218))
    palette.setColor(palette.HighlightedText, Qt.black)
    app.setPalette(palette)

    editor = AnnotationEditor(data_dir=data_dir)
    editor.show()

    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
