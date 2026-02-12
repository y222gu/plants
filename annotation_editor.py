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
import sys
from pathlib import Path
from typing import List, Optional, Tuple

import cv2
import numpy as np
from PyQt5.QtCore import Qt, QPoint, QRectF
from PyQt5.QtGui import (
    QImage, QPixmap, QPainter, QPen, QColor, QBrush,
    QPolygonF, QFont, QKeySequence
)
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QLabel, QPushButton, QComboBox, QMessageBox, QShortcut,
    QSplitter, QFrame, QScrollArea, QStatusBar, QGroupBox,
    QRadioButton, QButtonGroup, QSizePolicy
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


class PolygonCanvas(QLabel):
    """Canvas widget that displays an image with polygon overlays and handles interaction."""

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
        self.scale_factor = 1.0
        self.offset = (0, 0)

        self.setMinimumSize(300, 300)
        self.setAlignment(Qt.AlignCenter)
        self.setStyleSheet("border: 1px solid #555; background-color: #1a1a1a;")
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)

        if editable:
            self.setMouseTracking(True)

    def set_image(self, img: np.ndarray):
        """Set the base image (H, W, 3) uint8 RGB."""
        self.base_image = img.copy()
        self.update_display()

    def set_polygons(self, polygons: List[dict]):
        """Set the list of polygons to display."""
        self.polygons = polygons
        self.selected_idx = None
        self.update_display()

    def update_display(self):
        """Redraw the canvas with image and polygon overlays."""
        if self.base_image is None:
            return

        h, w = self.base_image.shape[:2]

        # Create QImage from numpy array
        qimg = QImage(self.base_image.data, w, h, 3 * w, QImage.Format_RGB888)
        pixmap = QPixmap.fromImage(qimg)

        # Scale to fit widget while maintaining aspect ratio
        widget_size = self.size()
        scaled = pixmap.scaled(
            widget_size.width() - 4, widget_size.height() - 4,
            Qt.KeepAspectRatio, Qt.SmoothTransformation
        )

        # Calculate scale factor and offset for coordinate mapping
        self.scale_factor = scaled.width() / w
        self.offset = (
            (widget_size.width() - scaled.width()) // 2,
            (widget_size.height() - scaled.height()) // 2
        )

        # Draw polygons on the pixmap
        painter = QPainter(scaled)
        painter.setRenderHint(QPainter.Antialiasing)

        for idx, poly_data in enumerate(self.polygons):
            class_id = poly_data["class_id"]
            polygon = poly_data["polygon"]

            if idx == self.selected_idx:
                color = SELECTED_COLOR
                pen_width = 3
            else:
                color = CLASS_QCOLORS.get(class_id, QColor(128, 128, 128, 180))
                pen_width = 2

            # Scale polygon coordinates
            scaled_pts = polygon * self.scale_factor

            # Create QPolygonF
            qpoly = QPolygonF()
            for pt in scaled_pts:
                qpoly.append(QPoint(int(pt[0]), int(pt[1])))

            # Draw filled polygon
            painter.setPen(QPen(color, pen_width))
            fill_color = QColor(color)
            fill_color.setAlpha(60)
            painter.setBrush(QBrush(fill_color))
            painter.drawPolygon(qpoly)

        # Draw points being drawn (new polygon)
        if self.drawing_mode and self.drawing_points:
            pen = QPen(QColor(255, 0, 255), 2)  # Magenta for drawing
            painter.setPen(pen)
            painter.setBrush(QBrush(QColor(255, 0, 255, 80)))

            for pt in self.drawing_points:
                scaled_pt = (int(pt[0] * self.scale_factor), int(pt[1] * self.scale_factor))
                painter.drawEllipse(QPoint(*scaled_pt), 4, 4)

            # Draw lines between points
            if len(self.drawing_points) > 1:
                for i in range(len(self.drawing_points) - 1):
                    p1 = self.drawing_points[i]
                    p2 = self.drawing_points[i + 1]
                    sp1 = (int(p1[0] * self.scale_factor), int(p1[1] * self.scale_factor))
                    sp2 = (int(p2[0] * self.scale_factor), int(p2[1] * self.scale_factor))
                    painter.drawLine(QPoint(*sp1), QPoint(*sp2))

        # Draw title
        painter.setPen(QPen(Qt.white))
        painter.setFont(QFont("Arial", 12, QFont.Bold))
        painter.drawText(10, 25, self.title)

        painter.end()
        self.setPixmap(scaled)

    def resizeEvent(self, event):
        super().resizeEvent(event)
        self.update_display()

    def widget_to_image_coords(self, widget_pos: QPoint) -> Tuple[int, int]:
        """Convert widget coordinates to image coordinates."""
        x = (widget_pos.x() - self.offset[0]) / self.scale_factor
        y = (widget_pos.y() - self.offset[1]) / self.scale_factor
        return int(x), int(y)

    def mousePressEvent(self, event):
        if not self.editable or self.base_image is None:
            return

        img_x, img_y = self.widget_to_image_coords(event.pos())
        h, w = self.base_image.shape[:2]

        # Check bounds
        if not (0 <= img_x < w and 0 <= img_y < h):
            return

        if self.drawing_mode:
            # Add point to current polygon being drawn
            self.drawing_points.append((img_x, img_y))
            self.update_display()
        else:
            # Try to select a polygon
            self.selected_idx = self.find_polygon_at(img_x, img_y)
            self.update_display()

            if self.selected_idx is not None:
                parent = self.get_editor_parent()
                if parent:
                    parent.update_status(f"Selected polygon {self.selected_idx} (class {self.polygons[self.selected_idx]['class_id']})")

    def find_polygon_at(self, x: int, y: int) -> Optional[int]:
        """Find polygon containing the point (x, y). Returns index or None."""
        h, w = self.base_image.shape[:2]

        # Check in reverse order (top polygons first)
        for idx in range(len(self.polygons) - 1, -1, -1):
            polygon = self.polygons[idx]["polygon"]
            mask = polygon_to_mask(polygon, h, w)
            if mask[y, x] > 0:
                return idx
        return None

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
        self.update_display()

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
        self.update_display()

    def delete_selected(self) -> Optional[dict]:
        """Delete the selected polygon and return it."""
        if self.selected_idx is not None and 0 <= self.selected_idx < len(self.polygons):
            removed = self.polygons.pop(self.selected_idx)
            self.selected_idx = None
            self.update_display()
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
        self.draw_btn.clicked.connect(self.start_drawing)
        action_layout.addWidget(self.draw_btn)

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

        splitter.setSizes([400, 500, 400])
        main_layout.addWidget(splitter, stretch=1)

        # Status bar
        self.status_bar = QStatusBar()
        self.setStatusBar(self.status_bar)
        self.update_status("Ready. Use A/D to navigate, N to draw, Del to delete.")

        # Legend
        legend_layout = QHBoxLayout()
        legend_layout.addWidget(QLabel("Legend:"))
        for cid, cname in ANNOTATED_CLASSES.items():
            color = CLASS_COLORS_RGB.get(cid, (128, 128, 128))
            lbl = QLabel(f"  {cname}")
            lbl.setStyleSheet(f"color: rgb({color[0]},{color[1]},{color[2]}); font-weight: bold;")
            legend_layout.addWidget(lbl)
        legend_layout.addStretch()
        main_layout.addLayout(legend_layout)

    def setup_shortcuts(self):
        # Navigation
        QShortcut(QKeySequence(Qt.Key_A), self, self.prev_sample)
        QShortcut(QKeySequence(Qt.Key_Left), self, self.prev_sample)
        QShortcut(QKeySequence(Qt.Key_D), self, self.next_sample)
        QShortcut(QKeySequence(Qt.Key_Right), self, self.next_sample)

        # Actions
        QShortcut(QKeySequence(Qt.Key_N), self, self.start_drawing)
        QShortcut(QKeySequence(Qt.Key_Delete), self, self.delete_selected)
        QShortcut(QKeySequence(Qt.Key_Backspace), self, self.delete_selected)
        QShortcut(QKeySequence(Qt.Key_S), self, self.save_annotations)
        QShortcut(QKeySequence(Qt.Key_Escape), self, self.cancel_drawing)
        QShortcut(QKeySequence(Qt.Key_Return), self, self.finish_drawing)
        QShortcut(QKeySequence(Qt.Key_Enter), self, self.finish_drawing)

        # Class shortcuts (1-4)
        for i in range(4):
            QShortcut(QKeySequence(Qt.Key_1 + i), self, lambda idx=i: self.set_drawing_class(idx))

        # Copy predictions to GT
        QShortcut(QKeySequence(Qt.Key_C), self, self.copy_predictions_to_gt)

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
        sample = self.samples[idx]

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

    def start_drawing(self):
        """Start drawing a new polygon on GT canvas."""
        class_id = self.class_buttons.checkedId()
        self.gt_canvas.start_drawing(class_id)
        self.update_status(f"Drawing mode: Click to add points for class {class_id}. Press Enter to finish, Escape to cancel.")

    def finish_drawing(self):
        """Finish drawing the current polygon."""
        if self.gt_canvas.drawing_mode:
            result = self.gt_canvas.finish_drawing()
            if result:
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
            self.update_status("Drawing cancelled")

    def delete_selected(self):
        """Delete the currently selected polygon."""
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
            # Deep copy predictions to GT
            import copy
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
