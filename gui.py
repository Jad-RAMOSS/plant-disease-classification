import os
import sys
import warnings

warnings.filterwarnings('ignore')

from PyQt6.QtCore import Qt, QObject, QThread, pyqtSignal
from PyQt6.QtGui import QPixmap, QDragEnterEvent, QDropEvent
from PyQt6.QtWidgets import (
    QApplication, QFileDialog, QFrame, QHBoxLayout,
    QLabel, QMainWindow, QProgressBar, QPushButton,
    QSizePolicy, QVBoxLayout, QWidget,
)

# ── Palette ───────────────────────────────────────────────────────────────────
BG      = '#0d1117'
CARD    = '#161b22'
SURFACE = '#21262d'
BORDER  = '#30363d'
GREEN   = '#3fb950'
RED     = '#f85149'
YELLOW  = '#d29922'
TEXT    = '#c9d1d9'
MUTED   = '#8b949e'


# ── Background workers ────────────────────────────────────────────────────────

class ModelWorker(QObject):
    finished = pyqtSignal(object, object, object)
    error    = pyqtSignal(str)

    def run(self):
        try:
            from plant_disease.predict import find_model, load_artifacts
            model_path, csv_path = find_model()
            model, class_dict, img_size = load_artifacts(model_path, csv_path)
            self.finished.emit(model, class_dict, img_size)
        except Exception as exc:
            self.error.emit(str(exc))


class PredictWorker(QObject):
    finished = pyqtSignal(list)
    error    = pyqtSignal(str)

    def __init__(self, model, class_dict, img_size, path):
        super().__init__()
        self.model      = model
        self.class_dict = class_dict
        self.img_size   = img_size
        self.path       = path

    def run(self):
        try:
            from plant_disease.predict import predict_image
            results = predict_image(self.model, self.class_dict, self.img_size, self.path)
            self.finished.emit(results)
        except Exception as exc:
            self.error.emit(str(exc))


# ── Drop zone widget ──────────────────────────────────────────────────────────

class DropZone(QFrame):
    file_dropped = pyqtSignal(str)

    _IDLE_STYLE = f'''
        QFrame {{
            background: {SURFACE};
            border: 2px dashed {BORDER};
            border-radius: 10px;
        }}
    '''
    _HOVER_STYLE = f'''
        QFrame {{
            background: {SURFACE};
            border: 2px dashed {GREEN};
            border-radius: 10px;
        }}
    '''

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setAcceptDrops(True)
        self.setMinimumHeight(140)
        self.setStyleSheet(self._IDLE_STYLE)

        layout = QVBoxLayout(self)
        layout.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.setSpacing(6)

        icon = QLabel('🖼')
        icon.setAlignment(Qt.AlignmentFlag.AlignCenter)
        icon.setStyleSheet('font-size: 30px; border: none; background: transparent;')

        self.text = QLabel('Drag & drop a leaf image here')
        self.text.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.text.setStyleSheet(f'color: {MUTED}; font-size: 13px; border: none; background: transparent;')

        sep = QLabel('or')
        sep.setAlignment(Qt.AlignmentFlag.AlignCenter)
        sep.setStyleSheet(f'color: {MUTED}; font-size: 11px; border: none; background: transparent;')

        browse = QPushButton('Browse…')
        browse.setFixedSize(120, 32)
        browse.setCursor(Qt.CursorShape.PointingHandCursor)
        browse.setStyleSheet(f'''
            QPushButton {{
                background: {SURFACE};
                color: {TEXT};
                border: 1px solid {BORDER};
                border-radius: 6px;
                font-size: 12px;
            }}
            QPushButton:hover {{ background: {BORDER}; }}
        ''')
        browse.clicked.connect(self._browse)

        layout.addWidget(icon)
        layout.addWidget(self.text)
        layout.addWidget(sep)
        layout.addWidget(browse, alignment=Qt.AlignmentFlag.AlignCenter)

    def _browse(self):
        path, _ = QFileDialog.getOpenFileName(
            self, 'Open Image', '',
            'Images (*.jpg *.jpeg *.png *.bmp *.webp)',
        )
        if path:
            self.file_dropped.emit(path)

    def dragEnterEvent(self, event: QDragEnterEvent):
        if event.mimeData().hasUrls():
            event.acceptProposedAction()
            self.setStyleSheet(self._HOVER_STYLE)

    def dragLeaveEvent(self, event):
        self.setStyleSheet(self._IDLE_STYLE)

    def dropEvent(self, event: QDropEvent):
        self.setStyleSheet(self._IDLE_STYLE)
        urls = event.mimeData().urls()
        if urls:
            path = urls[0].toLocalFile()
            if path.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp', '.webp')):
                self.file_dropped.emit(path)


# ── Main window ───────────────────────────────────────────────────────────────

class PlantVisionApp(QMainWindow):

    def __init__(self):
        super().__init__()
        self.setWindowTitle('PlantVision — Plant Disease Classifier')
        self.setFixedSize(1020, 660)

        self.model      = None
        self.class_dict = None
        self.img_size   = None

        self.setStyleSheet(f'QMainWindow {{ background: {BG}; }}')
        self._build_ui()
        self._start_model_load()

    # ── UI construction ───────────────────────────────────────────────────────

    def _build_ui(self):
        root = QWidget()
        root.setStyleSheet(f'background: {BG};')
        self.setCentralWidget(root)

        layout = QVBoxLayout(root)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)
        layout.addWidget(self._make_header())
        layout.addWidget(self._make_body(), stretch=1)
        layout.addWidget(self._make_footer())

    def _make_header(self):
        hdr = QFrame()
        hdr.setFixedHeight(56)
        hdr.setStyleSheet(f'background: {CARD};')

        row = QHBoxLayout(hdr)
        row.setContentsMargins(24, 0, 24, 0)

        title = QLabel('🌿  PlantVision')
        title.setStyleSheet(f'color: {GREEN}; font-size: 20px; font-weight: 700;')

        self.status_lbl = QLabel('● Initialising…')
        self.status_lbl.setStyleSheet(f'color: {YELLOW}; font-size: 12px;')

        row.addWidget(title)
        row.addStretch()
        row.addWidget(self.status_lbl)
        return hdr

    def _make_body(self):
        body = QWidget()
        body.setStyleSheet(f'background: {BG};')

        row = QHBoxLayout(body)
        row.setContentsMargins(16, 12, 16, 12)
        row.setSpacing(12)
        row.addWidget(self._make_left(), stretch=3)
        row.addWidget(self._make_right())
        return body

    def _make_left(self):
        panel = QFrame()
        panel.setStyleSheet(f'QFrame {{ background: {CARD}; border-radius: 12px; }}')

        col = QVBoxLayout(panel)
        col.setContentsMargins(16, 16, 16, 16)
        col.setSpacing(10)

        self.drop_zone = DropZone()
        self.drop_zone.file_dropped.connect(self._process)
        col.addWidget(self.drop_zone)

        self.preview = QLabel()
        self.preview.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.preview.setStyleSheet('background: transparent;')
        self.preview.setSizePolicy(
            QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding,
        )
        col.addWidget(self.preview, stretch=1)
        return panel

    def _make_right(self):
        panel = QFrame()
        panel.setFixedWidth(330)
        panel.setStyleSheet(f'QFrame {{ background: {CARD}; border-radius: 12px; }}')

        col = QVBoxLayout(panel)
        col.setContentsMargins(20, 20, 20, 20)
        col.setSpacing(0)
        col.setAlignment(Qt.AlignmentFlag.AlignTop)

        def muted_cap(text):
            lbl = QLabel(text)
            lbl.setStyleSheet(f'color: {MUTED}; font-size: 10px; font-weight: 700; letter-spacing: 1px;')
            return lbl

        col.addWidget(muted_cap('RESULT'))
        col.addSpacing(8)

        self.lbl_plant = QLabel('—')
        self.lbl_plant.setStyleSheet(f'color: {TEXT}; font-size: 24px; font-weight: 700;')
        col.addWidget(self.lbl_plant)

        self.lbl_condition = QLabel('Drop an image to start')
        self.lbl_condition.setStyleSheet(f'color: {MUTED}; font-size: 13px;')
        col.addWidget(self.lbl_condition)
        col.addSpacing(20)

        col.addWidget(muted_cap('CONFIDENCE'))
        col.addSpacing(4)

        self.conf_bar = QProgressBar()
        self.conf_bar.setFixedHeight(6)
        self.conf_bar.setTextVisible(False)
        self.conf_bar.setValue(0)
        self._set_bar_color(GREEN)
        col.addWidget(self.conf_bar)
        col.addSpacing(4)

        self.conf_pct = QLabel('—')
        self.conf_pct.setStyleSheet(f'color: {GREEN}; font-size: 32px; font-weight: 700;')
        col.addWidget(self.conf_pct)
        col.addSpacing(16)

        div = QFrame()
        div.setFrameShape(QFrame.Shape.HLine)
        div.setFixedHeight(1)
        div.setStyleSheet(f'background: {BORDER};')
        col.addWidget(div)
        col.addSpacing(16)

        col.addWidget(muted_cap('TOP 5'))
        col.addSpacing(8)

        self.top5_rows = []
        for _ in range(5):
            row = QHBoxLayout()
            row.setSpacing(4)
            name = QLabel('')
            name.setStyleSheet(f'color: {MUTED}; font-size: 11px;')
            pct = QLabel('')
            pct.setAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
            pct.setStyleSheet(f'color: {MUTED}; font-size: 11px;')
            pct.setFixedWidth(52)
            row.addWidget(name, stretch=1)
            row.addWidget(pct)
            col.addLayout(row)
            col.addSpacing(3)
            self.top5_rows.append((name, pct))

        col.addStretch()
        return panel

    def _make_footer(self):
        ftr = QFrame()
        ftr.setFixedHeight(32)
        ftr.setStyleSheet(f'background: {CARD};')

        row = QHBoxLayout(ftr)
        row.setContentsMargins(16, 0, 16, 0)

        self.footer_model = QLabel('Model: —')
        self.footer_model.setStyleSheet(f'color: {MUTED}; font-size: 10px;')

        right = QLabel('PlantVillage · 38 classes')
        right.setStyleSheet(f'color: {MUTED}; font-size: 10px;')

        row.addWidget(self.footer_model)
        row.addStretch()
        row.addWidget(right)
        return ftr

    # ── Helpers ───────────────────────────────────────────────────────────────

    def _set_bar_color(self, color):
        self.conf_bar.setStyleSheet(f'''
            QProgressBar {{
                background: {SURFACE};
                border: none;
                border-radius: 3px;
            }}
            QProgressBar::chunk {{
                background: {color};
                border-radius: 3px;
            }}
        ''')

    def _set_status(self, text, color):
        self.status_lbl.setText(text)
        self.status_lbl.setStyleSheet(f'color: {color}; font-size: 12px;')

    # ── Model loading ─────────────────────────────────────────────────────────

    def _start_model_load(self):
        self._model_thread = QThread()
        self._model_worker = ModelWorker()
        self._model_worker.moveToThread(self._model_thread)
        self._model_thread.started.connect(self._model_worker.run)
        self._model_worker.finished.connect(self._on_model_loaded)
        self._model_worker.error.connect(self._on_model_error)
        self._model_worker.finished.connect(self._model_thread.quit)
        self._model_worker.error.connect(self._model_thread.quit)
        self._model_thread.start()

    def _on_model_loaded(self, model, class_dict, img_size):
        self.model      = model
        self.class_dict = class_dict
        self.img_size   = img_size
        from plant_disease.predict import find_model
        model_path, _ = find_model()
        self.footer_model.setText(f'Model: {os.path.basename(model_path)}')
        self._set_status('● Ready', GREEN)

    def _on_model_error(self, err):
        self._set_status(f'● {err}', RED)

    # ── Prediction pipeline ───────────────────────────────────────────────────

    def _process(self, path):
        self._show_preview(path)
        self._set_status('● Classifying…', YELLOW)

        self._pred_thread = QThread()
        self._pred_worker = PredictWorker(
            self.model, self.class_dict, self.img_size, path,
        )
        self._pred_worker.moveToThread(self._pred_thread)
        self._pred_thread.started.connect(self._pred_worker.run)
        self._pred_worker.finished.connect(self._render_results)
        self._pred_worker.error.connect(self._on_pred_error)
        self._pred_worker.finished.connect(self._pred_thread.quit)
        self._pred_worker.error.connect(self._pred_thread.quit)
        self._pred_thread.start()

    def _show_preview(self, path):
        pixmap = QPixmap(path).scaled(
            460, 320,
            Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.SmoothTransformation,
        )
        self.preview.setPixmap(pixmap)

    def _render_results(self, results):
        top  = results[0]
        conf = top['confidence']
        raw  = top['class']

        plant, condition = raw.split('___', 1) if '___' in raw else (raw, raw)
        plant     = plant.replace('_', ' ').strip()
        condition = condition.replace('_', ' ').strip()
        color     = GREEN if 'healthy' in condition.lower() else RED

        self.lbl_plant.setText(plant)
        self.lbl_condition.setText(condition)
        self.lbl_condition.setStyleSheet(f'color: {color}; font-size: 13px;')
        self.conf_bar.setValue(int(conf))
        self._set_bar_color(color)
        self.conf_pct.setText(f'{conf:.1f}%')
        self.conf_pct.setStyleSheet(f'color: {color}; font-size: 32px; font-weight: 700;')

        for i, (name_lbl, pct_lbl) in enumerate(self.top5_rows):
            if i < len(results):
                r       = results[i]
                display = r['class'].replace('___', ' — ').replace('_', ' ')
                is_top  = i == 0
                c       = (GREEN if 'healthy' in display.lower() else RED) if is_top else MUTED
                prefix  = '▸ ' if is_top else f'  {i + 1}. '
                name_lbl.setText(f'{prefix}{display}')
                name_lbl.setStyleSheet(f'color: {c}; font-size: 11px;')
                pct_lbl.setText(f"{r['confidence']:.1f}%")
                pct_lbl.setStyleSheet(f'color: {c}; font-size: 11px;')
            else:
                name_lbl.setText('')
                pct_lbl.setText('')

        self._set_status('● Ready', GREEN)

    def _on_pred_error(self, err):
        self._set_status(f'● Error: {err}', RED)


if __name__ == '__main__':
    app = QApplication(sys.argv)
    app.setStyle('Fusion')
    window = PlantVisionApp()
    window.show()
    sys.exit(app.exec())
