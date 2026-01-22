import sys
import os
import time
from collections import Counter
from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, 
                             QHBoxLayout, QTreeWidget, QTreeWidgetItem, QPushButton, 
                             QTextEdit, QFileDialog, QSplitter, QLabel, QMessageBox,
                             QProgressBar, QGroupBox)
from PyQt5.QtCore import Qt, QThread, pyqtSignal
from PyQt5.QtGui import QFont

# 导入后端
try:
    from inference_backend import IntelliMFRBackend
except ImportError:
    print("错误: 找不到 inference_backend.py")
    sys.exit(1)

# 导入 3D 组件
try:
    from occ_widget import OCCViewerWidget
except ImportError:
    print("警告: 找不到 occ_widget.py，3D 功能将不可用。")
    OCCViewerWidget = None

# 【请修改】您的模型路径
MODEL_PATH = "D:/CODE/mraag/predict/best_model.pth"

# --- 异步工作线程 ---
class WorkerThread(QThread):
    finished = pyqtSignal(list)
    error = pyqtSignal(str)
    def __init__(self, backend, file_path):
        super().__init__()
        self.backend = backend
        self.file_path = file_path
    def run(self):
        try:
            results = self.backend.process_file(self.file_path)
            self.finished.emit(results)
        except Exception as e:
            self.error.emit(str(e))

# --- 主窗口 ---
class IntelliMFRWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.backend = None
        self.current_step_file = None
        self.init_ui()
        self.init_backend()

    def init_ui(self):
        self.setWindowTitle("IntelliMFR - 智能加工特征识别系统 (Prototype)")
        self.resize(1200, 900)
        
        # --- 现代工业风样式 ---
        self.setStyleSheet("""
            QMainWindow { background-color: #2b2b2b; color: #ffffff; }
            QGroupBox { border: 1px solid #555; margin-top: 10px; font-weight: bold; color: #aaa; }
            QGroupBox::title { subcontrol-origin: margin; subcontrol-position: top left; padding: 0 3px; }
            QTreeWidget { background-color: #3c3f41; color: #dcdcdc; border: none; }
            QTreeWidget::item:selected { background-color: #4b6eaf; }
            QTextEdit { background-color: #1e1e1e; color: #00ff00; font-family: Consolas; border: none; }
            QPushButton { background-color: #007acc; color: white; border: none; padding: 8px; border-radius: 4px; font-weight: bold; }
            QPushButton:hover { background-color: #0098ff; }
            QPushButton:disabled { background-color: #444; color: #888; }
            QLabel { color: #aaaaaa; }
            QProgressBar { border: none; background-color: #444; height: 4px; text-align: center; }
            QProgressBar::chunk { background-color: #007acc; }
        """)

        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QVBoxLayout(central_widget)
        main_layout.setContentsMargins(10, 10, 10, 10)
        main_layout.setSpacing(10)

        # === 顶部: 3D 视图 ===
        view_group = QGroupBox("3D Visualization")
        view_layout = QVBoxLayout(view_group)
        view_layout.setContentsMargins(0, 15, 0, 0)

        if OCCViewerWidget:
            self.view_3d = OCCViewerWidget()
            view_layout.addWidget(self.view_3d)
        else:
            self.view_3d = QLabel("3D Module Missing")
            self.view_3d.setAlignment(Qt.AlignCenter)
            self.view_3d.setStyleSheet("background-color:#111; color:red; font-size:16px;")
            view_layout.addWidget(self.view_3d)

        # === 底部: 操作区 (左右分割) ===
        bottom_splitter = QSplitter(Qt.Horizontal)
        bottom_splitter.setHandleWidth(1)

        # 左下: 特征树
        tree_group = QGroupBox("Recognized Features")
        tree_layout = QVBoxLayout(tree_group)
        self.tree = QTreeWidget()
        self.tree.setHeaderLabels(["Type", "ID", "Faces"])
        self.tree.setColumnWidth(0, 180)
        self.tree.setColumnWidth(1, 60)
        self.tree.itemClicked.connect(self.on_tree_item_clicked)
        tree_layout.addWidget(self.tree)
        bottom_splitter.addWidget(tree_group)

        # 右下: 控制与日志
        control_group = QGroupBox("Controls & Logs")
        control_layout = QVBoxLayout(control_group)

        # 按钮
        btn_layout = QHBoxLayout()
        self.btn_open = QPushButton("📂 Open STEP")
        self.btn_open.clicked.connect(self.open_file)
        self.btn_run = QPushButton("⚡ Start Recognition")
        self.btn_run.clicked.connect(self.run_recognition)
        self.btn_run.setEnabled(False)
        btn_layout.addWidget(self.btn_open)
        btn_layout.addWidget(self.btn_run)
        control_layout.addLayout(btn_layout)

        # 状态与日志
        self.lbl_status = QLabel("System Ready")
        control_layout.addWidget(self.lbl_status)
        self.log_window = QTextEdit()
        self.log_window.setReadOnly(True)
        control_layout.addWidget(self.log_window)

        bottom_splitter.addWidget(control_group)
        bottom_splitter.setStretchFactor(0, 4) # 左 40%
        bottom_splitter.setStretchFactor(1, 6) # 右 60%

        # === 主分割器 ===
        main_splitter = QSplitter(Qt.Vertical)
        main_splitter.addWidget(view_group)
        main_splitter.addWidget(bottom_splitter)
        main_splitter.setStretchFactor(0, 7) # 上 70%
        main_splitter.setStretchFactor(1, 3) # 下 30%

        main_layout.addWidget(main_splitter)
        
        # 进度条
        self.progress = QProgressBar()
        self.progress.setVisible(False)
        main_layout.addWidget(self.progress)

    def showEvent(self, event):
        super().showEvent(event)
        if hasattr(self, 'view_3d') and isinstance(self.view_3d, OCCViewerWidget):
            self.view_3d.init_driver()

    def log(self, message):
        timestamp = time.strftime("[%H:%M:%S] ", time.localtime())
        self.log_window.append(f"{timestamp}{message}")
        self.log_window.verticalScrollBar().setValue(self.log_window.verticalScrollBar().maximum())

    def init_backend(self):
        self.log("Initializing AI Engine...")
        QApplication.processEvents()
        if not os.path.exists(MODEL_PATH):
            QMessageBox.critical(self, "Error", f"Model missing:\n{MODEL_PATH}")
            return
        try:
            self.backend = IntelliMFRBackend(MODEL_PATH, device='cpu')
            self.log("✅ AI Engine Ready (CPU Mode).")
            self.lbl_status.setText(f"Model: {os.path.basename(MODEL_PATH)}")
        except Exception as e:
            self.log(f"❌ Init Failed: {str(e)}")

    def open_file(self):
        fname, _ = QFileDialog.getOpenFileName(self, 'Open STEP', '.', "STEP (*.step *.stp)")
        if fname:
            self.current_step_file = fname
            self.log(f"📂 Loaded: {os.path.basename(fname)}")
            self.lbl_status.setText(f"File: {os.path.basename(fname)}")
            self.btn_run.setEnabled(True)
            self.tree.clear()
            
            if hasattr(self, 'view_3d') and isinstance(self.view_3d, OCCViewerWidget):
                self.log("Rendering 3D model...")
                if self.view_3d.load_step(fname):
                    self.log("✅ 3D Render Complete")
                else:
                    self.log("❌ 3D Render Failed")

    def run_recognition(self):
        if not self.backend: return
        self.btn_run.setEnabled(False)
        self.btn_open.setEnabled(False)
        self.progress.setVisible(True)
        self.progress.setRange(0, 0)
        self.log("🚀 Analyzing...")
        
        self.worker = WorkerThread(self.backend, self.current_step_file)
        self.worker.finished.connect(self.on_recognition_finished)
        self.worker.error.connect(self.on_recognition_error)
        self.worker.start()

    def on_recognition_finished(self, results):
        self.progress.setRange(0, 100)
        self.progress.setValue(100)
        self.progress.setVisible(False)
        self.btn_run.setEnabled(True)
        self.btn_open.setEnabled(True)
        
        # --- 统计报告 ---
        type_list = [feat['type'] for feat in results]
        counts = Counter(type_list)
        self.log("\n" + "="*40)
        self.log(f"📊 Recognition Report: {len(results)} Features Found")
        self.log("-" * 40)
        for ftype, count in counts.items():
            self.log(f"   • {ftype:<20}: {count}")
        self.log("="*40 + "\n")
        # ---------------

        self.populate_tree(results)

    def on_recognition_error(self, err_msg):
        self.progress.setVisible(False)
        self.btn_run.setEnabled(True)
        self.btn_open.setEnabled(True)
        self.log(f"❌ Error: {err_msg}")
        QMessageBox.warning(self, "Error", err_msg)

    def populate_tree(self, features):
        self.tree.clear()
        grouped = {}
        for feat in features:
            ftype = feat['type']
            if ftype not in grouped: grouped[ftype] = []
            grouped[ftype].append(feat)

        for ftype, feat_list in grouped.items():
            parent = QTreeWidgetItem(self.tree)
            parent.setText(0, f"{ftype} ({len(feat_list)})")
            parent.setExpanded(True)
            for feat in feat_list:
                child = QTreeWidgetItem(parent)
                child.setText(0, f"Instance {feat['id']}")
                child.setText(1, str(feat['id']))
                faces_str = str(feat['face_indices'])
                if len(faces_str) > 20: faces_str = faces_str[:20] + "..."
                child.setText(2, faces_str)
                child.setData(0, Qt.UserRole, feat)

    def on_tree_item_clicked(self, item, column):
        feat_data = item.data(0, Qt.UserRole)
        if feat_data:
            self.log(f"👉 Selected: {feat_data['type']} #{feat_data['id']} | Faces: {feat_data['face_indices']}")
            if hasattr(self, 'view_3d') and isinstance(self.view_3d, OCCViewerWidget):
                self.view_3d.highlight_features(feat_data['face_indices'])

if __name__ == "__main__":
    app = QApplication(sys.argv)
    font = QFont("Segoe UI", 9)
    app.setFont(font)
    window = IntelliMFRWindow()
    window.show()
    sys.exit(app.exec_())