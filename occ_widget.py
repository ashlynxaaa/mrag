import sys
import os
from PyQt5.QtWidgets import QWidget, QVBoxLayout

# --- OpenCASCADE ---
from OCC.Core.STEPControl import STEPControl_Reader, STEPControl_AsIs
from OCC.Core.IFSelect import IFSelect_RetDone
from OCC.Core.TopExp import TopExp_Explorer
from OCC.Core.TopAbs import TopAbs_FACE
from OCC.Core.AIS import AIS_Shape
from OCC.Core.Quantity import Quantity_Color, Quantity_TOC_RGB
from OCC.Display.backend import load_backend

# 必须在导入 qtViewer3d 前加载后端
load_backend("qt-pyqt5")
from OCC.Display.qtDisplay import qtViewer3d

class OCCViewerWidget(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.layout = QVBoxLayout(self)
        self.layout.setContentsMargins(0, 0, 0, 0)
        
        self.canvas = qtViewer3d(self)
        self.layout.addWidget(self.canvas)
        
        self.display = self.canvas._display
        self.indexed_faces = [] 
        self.current_ais_shapes = [] 

    def init_driver(self):
        self.canvas.InitDriver()
        self.display.display_triedron()
        # 设置漂亮的渐变背景
        self.display.set_bg_gradient_color([40, 40, 40], [100, 100, 100])

    def load_step(self, step_filename):
        if not os.path.exists(step_filename): return False

        self.display.EraseAll()
        self.indexed_faces = []
        self.current_ais_shapes = []

        step_reader = STEPControl_Reader()
        status = step_reader.ReadFile(step_filename)
        
        if status == IFSelect_RetDone:
            step_reader.TransferRoots()
            shape = step_reader.Shape()
            
            # 这里的遍历顺序必须和 step_parser 中的一致
            exp = TopExp_Explorer(shape, TopAbs_FACE)
            while exp.More():
                face = exp.Current()
                self.indexed_faces.append(face)
                exp.Next()

            self.ais_main = AIS_Shape(shape)
            self.ais_main.SetColor(Quantity_Color(0.8, 0.8, 0.8, Quantity_TOC_RGB))
            self.ais_main.SetTransparency(0.6) 
            self.display.Context.Display(self.ais_main, True)
            
            self.display.FitAll()
            return True
        else:
            return False

    def highlight_features(self, face_indices):
        """高亮显示指定面"""
        for ais in self.current_ais_shapes:
            self.display.Context.Remove(ais, True)
        self.current_ais_shapes = []

        for idx in face_indices:
            if 0 <= idx < len(self.indexed_faces):
                face_shape = self.indexed_faces[idx]
                ais_face = AIS_Shape(face_shape)
                ais_face.SetColor(Quantity_Color(1.0, 0.0, 0.0, Quantity_TOC_RGB)) # 红色
                ais_face.SetWidth(2.5)
                
                self.display.Context.Display(ais_face, True)
                self.current_ais_shapes.append(ais_face)
        
        self.display.Repaint()