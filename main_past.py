
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Artery GUI — Qt-compatible skeleton using pyqtgraph.Qt (PyQt5/6 or PySide2/6)
UI texts in English (as requested). Code comments in Chinese。

依赖（与您的栈一致）：
- pyqtgraph (2D三视图 / Qt 适配)
- vispy (3D 体/线框展示，占位)
- qt_material (主题，可选)
- 其余 SimpleITK、skimage、scipy 等后续接入时使用

运行：
python ArteryGUI_QtCompat_Skeleton.py

说明：
1) 保持 from pyqtgraph.Qt import QtCore, QtWidgets, QtGui 的导入风格；
2) 通过 Signal/Slot 兼容层适配 PyQt 与 PySide；
3) 右侧：上方两行路径；左列三视图(2D, pyqtgraph)；右列 3D (vispy)；
4) 左侧：文件/显示/编辑/Prompt/推理/中心线/保存列表；
5) 所有 TODO 均为后续接入业务逻辑位置。
"""

import sys
import os
from pathlib import Path

import pyqtgraph as pg
from pyqtgraph.Qt import QtCore, QtWidgets, QtGui
from qt_material import apply_stylesheet
from vispy import scene
import SimpleITK as sitk
import numpy as np

# 可选：其余库待接入
# import SimpleITK as sitk
# import numpy as np
# from skimage.morphology import skeletonize_3d
# ...

# -----------------------------
# Qt 信号/槽 兼容层（关键）
# -----------------------------
try:
    Signal = QtCore.pyqtSignal      # PyQt5/6
    Slot   = QtCore.pyqtSlot
except AttributeError:
    Signal = QtCore.Signal          # PySide2/6
    Slot   = QtCore.Slot

QSettings = getattr(QtCore, "QSettings", None)


# =============================
# 左侧操作面板
# =============================
class LeftPanel(QtWidgets.QWidget):
    # 对外信号（英文命名；内部注释中文）
    sigLoadImage         = Signal()
    sigLoadMask          = Signal()
    sigRunInference      = Signal()
    sigExtractCenterline = Signal()
    sigSelectBox         = Signal(bool)
    sigClearBox          = Signal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self._build_ui()

    def _build_ui(self):
        # 根布局（竖直）
        root = QtWidgets.QVBoxLayout(self)
        root.setContentsMargins(8, 8, 8, 8)
        root.setSpacing(10)
        self.setMinimumWidth(320)

        # --- File ---
        grp_file = QtWidgets.QGroupBox("File")  # 文件加载
        v = QtWidgets.QVBoxLayout(grp_file)
        self.btnLoadImage = QtWidgets.QPushButton("Load Image…")
        self.btnLoadMask  = QtWidgets.QPushButton("Load Mask…")
        v.addWidget(self.btnLoadImage)
        v.addWidget(self.btnLoadMask)
        root.addWidget(grp_file)

        # --- Display ---
        grp_disp = QtWidgets.QGroupBox("Display")  # 显示选项
        g = QtWidgets.QGridLayout(grp_disp)
        self.ckShowDebone2D   = QtWidgets.QCheckBox("Show Deboned 2D")
        self.ckShowDebone3D   = QtWidgets.QCheckBox("Show Deboned 3D")
        self.ckShowPred2D     = QtWidgets.QCheckBox("Show Prediction 2D")
        self.ckShowPred3D     = QtWidgets.QCheckBox("Show Prediction 3D")
        self.ckShowCenterline = QtWidgets.QCheckBox("Show Centerline")
        self.sldImgAlpha  = QtWidgets.QSlider(QtCore.Qt.Horizontal)  # 图像不透明度
        self.sldMaskAlpha = QtWidgets.QSlider(QtCore.Qt.Horizontal)  # 掩膜不透明度
        for s in (self.sldImgAlpha, self.sldMaskAlpha):
            s.setRange(0, 100); s.setValue(100)
        g.addWidget(self.ckShowDebone2D,   0, 0)
        g.addWidget(self.ckShowDebone3D,   0, 1)
        g.addWidget(self.ckShowPred2D,     1, 0)
        g.addWidget(self.ckShowPred3D,     1, 1)
        g.addWidget(self.ckShowCenterline, 2, 0)
        

          # 窗宽窗位（CTA 初始：-50, 550）
        self.dsbWLLow  = QtWidgets.QDoubleSpinBox(); self.dsbWLLow.setRange(-2048, 4096); self.dsbWLLow.setDecimals(1)
        self.dsbWLHigh = QtWidgets.QDoubleSpinBox(); self.dsbWLHigh.setRange(-2048, 4096); self.dsbWLHigh.setDecimals(1)
        self.dsbWLLow.setValue(-50.0); self.dsbWLHigh.setValue(550.0)
        g.addWidget(QtWidgets.QLabel("W/L Low"),  3, 0)
        g.addWidget(self.dsbWLLow,               3, 1)
        g.addWidget(QtWidgets.QLabel("W/L High"), 4, 0)
        g.addWidget(self.dsbWLHigh,              4, 1)
        g.addWidget(QtWidgets.QLabel("Image Opacity"), 5, 0)
        g.addWidget(self.sldImgAlpha,                 5, 1)
        g.addWidget(QtWidgets.QLabel("Mask Opacity"), 6, 0)
        g.addWidget(self.sldMaskAlpha,                6, 1)
        root.addWidget(grp_disp)
        # === 默认开关：让新手打开就能看到效果 ===
        # 2D Mask 叠加默认打开；3D 底图默认打开
        self.ckShowPred2D.setChecked(True)
        self.ckShowDebone3D.setChecked(True)
        # 2D 底图本就总显示（当前未设开关），如需开关可另加 ckShowImage2D

        # --- Editing (2D) ---
        grp_edit = QtWidgets.QGroupBox("Editing (2D)")  # 2D编辑工具
        f = QtWidgets.QFormLayout(grp_edit)
        self.modeCombo  = QtWidgets.QComboBox(); self.modeCombo.addItems(["Cursor", "Brush", "Polygon", "Eraser"])  # 模式
        self.spinBrush  = QtWidgets.QSpinBox(); self.spinBrush.setRange(1, 200); self.spinBrush.setValue(5)         # 画笔半径
        self.comboLabel = QtWidgets.QComboBox(); self.comboLabel.addItems(["0: Erase", "1: Mark"])                # 活动标签
        hb = QtWidgets.QHBoxLayout(); self.btnUndo = QtWidgets.QPushButton("Undo"); self.btnRedo = QtWidgets.QPushButton("Redo")
        hb.addWidget(self.btnUndo); hb.addWidget(self.btnRedo)
        f.addRow("Mode", self.modeCombo)
        f.addRow("Brush Radius", self.spinBrush)
        f.addRow("Active Label", self.comboLabel)
        f.addRow(hb)
        root.addWidget(grp_edit)

        # --- Prompt ---
        grp_prompt = QtWidgets.QGroupBox("Prompt")  # 框选提示
        h = QtWidgets.QHBoxLayout(grp_prompt)
        self.tglBox     = QtWidgets.QPushButton("Box Select"); self.tglBox.setCheckable(True)  # 进入/退出框选
        self.btnClrBox  = QtWidgets.QPushButton("Clear Box")
        h.addWidget(self.tglBox); h.addWidget(self.btnClrBox)
        root.addWidget(grp_prompt)

        # --- Inference ---
        grp_inf = QtWidgets.QGroupBox("Inference")  # 远程推理
        ff = QtWidgets.QFormLayout(grp_inf)
        self.leEndpoint = QtWidgets.QLineEdit(); self.leEndpoint.setPlaceholderText("http://server:port/predict")  # 接口地址
        self.btnRunInf  = QtWidgets.QPushButton("Run Inference")  # 运行按钮
        ff.addRow("Endpoint", self.leEndpoint)
        ff.addRow(self.btnRunInf)
        root.addWidget(grp_inf)

        # --- Centerline ---
        grp_cl = QtWidgets.QGroupBox("Centerline")  # 中心线
        cf = QtWidgets.QFormLayout(grp_cl)
        self.dsbThresh = QtWidgets.QDoubleSpinBox(); self.dsbThresh.setRange(0.0, 1.0); self.dsbThresh.setSingleStep(0.05); self.dsbThresh.setValue(0.5)
        self.ckSmooth  = QtWidgets.QCheckBox("Smoothing"); self.ckSmooth.setChecked(True)
        self.btnExtractCL = QtWidgets.QPushButton("Extract Centerline")
        cf.addRow("Threshold", self.dsbThresh)
        cf.addRow(self.ckSmooth)
        cf.addRow(self.btnExtractCL)
        root.addWidget(grp_cl)

        # --- Saved Results ---
        grp_saved = QtWidgets.QGroupBox("Saved Results")  # 保存结果列表
        sv = QtWidgets.QVBoxLayout(grp_saved)
        self.listSaved = QtWidgets.QListWidget()
        sv.addWidget(self.listSaved)
        root.addWidget(grp_saved, 1)

        # 信号转发（将按钮与对外信号绑定）
        self.btnLoadImage.clicked.connect(self.sigLoadImage)
        self.btnLoadMask.clicked.connect(self.sigLoadMask)
        self.btnRunInf.clicked.connect(self.sigRunInference)
        self.btnExtractCL.clicked.connect(self.sigExtractCenterline)
        self.tglBox.toggled.connect(self.sigSelectBox)
        self.btnClrBox.clicked.connect(self.sigClearBox)

        root.addStretch(1)


# =============================
# 三视图（2D, pyqtgraph）
# =============================
class ThreeView2D(QtWidgets.QWidget):
    sigSliceChanged = Signal(str, int)  # （轴向标识, 索引）

    def __init__(self, parent=None):
        super().__init__(parent)
        self._build_ui()

    def _build_ui(self):
        grid = QtWidgets.QGridLayout(self)
        grid.setContentsMargins(0, 0, 0, 0)
        grid.setHorizontalSpacing(8)
        grid.setVerticalSpacing(8)

        # 生成三个视图 + 各自滑动条（竖直）
        self.viewAx, self.sliderAx = self._make_view("Axial")    # 轴状
        self.viewCor, self.sliderCor = self._make_view("Coronal") # 冠状
        self.viewSag, self.sliderSag = self._make_view("Sagittal")# 矢状

        grid.addWidget(self.viewAx, 0, 0)
        grid.addWidget(self.viewCor, 1, 0)
        grid.addWidget(self.viewSag, 2, 0)

        grid.addWidget(self.sliderAx, 0, 1)
        grid.addWidget(self.sliderCor, 1, 1)
        grid.addWidget(self.sliderSag, 2, 1)

        # 连接滑条
        self.sliderAx.valueChanged.connect(lambda v: self.sigSliceChanged.emit('ax', v))
        self.sliderCor.valueChanged.connect(lambda v: self.sigSliceChanged.emit('cor', v))
        self.sliderSag.valueChanged.connect(lambda v: self.sigSliceChanged.emit('sag', v))
        self.lblAx  = QtWidgets.QLabel("0 / 0")
        self.lblCor = QtWidgets.QLabel("0 / 0")
        self.lblSag = QtWidgets.QLabel("0 / 0")
        for lbl in (self.lblAx, self.lblCor, self.lblSag):
            lbl.setAlignment(QtCore.Qt.AlignCenter)
            lbl.setMinimumWidth(70)
        grid.addWidget(self.lblAx,  0, 2)
        grid.addWidget(self.lblCor, 1, 2)
        grid.addWidget(self.lblSag, 2, 2)

        


    def _make_view(self, title: str):
        # 单个视图容器（包含标题 + 图像视图）
        wrapper = QtWidgets.QWidget()
        v = QtWidgets.QVBoxLayout(wrapper)
        v.setContentsMargins(0, 0, 0, 0)

        lbl = QtWidgets.QLabel(title)
        lbl.setAlignment(QtCore.Qt.AlignLeft | QtCore.Qt.AlignVCenter)

        pgw = pg.GraphicsLayoutWidget()
        pgw.setSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Expanding)

        vb = pg.ViewBox(lockAspect=True)
        pgw.addItem(vb)
        imgItem = pg.ImageItem()
        vb.addItem(imgItem)

        v.addWidget(lbl)
        v.addWidget(pgw, 1)

        # 调整为水平 Slider，便于 2×2 网格布局更紧凑
        slider = QtWidgets.QSlider(QtCore.Qt.Horizontal)
        slider.setRange(0, 0)  # TODO: 加载体数据后设置最大值
        slider.setSingleStep(1)
        slider.setPageStep(5)

        # 附加引用，方便外部访问
        wrapper._vb = vb
        wrapper._imgItem = imgItem
        return wrapper, slider

    # 便捷属性（获取三个 ImageItem）
    @property
    def img_ax(self) -> pg.ImageItem:
        return self.viewAx._imgItem

    @property
    def img_cor(self) -> pg.ImageItem:
        return self.viewCor._imgItem

    @property
    def img_sag(self) -> pg.ImageItem:
        return self.viewSag._imgItem
    
    def ensure_mask_layers(self):
        """确保三视图都有一个 Mask 叠加层（ImageItem），延迟创建"""
        for vw in (self.viewAx, self.viewCor, self.viewSag):
            if not hasattr(vw, "_maskItem") or vw._maskItem is None:
                maskItem = pg.ImageItem()
                maskItem.setZValue(10)
                vw._vb.addItem(maskItem)
                vw._maskItem = maskItem
    @property
    def mask_ax(self) -> pg.ImageItem:
        self.ensure_mask_layers(); return self.viewAx._maskItem
    @property
    def mask_cor(self) -> pg.ImageItem:
        self.ensure_mask_layers(); return self.viewCor._maskItem
    @property
    def mask_sag(self) -> pg.ImageItem:
        self.ensure_mask_layers(); return self.viewSag._maskItem

    # === 新增：把内部三视图“拆”出来，重排为 2×2（右下留给 3D） ===
    def rearrange_to_2x2(self, container: QtWidgets.QGridLayout, view3d_widget: QtWidgets.QWidget):
        # 清空自身布局（不销毁子部件）
        old_layout = self.layout()
        if old_layout is not None:
            while old_layout.count():
                it = old_layout.takeAt(0)
                w = it.widget()
                if w is not None:
                    w.setParent(None)
        # Axial 行：视图 + 水平 slider + 数字
        self.lblAx  = QtWidgets.QLabel("0 / 0"); self.lblAx.setAlignment(QtCore.Qt.AlignRight|QtCore.Qt.AlignVCenter)
        h_ax = QtWidgets.QHBoxLayout(); w_ax = QtWidgets.QWidget()
        h_ax.addWidget(self.viewAx, 1); h_ax.addWidget(self.sliderAx); h_ax.addWidget(self.lblAx)
        w_ax.setLayout(h_ax)
        # Coronal 行
        self.lblCor = QtWidgets.QLabel("0 / 0"); self.lblCor.setAlignment(QtCore.Qt.AlignRight|QtCore.Qt.AlignVCenter)
        h_cor = QtWidgets.QHBoxLayout(); w_cor = QtWidgets.QWidget()
        h_cor.addWidget(self.viewCor, 1); h_cor.addWidget(self.sliderCor); h_cor.addWidget(self.lblCor)
        w_cor.setLayout(h_cor)
        # Sagittal 行
        self.lblSag = QtWidgets.QLabel("0 / 0"); self.lblSag.setAlignment(QtCore.Qt.AlignRight|QtCore.Qt.AlignVCenter)
        h_sag = QtWidgets.QHBoxLayout(); w_sag = QtWidgets.QWidget()
        h_sag.addWidget(self.viewSag, 1); h_sag.addWidget(self.sliderSag); h_sag.addWidget(self.lblSag)
        w_sag.setLayout(h_sag)
        # 放入 2×2 网格：左上 Axial，右上 Coronal，左下 Sagittal，右下 3D
        container.addWidget(w_ax,  0, 0)
        container.addWidget(w_cor, 0, 1)
        container.addWidget(w_sag, 1, 0)
        container.addWidget(view3d_widget, 1, 1)

    
# =============================
# 3D 视图（vispy SceneCanvas 嵌入）
# =============================
class Vispy3DWidget(QtWidgets.QWidget):
    """将 vispy.scene.SceneCanvas 嵌入到 Qt 小部件中。
    后续可在此添加体渲染/中心线(GLLine)/点选交互等。
    """
    def __init__(self, parent=None):
        super().__init__(parent)
        # 创建 Canvas（使用 native QWidget）
        # self.canvas = scene.SceneCanvas(keys=None, bgcolor=(0.1, 0.1, 0.1, 1.0), size=(600, 600), show=False)
        self.canvas = scene.SceneCanvas(keys=None, bgcolor=(0.1, 0.1, 0.1, 1.0), size=(200, 200), show=False)

        self.view = self.canvas.central_widget.add_view()
        self.view.camera = 'turntable'  # 旋转相机
        self.view.camera.fov = 45
        self.view.camera.distance = 600

        # 轴与网格（占位）
        axis = scene.visuals.XYZAxis(parent=self.view.scene)
        grid = scene.visuals.GridLines(color=(0.4, 0.4, 0.4, 1), parent=self.view.scene)
        grid.transform = scene.transforms.STTransform(translate=(0, 0, 0))

        lay = QtWidgets.QVBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        self.canvas.native.setMinimumSize(100, 100)
        self.canvas.native.setSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Expanding)

        lay.addWidget(self.canvas.native)


    def clear(self):
        """清空非基础对象（保留坐标轴/网格）"""
        for obj in (getattr(self, "_vol_image", None),
                    getattr(self, "_iso_mask", None),
                    getattr(self, "_vol_mask", None)):
            if obj is not None:
                obj.parent = None
        self._vol_image = None
        self._iso_mask = None
        self._vol_mask = None

    def show_volume(self, vol, spacing=(1,1,1), clim=None):
        """显示灰度体（图像），vol 形状 [Z,Y,X]"""
        try:
            clim = clim or (float(np.min(vol)), float(np.max(vol)))
            if not hasattr(self, "_vol_image") or self._vol_image is None:
                self._vol_image = scene.visuals.Volume(vol, clim=clim, parent=self.view.scene)
            else:
                self._vol_image.set_data(vol)
                self._vol_image.clim = clim
            self._vol_image.transform = scene.transforms.STTransform(scale=spacing)
        except Exception:
            pass

    def show_mask_isosurface(self, mask, spacing=(1,1,1), level=0.5, color=(1,0,0,0.35)):
        """显示二值mask的等值面"""
        if hasattr(self, "_iso_mask") and self._iso_mask is not None:
            self._iso_mask.parent = None
            self._iso_mask = None
        try:
            self._iso_mask = scene.visuals.Isosurface(mask.astype(float), level=level, color=color, parent=self.view.scene)
            self._iso_mask.transform = scene.transforms.STTransform(scale=spacing)
        except Exception:
            pass

    def show_mask_volume(self, mask, spacing=(1,1,1), clim=(0,1)):
        """备选：mask 体渲染（半透明）"""
        try:
            if not hasattr(self, "_vol_mask") or self._vol_mask is None:
                self._vol_mask = scene.visuals.Volume(mask.astype(float), clim=clim, parent=self.view.scene)
            else:
                self._vol_mask.set_data(mask.astype(float))
            self._vol_mask.transform = scene.transforms.STTransform(scale=spacing)
        except Exception:
            pass


# =============================
# 右侧展示面板（路径 + 2D + 3D）
# =============================
class RightPanel(QtWidgets.QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self._build_ui()

    def _build_ui(self):
        # root = QtWidgets.QVBoxLayout(self)
        # root.setContentsMargins(8, 8, 8, 8)
        # root.setSpacing(6)
        root = QtWidgets.QVBoxLayout(self); root.setContentsMargins(8, 8, 8, 8); root.setSpacing(6)
  

        # 顶部两行路径（英文字段名；可选择复制）
        self.lblImagePath = QtWidgets.QLabel("Image: -")
        self.lblMaskPath  = QtWidgets.QLabel("Mask : -")
        mono = "font-family: Consolas, Menlo, monospace; color:#666;"
        self.lblImagePath.setStyleSheet(mono)
        self.lblMaskPath.setStyleSheet(mono)
        self.lblImagePath.setTextInteractionFlags(QtCore.Qt.TextSelectableByMouse)
        self.lblMaskPath.setTextInteractionFlags(QtCore.Qt.TextSelectableByMouse)
        root.addWidget(self.lblImagePath)
        root.addWidget(self.lblMaskPath)

        # # 中部左右分栏：左 2D 三视图，右 3D 视图
        # splitter = QtWidgets.QSplitter(QtCore.Qt.Horizontal)

        # self.views2d = ThreeView2D()
        # splitter.addWidget(self.views2d)

        # self.view3d = Vispy3DWidget()
        # splitter.addWidget(self.view3d)

        # splitter.setStretchFactor(0, 3)
        # splitter.setStretchFactor(1, 4)
        # root.addWidget(splitter, 1)
        # 2×2 网格：Ax / Cor / Sag / 3D
        grid = QtWidgets.QGridLayout(); grid.setHorizontalSpacing(8); grid.setVerticalSpacing(8)
        grid.setColumnStretch(0, 1)
        grid.setColumnStretch(1, 1)
        grid.setRowStretch(0, 1)
        grid.setRowStretch(1, 1)
        root.addLayout(grid, 1)
        self.views2d = ThreeView2D()
        self.view3d  = Vispy3DWidget()
        self.views2d.rearrange_to_2x2(grid, self.view3d)
  

        # === 新增：按屏幕可用区域做窗口尺寸 clamp，避免超屏 ===
        screen_geo = QtWidgets.QApplication.primaryScreen().availableGeometry()
        max_w = int(screen_geo.width()  * 0.9)
        max_h = int(screen_geo.height() * 0.9)

        # 右侧面板作为中央大容器的一部分，其父窗口由 MainWindow 统一 resize。
        # 这里仅保存最大尺寸供 MainWindow 参考（也可以直接在 MainWindow 里 clamp）。
        self._max_size_hint = (max_w, max_h)

# =============================
# 主窗口
# =============================
class MainWindow(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Artery GUI — QtCompat Skeleton")
        self.resize(1480, 600)
        self.img_vol = None
        self.mask_vol = None
        self.spacing = None
        self._fit_done = {'ax': False, 'cor': False, 'sag': False}

        self._build_ui()
        self._connect_signals()

        # 占位状态
        self.img_path: Path | None = None
        self.mask_path: Path | None = None

    def _build_ui(self):
        central = QtWidgets.QWidget(); self.setCentralWidget(central)
        h = QtWidgets.QHBoxLayout(central)

        self.left  = LeftPanel()
        self.right = RightPanel()

        splitter = QtWidgets.QSplitter(QtCore.Qt.Horizontal)
        splitter.addWidget(self.left)
        splitter.addWidget(self.right)
        splitter.setStretchFactor(0, 0)
        splitter.setStretchFactor(1, 1)
        h.addWidget(splitter)

        # 菜单栏（示例，英文）。后续可加入 Settings / Tracing / Labeling 等
        menu = self.menuBar().addMenu("&File")
        actOpenImg = QtGui.QAction("&Open Image", self)
        actOpenMsk = QtGui.QAction("&Open Mask", self)
        menu.addAction(actOpenImg); menu.addAction(actOpenMsk)
        actOpenImg.triggered.connect(self.left.sigLoadImage)
        actOpenMsk.triggered.connect(self.left.sigLoadMask)

        self.statusBar()
        if hasattr(self.right, "_max_size_hint"):
            max_w, max_h = self.right._max_size_hint
            w = min(self.width(),  max_w)
            h = min(self.height(), max_h)
            self.resize(w, h)


    def _connect_signals(self):
        L, R = self.left, self.right
        L.sigLoadImage.connect(self.on_load_image)
        L.sigLoadMask.connect(self.on_load_mask)
        L.sigRunInference.connect(self.on_run_inference)
        L.sigExtractCenterline.connect(self.on_extract_centerline)
        L.sigSelectBox.connect(self.on_toggle_box)
        L.sigClearBox.connect(self.on_clear_box)

        R.views2d.sigSliceChanged.connect(self.on_slice_changed)

        L.ckShowPred2D.toggled.connect(lambda _: self._refresh_all_views())
        L.ckShowPred3D.toggled.connect(lambda _: self._update_3d_scene())
        L.ckShowDebone2D.toggled.connect(lambda _: self._refresh_all_views())
        L.ckShowDebone3D.toggled.connect(lambda _: self._update_3d_scene())
        L.sldImgAlpha.valueChanged.connect(lambda _: self._apply_opacity_2d())
        L.sldMaskAlpha.valueChanged.connect(lambda _: self._refresh_all_views())
        # 窗宽窗位联动：2D/3D 同步
        # L.dsbWLLow.valueChanged.connect(lambda _: (self._refresh_all_views(), self._update_3d_scene()))
        # L.dsbWLHigh.valueChanged.connect(lambda _: (self._refresh_all_views(), self._update_3d_scene()))
        L.dsbWLLow.valueChanged.connect(lambda _: (self._apply_wl_to_2d(), self._update_3d_scene()))
        L.dsbWLHigh.valueChanged.connect(lambda _: (self._apply_wl_to_2d(), self._update_3d_scene()))        
                
        R.views2d.sliderAx.valueChanged.connect(lambda v: R.views2d.lblAx.setText(f"{v} / {R.views2d.sliderAx.maximum()}"))
        R.views2d.sliderCor.valueChanged.connect(lambda v: R.views2d.lblCor.setText(f"{v} / {R.views2d.sliderCor.maximum()}"))
        R.views2d.sliderSag.valueChanged.connect(lambda v: R.views2d.lblSag.setText(f"{v} / {R.views2d.sliderSag.maximum()}"))
   
    
    # ----------- 占位事件处理 -----------
    
    def _ask_type_dialog(self, default='Image') -> str:
        """
        弹窗选择要把所选文件当作 Image 还是 Mask，默认 Image。
        返回 'Image' 或 'Mask'
        """
        dlg = QtWidgets.QMessageBox(self)
        dlg.setWindowTitle("Open As")
        dlg.setText("Open this file as?")
        img_btn = dlg.addButton("Image", QtWidgets.QMessageBox.AcceptRole)
        msk_btn = dlg.addButton("Mask", QtWidgets.QMessageBox.DestructiveRole)
        # 默认选中 Image
        dlg.setDefaultButton(img_btn if default == 'Image' else msk_btn)
        dlg.exec_()
        return 'Mask' if dlg.clickedButton() is msk_btn else 'Image'
    
    def on_load_image(self):
        path, _ = QtWidgets.QFileDialog.getOpenFileName(
            self, "Choose Image or Mask", '', "NIfTI (*.nii *.nii.gz)")
        if not path:
            return

        kind = self._ask_type_dialog(default='Image')

        if kind == 'Mask':
            # 走你现有的 Mask 分支（保持不变）
            self.mask_path = Path(path)
            self.right.lblMaskPath.setText(f"Mask : {self.mask_path}")
            try:
                m = sitk.ReadImage(str(self.mask_path))
                self.mask_vol = sitk.GetArrayFromImage(m)  # [Z, Y, X]
            except Exception as e:
                QtWidgets.QMessageBox.critical(self, "Read Error", f"Failed to read mask: {e}")
                return
            self._refresh_all_views()
            self._update_3d_scene()
            self.statusBar().showMessage("Mask loaded.", 2500)
            return

        # ===== 下面是“作为 Image 打开”的分支（新增/补全）=====
        self.img_path = Path(path)
        self.right.lblImagePath.setText(f"Image: {self.img_path}")
        try:
            im = sitk.ReadImage(str(self.img_path))
            self.img_vol = sitk.GetArrayFromImage(im)       # 形状 [Z, Y, X]
            # SITK spacing 顺序是 (x, y, z)，与 numpy [Z,Y,X] 相反，这里倒序以便 vispy 正确缩放
            self.spacing = tuple(reversed(im.GetSpacing())) # -> (z, y, x)
            self._deboned_vol = None

        except Exception as e:
            QtWidgets.QMessageBox.critical(self, "Read Error", f"Failed to read image: {e}")
            return

        # >>> 这里就该设置 3 个滑条的范围和值（你问的那段代码就放在这里）
        V = self.right.views2d
        Z, Y, X = self.img_vol.shape
        V.sliderAx.setRange(0, Z-1); V.sliderAx.setValue(Z//2)
        V.sliderCor.setRange(0, Y-1); V.sliderCor.setValue(Y//2)
        V.sliderSag.setRange(0, X-1); V.sliderSag.setValue(X//2)

        # 同步数字标签（当前/最大）
        V.lblAx.setText(f"{V.sliderAx.value()} / {V.sliderAx.maximum()}")
        V.lblCor.setText(f"{V.sliderCor.value()} / {V.sliderCor.maximum()}")
        V.lblSag.setText(f"{V.sliderSag.value()} / {V.sliderSag.maximum()}")


        # 刷新 2D/3D 显示
        self._fit_done = {'ax': False, 'cor': False, 'sag': False} 
        self._refresh_all_views()
        self._update_3d_scene()
        self._apply_wl_to_2d() 
        self.statusBar().showMessage("Image loaded.", 2500)


    def _apply_wl_to_2d(self):
        if self.img_vol is None:
            return
        V = self.right.views2d
        low, high = self._wl_to_levels()
        for it in (V.img_ax, V.img_cor, V.img_sag):
            it.setLevels((low, high))

    @Slot()
    def on_load_mask(self):
        path, _ = QtWidgets.QFileDialog.getOpenFileName(self, "Choose Mask", '', "NIfTI (*.nii *.nii.gz)")
        if not path:
            return
        self.mask_path = Path(path)
        self.right.lblMaskPath.setText(f"Mask : {self.mask_path}")
        # 真实读入
        try:
            m = sitk.ReadImage(str(self.mask_path))
            self.mask_vol = sitk.GetArrayFromImage(m)  # [Z, Y, X]
        except Exception as e:
            QtWidgets.QMessageBox.critical(self, "Read Error", f"Failed to read mask: {e}")
            return
        self.left.ckShowPred2D.setChecked(True)
        self._deboned_vol = None  

             
        # 刷新显示
        self._refresh_all_views()
        self._update_3d_scene()
        self.statusBar().showMessage("Mask loaded.", 2500)

    def _update_3d_scene(self):
        """根据复选框状态刷新 3D（图像体 + mask 等值面）"""
        v3d = self.right.view3d
        v3d.clear()
        # 图像体
        if self.img_vol is not None:
            base = self._current_base_volume(for_3d=True)
            vmin, vmax = self._wl_to_levels()
            v3d.show_volume(base, spacing=self.spacing or (1,1,1), clim=(vmin, vmax))

            
        # 分割/Mask
        if self.mask_vol is not None and self.left.ckShowPred3D.isChecked():
            try:
                v3d.show_mask_isosurface((self.mask_vol > 0).astype(np.float32),
                                        spacing=self.spacing or (1,1,1),
                                        level=0.5, color=(1,0,0,0.35))
            except Exception:
                v3d.show_mask_volume((self.mask_vol > 0).astype(np.float32),
                                    spacing=self.spacing or (1,1,1), clim=(0,1))

    def _wl_to_levels(self):
        """兜底的窗口/窗位：根据体数据分位数估计显示范围"""
 
        low = float(self.left.dsbWLLow.value())
        high = float(self.left.dsbWLHigh.value())
        if high <= low:
            high = low + 1.0
        return low, high

    def _make_deboned(self):
        """根据 mask 生成‘去骨’底图（mask==1 的体素置为 0）；无 mask 时返回原图。"""
        if self.img_vol is None or self.mask_vol is None:
            return self.img_vol
        if getattr(self, "_deboned_vol", None) is None:
            self._deboned_vol = np.where(self.mask_vol.astype(bool), self.img_vol, 0)

        return self._deboned_vol
    

    def _current_base_volume(self, for_3d: bool):
        """根据 2D/3D 开关选择显示哪份底图"""
        use_debone = self.left.ckShowDebone3D.isChecked() if for_3d else self.left.ckShowDebone2D.isChecked()
        if use_debone:
            return self._make_deboned()
        return self.img_vol


    def _refresh_mask_view(self, axis: str, idx: int):
        """刷新某个视角的 Mask 叠加层（使用 LUT 控制 RGBA）"""
        if self.mask_vol is None or not self.left.ckShowPred2D.isChecked():
            # 关闭叠加时清空
            V = self.right.views2d
            {'ax': V.mask_ax, 'cor': V.mask_cor, 'sag': V.mask_sag}[axis].clear()
            return

        V = self.right.views2d
        if axis == 'ax':
            sl = self.mask_vol[idx, :, :]
            item = V.mask_ax
        elif axis == 'cor':
            sl = self.mask_vol[:, idx, :]
            item = V.mask_cor
        else:
            sl = self.mask_vol[:, :, idx]
            item = V.mask_sag

        # sl = (sl > 0).astype(np.uint8)
        # alpha = int(np.clip(self.left.sldMaskAlpha.value(), 0, 100) * 2.55)
        # lut = np.zeros((2, 4), dtype=np.ubyte)   # 0=透明, 1=红色半透明
        # lut[1] = (255, 0, 0, alpha)
        # item.setLookupTable(lut)
        # vmin, vmax = self._wl_to_levels()
        # item.setImage(sl.T, autoLevels=False, levels=(vmin, vmax))

        self.right.views2d.ensure_mask_layers()
        sl = (sl > 0).astype(np.uint8)
        alpha = int(np.clip(self.left.sldMaskAlpha.value(), 0, 100) * 2.55)
        lut = np.zeros((2, 4), dtype=np.ubyte)
        lut[1] = (255, 80, 80, alpha)  # 略亮一点
        item.setLookupTable(lut)
        item.setImage(sl.T, autoLevels=False)


    def _apply_opacity_2d(self):
        """应用底图不透明度（Mask 用 LUT 的 alpha 控制）"""
        V = self.right.views2d
        img_alpha = float(self.left.sldImgAlpha.value()) / 100.0
        for it in (V.img_ax, V.img_cor, V.img_sag):
            it.setOpacity(img_alpha)

    def _fit_once(self, axis: str):
        """首次加载时让该视图自动缩放以完整显示切片；之后不再自动缩放，避免用户浏览被打断。"""
        if self._fit_done.get(axis):
            return
        V = self.right.views2d
        vb = {'ax': V.viewAx._vb, 'cor': V.viewCor._vb, 'sag': V.viewSag._vb}[axis]
        img = {'ax': V.img_ax,    'cor': V.img_cor,    'sag': V.img_sag}[axis]
        try:
            vb.setRange(img.mapRectToParent(img.boundingRect()), padding=0.02)
        except Exception:
            vb.autoRange(padding=0.02)
        self._fit_done[axis] = True

    def _clamp_to_screen(self):
        screen = QtWidgets.QApplication.primaryScreen().availableGeometry()
        # 限制最大尺寸不超过可用屏幕
        self.setMaximumSize(screen.width(), screen.height())
        w = min(self.width(),  int(screen.width()  * 0.9))
        h = min(self.height(), int(screen.height() * 0.9))
        self.resize(w, h)

    def showEvent(self, e):
        super().showEvent(e)
        self._clamp_to_screen()

    @Slot()
    def on_run_inference(self):
        endpoint = self.left.leEndpoint.text().strip()
        if not endpoint:
            QtWidgets.QMessageBox.warning(self, "Notice", "Please set Endpoint")
            return
        QtWidgets.QMessageBox.information(self, "Inference (stub)", f"Will call remote endpoint: {endpoint}")
        # TODO：QThread/QtConcurrent 异步上传去骨体与 ROI，接收预测体并显示

    @Slot()
    def on_extract_centerline(self):
        QtWidgets.QMessageBox.information(self, "Centerline (stub)", "Extract centerline…")
        # TODO：对预测Mask进行三维骨架化，生成折线；vispy 3D 绘制

    @Slot(bool)
    def on_toggle_box(self, checked: bool):
        self.statusBar().showMessage(f"Box Select: {'ON' if checked else 'OFF'}", 1500)
        # TODO：进入/退出框选模式；在当前 2D 视图捕获矩形并保存到 state

    @Slot()
    def on_clear_box(self):
        self.statusBar().showMessage("Cleared ROI box", 1500)
        # TODO：清除 ROI 并刷新显示

    @Slot(str, int)
    def on_slice_changed(self, axis: str, idx: int):
        if self.img_vol is None:
            return
        self._refresh_view(axis, idx)
        if self.mask_vol is not None:
            self._refresh_mask_view(axis, idx)
        self._apply_opacity_2d()

    def _refresh_all_views(self):
        V = self.right.views2d
        self._refresh_view('ax', V.sliderAx.value())
        self._refresh_view('cor', V.sliderCor.value())
        self._refresh_view('sag', V.sliderSag.value())
        if self.mask_vol is not None:
            self._refresh_mask_view('ax', V.sliderAx.value())
            self._refresh_mask_view('cor', V.sliderCor.value())
            self._refresh_mask_view('sag', V.sliderSag.value())
        self._apply_opacity_2d()

    def _refresh_view(self, axis: str, idx: int):
        """刷新某个 2D 视图的底图（若有）。需要在加载影像后设置 slider 的范围。"""
        if self.img_vol is None:
            return
        V = self.right.views2d
        if axis == 'ax':
            base = self._current_base_volume(for_3d=False)
            sl = base[idx, :, :]
            item = V.img_ax

        elif axis == 'cor':
            base = self._current_base_volume(for_3d=False)
            sl = base[:, idx, :]
            item = V.img_cor
        else:
            base = self._current_base_volume(for_3d=False)
            sl = base[:, :, idx]
            item = V.img_sag
        # 注意转置以匹配 pyqtgraph 坐标

        low, high = self._wl_to_levels()
        item.setImage(sl.T, autoLevels=False, levels=(low, high))
        self._fit_once(axis)



# =============================
# 入口
# =============================
def main():
    app = QtWidgets.QApplication(sys.argv)

    # 可选：主题
    try:
        apply_stylesheet(app, theme='dark_teal.xml')
    except Exception:
        pass

    # pyqtgraph 配置（与医学影像数组行主序对应，按需调整）
    pg.setConfigOptions(imageAxisOrder='row-major')

    win = MainWindow()
    win.show()
    sys.exit(app.exec_())


if __name__ == '__main__':
    main()
