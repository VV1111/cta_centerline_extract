#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Artery GUI — Qt-compatible skeleton using pyqtgraph (PyQt5/6 or PySide2/6)

Requirements
- pyqtgraph
- nibabel
- numpy

Run
    python ArteryGUI_QtCompat_Skeleton_v1.py

Notes
- Import style is fixed to: `import pyqtgraph as pg` and
  `from pyqtgraph.Qt import QtCore, QtWidgets, QtGui` per your request.
- Left column: toolbar with "Open NIfTI" and basic controls.
- Middle column: three small preview panes (Axial / Coronal / Sagittal) + a single global slice slider.
- Right column: a large detail view that mirrors (and controls) the selected preview.
- Click the 🔍 icon on a preview to show that plane in the large detail view.
- Panning/zooming in the right (detail) view is mirrored back to the active preview.
- Any future drawing/ROI in the big view should use the same ViewBox transform to map to image coords.

This is a minimal but clean, extensible scaffold.
"""

import sys
import os
import numpy as np
import nibabel as nib
import pyqtgraph as pg
from pyqtgraph.Qt import QtCore, QtWidgets, QtGui
from pyqtgraph.Qt.QtGui import QKeySequence, QShortcut
import pyqtgraph.opengl as gl


# ---------- Utility: centralize plane names ----------
PLANES = ["Axial", "Coronal", "Sagittal"]


class VolumeData(QtCore.QObject):
    """Hold a 3D numpy array and expose plane-wise slice counts & getters."""
    dataChanged = QtCore.Signal()
    paramsChanged = QtCore.Signal() 

    def __init__(self):
        super().__init__()
 
        self.window_level = None   # None 表示未启用 WL
        self.window_width = None   # None 表示未启用 WW
        self._gmin = None          # 全局最小值（加载时计算）
        self._gmax = None          # 全局最大值（加载时计算）

        self._vol = None  # numpy array, shape (Z, Y, X) or any 3D
        self._affine = None  # NIfTI affine (unused now, reserved)
        # Current slice indices for each plane
        self.slices = {"Axial": 0, "Coronal": 0, "Sagittal": 0}
        
        # mask 相关
        self._mask = None          # 整型标签体素，形状与 _vol 一致
        self.mask_alpha = 0.6      # 0~1 之间
        # self.display_mode = "overlay"   # "overlay" | "image_only" | "mask_only"
        self.display_mode = "overlay"   # "overlay" | "image_only" | "mask_only" | "image_masked"

        self._lut_colors = None    # (K, 4) RGBA uint8 颜色表（A通道不用，按像素填）
        
    def is_loaded(self):
        return self._vol is not None and self._vol.ndim == 3

    def load_nii(self, path: str):
        img = nib.load(path)
        arr = img.get_fdata(dtype=np.float32)

        self._vol = np.asarray(arr)
        self._affine = img.affine
        # Reset slice indices to center slices
        z, y, x = self.shape_zyx()
        self.slices["Axial"] = z // 2
        self.slices["Coronal"] = y // 2
        self.slices["Sagittal"] = x // 2
        
        flat = self._vol[np.isfinite(self._vol)].ravel()
        if flat.size > 0:
            self._gmin, self._gmax = float(np.min(flat)), float(np.max(flat))
            if self._gmax <= self._gmin:
                self._gmax = self._gmin + 1.0
        else:
            self._gmin, self._gmax = 0.0, 1.0

        self.dataChanged.emit()
        self.paramsChanged.emit()

    def load_mask(self, path: str):
        img = nib.load(path)
        arr = img.get_fdata(dtype=np.float32)
        self._mask = np.asarray(np.rint(arr).astype(np.int32))  # 四舍五入到整数标签
        # 简单保护：形状不一致时尝试广播失败就报错
        if self._vol is None or self._mask.shape != self._vol.shape:
            raise ValueError(f"Mask shape {self._mask.shape} != image shape {None if self._vol is None else self._vol.shape}")
        # 构建颜色表
        self._build_lut_from_mask()
        self.paramsChanged.emit()   # 触发渲染刷新

    def _build_lut_from_mask(self):
        if self._mask is None:
            self._lut_colors = None
            return
        labels = np.unique(self._mask)
        max_label = int(labels.max()) if labels.size else 0
        K = max(2, max_label + 1)   # 至少包含 0 和最大标签
        lut = np.zeros((K, 4), dtype=np.uint8)  # RGBA
        # 0 号背景透明（颜色随意，这里置零）
        lut[0] = (0, 0, 0, 0)
        # 为 1..K-1 生成可分辨颜色（HSV 均匀取色）
        for i in range(1, K):
            h = (i * 0.61803398875) % 1.0  # 黄金比例避免相近
            s, v = 0.9, 0.95
            r, g, b = self._hsv_to_rgb(h, s, v)
            lut[i] = (int(r*255), int(g*255), int(b*255), 255)
        self._lut_colors = lut


    @staticmethod
    def _hsv_to_rgb(h, s, v):
        import colorsys
        return colorsys.hsv_to_rgb(h, s, v)

    def set_mask_alpha(self, a: float):
        a = float(np.clip(a, 0.0, 1.0))
        if a != self.mask_alpha:
            self.mask_alpha = a
            self.paramsChanged.emit()

    def set_display_mode(self, mode: str):
        mode = mode.lower()
        if mode not in ("overlay", "image_only", "mask_only", "image_masked"):
            mode = "overlay"
        if mode != self.display_mode:
            self.display_mode = mode
            self.paramsChanged.emit()

    def has_mask(self) -> bool:
        return self._mask is not None

    def get_mask_slice(self, plane: str, idx: int) -> np.ndarray | None:
        if not self.has_mask():
            return None
        z, y, x = self._mask.shape
        if plane == "Axial":
            idx = np.clip(idx, 0, z - 1);  return self._mask[idx, :, :]
        if plane == "Coronal":
            idx = np.clip(idx, 0, y - 1);  return self._mask[:, idx, :]
        if plane == "Sagittal":
            idx = np.clip(idx, 0, x - 1);  return self._mask[:, :, idx]
        return None

    def render_mask_rgba_slice(self, plane: str, idx: int) -> np.ndarray | None:
        """将标签切片（int）映射为 RGBA（HxWx4 uint8），背景标签0→alpha=0。"""
        if not self.has_mask() or self._lut_colors is None:
            return None
        lab = self.get_mask_slice(plane, idx)
        if lab is None:
            return None
        lab = np.asarray(lab, dtype=np.int32)
        lut = self._lut_colors
        K = lut.shape[0]
        lab_clip = np.clip(lab, 0, K-1)
        rgba = lut[lab_clip]                    # (H, W, 4)
        # 应用全局透明度：标签>0 才使用 alpha
        if self.mask_alpha < 1.0:
            alpha = rgba[..., 3].astype(np.float32) * self.mask_alpha
            rgba = rgba.copy()
            rgba[..., 3] = np.clip(alpha, 0, 255).astype(np.uint8)
        return rgba

    def render_image_masked_slice(self, plane: str, idx: int) -> np.ndarray | None:
        """
        返回只保留 mask>0 部分的灰度图（float32, 0~1），mask 外设为 0。
        若未加载 mask，则返回普通渲染结果（minmax 或 WL/WW）。
        """
        base = self.render_slice(plane, idx)
        if base is None:
            return None
        if not self.has_mask():
            return base
        lab = self.get_mask_slice(plane, idx)
        if lab is None:
            return base
        m = (lab > 0).astype(np.float32)
        # 注意：base 是按照 (row,col)；后续 setImage(img.T) 会转置，所以这里不转置
        out = base * m
        return np.nan_to_num(out, nan=0.0, posinf=1.0, neginf=0.0).astype(np.float32, copy=False)

    def render_slice(self, plane: str, idx: int) -> np.ndarray:
        img = self.get_slice(plane, idx)
        if img is None:
            return None
        if self.window_level is not None and self.window_width is not None:
            return self._apply_window(img)
        else:
            return self._apply_minmax(img)
        
    def _apply_window(self, img2d: np.ndarray) -> np.ndarray:
        L = float(self.window_level)
        W = float(max(1e-6, self.window_width))
        lo, hi = L - W * 0.5, L + W * 0.5
        out = np.clip((img2d - lo) / (hi - lo), 0.0, 1.0)
        out = np.nan_to_num(out, nan=0.0, posinf=1.0, neginf=0.0).astype(np.float32, copy=False)
        return out


    def _apply_minmax(self, img2d: np.ndarray) -> np.ndarray:
        gmin = self._gmin if self._gmin is not None else float(np.nanmin(img2d))
        gmax = self._gmax if self._gmax is not None else float(np.nanmax(img2d))
        if not np.isfinite(gmin) or not np.isfinite(gmax) or gmax <= gmin:
            gmin, gmax = 0.0, 1.0
        out = np.clip((img2d - gmin) / (gmax - gmin), 0.0, 1.0)
        # 可选兜底
        out = np.nan_to_num(out, nan=0.0, posinf=1.0, neginf=0.0).astype(np.float32, copy=False)
        return out




    def set_window(self, level: float, width: float):
        # 若想“清空 WL/WW 回到 MinMax”，可以传 None（UI 层不需要的话可不做）
        if level is None or width is None:
            changed = (self.window_level is not None) or (self.window_width is not None)
            self.window_level, self.window_width = None, None
        else:
            level = float(level)
            width = float(max(1e-6, width))
            changed = (self.window_level != level) or (self.window_width != width)
            self.window_level, self.window_width = level, width
        if changed:
            self.paramsChanged.emit()

    def shape_zyx(self):
        if not self.is_loaded():
            return (0, 0, 0)
        # Interpret array as (Z, Y, X)
        z, y, x = self._vol.shape
        return z, y, x

    def max_index(self, plane: str) -> int:
        z, y, x = self.shape_zyx()
        if plane == "Axial":
            return max(0, z - 1)
        if plane == "Coronal":
            return max(0, y - 1)
        if plane == "Sagittal":
            return max(0, x - 1)
        return 0

    def get_slice(self, plane: str, idx: int) -> np.ndarray:
        """Return raw 2D slice (float32) for a given plane+index (no normalization)."""
        if not self.is_loaded():
            return None
        z, y, x = self._vol.shape
        if plane == "Axial":
            idx = np.clip(idx, 0, z - 1)
            return self._vol[idx, :, :]
        if plane == "Coronal":
            idx = np.clip(idx, 0, y - 1)
            return self._vol[:, idx, :]
        if plane == "Sagittal":
            idx = np.clip(idx, 0, x - 1)
            return self._vol[:, :, idx]
        return None


class ImagePreview(QtWidgets.QFrame):
    """A small preview widget with a title bar and a 🔍 button to promote to the detail view."""
    # zoomRequested = QtCore.Signal(str)  # plane name
    zoomRequested = QtCore.Signal(object) 
    sliceChanged = QtCore.Signal(str, int)
    # update3DRequested = QtCore.Signal()


    def __init__(self, plane: str, volume: VolumeData):
        super().__init__()
        self.setObjectName(f"Preview_{plane}")
        self.plane = plane
        self.volume = volume
        
        self.setFrameShape(QtWidgets.QFrame.StyledPanel)
        self.setFrameShadow(QtWidgets.QFrame.Raised)

        # Title bar
        title = QtWidgets.QHBoxLayout()
        self.lbl = QtWidgets.QLabel(f"{plane}")
        self.lbl.setStyleSheet("font-weight: 600;")

        self.hdrSlider = QtWidgets.QSlider(QtCore.Qt.Horizontal)
        self.hdrSlider.setMinimum(0)
        self.hdrSlider.setMaximum(0)
        self.hdrSlider.setSingleStep(1)
        self.hdrSlider.setPageStep(5)
        self.hdrSlider.setFixedHeight(18)
        self.hdrSlider.valueChanged.connect(lambda v: self.set_slice(int(v)))

        self.btnZoom = QtWidgets.QToolButton()
        self.btnZoom.setText("🔍")
        self.btnZoom.setToolTip("Show this plane in the large view")
        # self.btnZoom.clicked.connect(lambda: self.zoomRequested.emit(self.plane))
        self.btnZoom.clicked.connect(lambda: self.zoomRequested.emit(self))
        # title.addWidget(self.lbl)
        # title.addWidget(self.hdrSlider, 1) 
        # title.addWidget(self.btnZoom)

        title.addWidget(self.lbl)

        # 中间的细长 slider
        title.addWidget(self.hdrSlider, 1)

        # 新增：Update 3D 按钮（点击后才渲染3D）
        # TODO 这个应该是只有这个3d 预览才有其他没有
        # self.btnUpdate3D = QtWidgets.QToolButton()
        # self.btnUpdate3D.setText("Update")
        # self.btnUpdate3D.setToolTip("Regenerate 3D preview with current mode")
        # self.btnUpdate3D.clicked.connect(self.update3DRequested.emit)

        # title.addWidget(self.btnUpdate3D)

        # 放大按钮
        title.addWidget(self.btnZoom)

        # Graphics view
        self.glw = pg.GraphicsLayoutWidget()
        self.view = self.glw.addViewBox(lockAspect=True, enableMenu=False)
        self.view.setMouseEnabled(x=True, y=True)
        self.img_item = pg.ImageItem()
        self.view.addItem(self.img_item)
        self.img_item.setLevels((0.0, 1.0)) 

        self.mask_item = pg.ImageItem()          # 顶层 mask（RGBA）
        self.mask_item.setZValue(10)
        self.view.addItem(self.mask_item)

        # Local slice slider for this preview (hidden; we use global slider primarily)
        self.slider = QtWidgets.QSlider(QtCore.Qt.Horizontal)
        self.slider.setMinimum(0)
        self.slider.setMaximum(0)
        self.slider.valueChanged.connect(self._on_local_slider_changed)
        self.slider.setVisible(False)  # keep it hidden to honor the single global slider design

        lay = QtWidgets.QVBoxLayout(self)
        lay.setContentsMargins(6, 6, 6, 6)
        lay.addLayout(title)
        lay.addWidget(self.glw)
        lay.addWidget(self.slider)

        # React to data changes
        
        self.volume.dataChanged.connect(self.refresh)
        self.volume.paramsChanged.connect(self.refresh)


    def _on_local_slider_changed(self, v: int):
        self.sliceChanged.emit(self.plane, v)

    def refresh(self):
        if not self.volume.is_loaded():
            self.img_item.clear()
            self.slider.setMaximum(0)
            self.slider.setValue(0)
            return
        max_idx = self.volume.max_index(self.plane)
        self.slider.setMaximum(max_idx)
        # Use current index in VolumeData
        idx = self.volume.slices[self.plane]
        self.slider.setValue(idx)

        self.hdrSlider.blockSignals(True)
        self.hdrSlider.setMaximum(max_idx)     
        self.hdrSlider.setValue(idx)
        self.hdrSlider.blockSignals(False)


        mode = self.volume.display_mode
        if mode == "image_masked":
            # 只显示 mask 区域的原图
            img = self.volume.render_image_masked_slice(self.plane, idx)
            if img is not None:
                self.img_item.setImage(img.T, autoLevels=False, levels=(0.0, 1.0))
                self.view.autoRange()
            # 该模式下不显示彩色 mask 覆盖
            self.mask_item.setVisible(False)
        else:
            # 正常原图
            img = self.volume.render_slice(self.plane, idx)
            if img is not None:
                self.img_item.setImage(img.T, autoLevels=False, levels=(0.0, 1.0))
                self.view.autoRange()
            # 叠加/仅 mask / 仅图 由这个函数处理
            self._update_mask_layer(idx)
                
        
        
        
    def _update_mask_layer(self, idx: int):
        mode = self.volume.display_mode
        has_mask = self.volume.has_mask()
        if mode == "image_only" or not has_mask:
            self.mask_item.setVisible(False)
            self.img_item.setVisible(True)
            return

        # 叠加或仅mask
        rgba = self.volume.render_mask_rgba_slice(self.plane, idx)
        if rgba is None:
            self.mask_item.setVisible(False)
            self.img_item.setVisible(True)
            return

        # 注意：RGBA 不转置会导致方向错位，这里也做转置
        self.mask_item.setImage(rgba.transpose(1, 0, 2), autoLevels=False)
        self.mask_item.setVisible(True)

        if mode == "mask_only":
            self.img_item.setVisible(False)
        else:  # overlay
            self.img_item.setVisible(True)


    # External API to set slice index and update image
    def set_slice(self, idx: int):
        if not self.volume.is_loaded():
            return
        max_idx = self.volume.max_index(self.plane)
        idx = int(np.clip(idx, 0, max_idx))
        if self.volume.slices[self.plane] != idx:
            self.volume.slices[self.plane] = idx
        # img = self.volume.render_slice(self.plane, idx)
        # if img is not None:
        #     self.img_item.setImage(img.T, autoLevels=False, levels=(0.0, 1.0))

        mode = self.volume.display_mode

        if mode == "image_masked":
            # 只显示 mask 区域的原图
            img = self.volume.render_image_masked_slice(self.plane, idx)
            if img is not None:
                self.img_item.setImage(img.T, autoLevels=False, levels=(0.0, 1.0))
                self.view.autoRange()
            # 该模式下不显示彩色 mask 覆盖
            self.mask_item.setVisible(False)
        else:
            # 正常原图
            img = self.volume.render_slice(self.plane, idx)
            if img is not None:
                self.img_item.setImage(img.T, autoLevels=False, levels=(0.0, 1.0))
                self.view.autoRange()
            # 叠加/仅 mask / 仅图 由这个函数处理
            self._update_mask_layer(idx)


# class Volume3DPreview(QtWidgets.QFrame):
#     """按需渲染的3D体视图。默认显示占位图；点击 Update 后才生成体素并显示 3D。"""
#     zoomRequested = QtCore.Signal(object)

#     def __init__(self, volume: VolumeData, title="3D"):
#         super().__init__()
#         self.volume = volume
#         self.setFrameShape(QtWidgets.QFrame.StyledPanel)
#         self.setFrameShadow(QtWidgets.QFrame.Raised)

#         # --- 标题行：3D | Update | 🔍 ---
#         top = QtWidgets.QHBoxLayout()
#         self.lbl = QtWidgets.QLabel(title)
#         self.lbl.setStyleSheet("font-weight: 600;")

#         self.btnUpdate = QtWidgets.QToolButton()
#         self.btnUpdate.setText("Update")
#         self.btnUpdate.setToolTip("Generate 3D preview with current mode")
#         self.btnUpdate.clicked.connect(self.regenerate)

#         self.btnZoom = QtWidgets.QToolButton()
#         self.btnZoom.setText("🔍")
#         self.btnZoom.setToolTip("Show this 3D view in the large view")
#         self.btnZoom.clicked.connect(lambda: self.zoomRequested.emit(self))

#         top.addWidget(self.lbl)
#         top.addStretch(1)
#         top.addWidget(self.btnUpdate)
#         top.addWidget(self.btnZoom)

#         # --- 占位页（Page 0）：显示一张轻量图 ---
#         self.ph_glw = pg.GraphicsLayoutWidget()
#         self.ph_vb = self.ph_glw.addViewBox(lockAspect=True, enableMenu=False)
#         self.ph_vb.setMouseEnabled(x=False, y=False)
#         self.ph_img = pg.ImageItem()
#         self.ph_vb.addItem(self.ph_img)
#         # 初始 levels 固定到 [0,1]
#         self.ph_img.setLevels((0.0, 1.0))

#         # --- 3D 页（Page 1）：GLViewWidget ---
#         self.view3d = gl.GLViewWidget()
#         self.view3d.opts["distance"] = 200
#         self.view3d.setBackgroundColor(30, 30, 30)
#         self.vol_item = None
#         self._last_rgba = None

#         # --- 堆栈：默认显示占位页 ---
#         self.stack = QtWidgets.QStackedLayout()
#         self.stack.addWidget(self.ph_glw)   # index 0
#         self.stack.addWidget(self.view3d)   # index 1
#         self.stack.setCurrentIndex(0)

#         lay = QtWidgets.QVBoxLayout(self)
#         lay.setContentsMargins(6, 6, 6, 6)
#         lay.addLayout(top)
#         lay.addLayout(self.stack)

#         # 数据/参数变化：只刷新占位图（不做 3D 重建）
#         self.volume.dataChanged.connect(self._refresh_placeholder)
#         self.volume.paramsChanged.connect(self._refresh_placeholder)

#         # 首次占位刷新
#         self._refresh_placeholder()

#     def _refresh_placeholder(self):
#         """根据当前显示模式，生成一张轻量级占位图（不触发 3D 体渲染）。"""
#         if not self.volume.is_loaded():
#             self.ph_img.clear()
#             return

#         # 取中间层的 Axial 切片做示意；尽量与当前显示模式一致
#         z, _, _ = self.volume.shape_zyx()
#         idx = max(0, z // 2)

#         mode = self.volume.display_mode
#         if mode == "image_masked":
#             base = self.volume.render_image_masked_slice("Axial", idx)
#             if base is None:
#                 self.ph_img.clear(); return
#             self.ph_img.setImage(base.T, autoLevels=False, levels=(0.0, 1.0))
#             return

#         # 其他模式：先画灰度底
#         base = self.volume.render_slice("Axial", idx)
#         if base is None:
#             self.ph_img.clear(); return
#         self.ph_img.setImage(base.T, autoLevels=False, levels=(0.0, 1.0))

#         # 若需要叠彩色 mask
#         if mode in ("overlay", "mask_only") and self.volume.has_mask():
#             rgba = self.volume.render_mask_rgba_slice("Axial", idx)  # (H,W,4) uint8
#             if rgba is not None:
#                 # 把彩色 mask 直接画到同一 ImageItem 上会被覆盖；
#                 # 这里做一次简单的 alpha 合成，得到一张 RGB 灰度（0~1）
#                 over = self._alpha_blend_gray_rgba(base, rgba)  # 返回 0~1 float
#                 self.ph_img.setImage(over.T, autoLevels=False, levels=(0.0, 1.0))
#         # 若是 image_only，就只显示灰度
#         self.ph_vb.autoRange()

#     @staticmethod
#     def _alpha_blend_gray_rgba(gray01: np.ndarray, rgba: np.ndarray) -> np.ndarray:
#         """把 [0,1] 灰度图与 uint8 RGBA 做前景覆盖，返回 [0,1] 的近似合成结果（只为占位显示）。"""
#         g = np.clip(gray01, 0.0, 1.0).astype(np.float32)
#         rgb = rgba[..., :3].astype(np.float32) / 255.0
#         a   = rgba[..., 3].astype(np.float32) / 255.0
#         # 简单"over"合成：out = fg*a + bg*(1-a)
#         out_rgb = rgb * a[..., None] + g[..., None] * (1.0 - a[..., None])
#         # 取亮度近似（平均）转回灰度以节省绘制
#         out_g = out_rgb.mean(axis=-1)
#         return np.clip(out_g, 0.0, 1.0).astype(np.float32)


#     def clear(self):
#         if self.vol_item is not None:
#             try:
#                 self.view3d.removeItem(self.vol_item)
#             except Exception:
#                 pass
#             self.vol_item = None
#         self._last_rgba = None
#         self.stack.setCurrentIndex(0)  # 回到占位页

#     def regenerate(self):
#         if not self.volume.is_loaded():
#             self.clear()
#             return
#         rgba = self._build_rgba_from_current()
#         self._last_rgba = rgba

#         if self.vol_item is not None:
#             try:
#                 self.view3d.removeItem(self.vol_item)
#             except Exception:
#                 pass
#             self.vol_item = None

#         self.vol_item = gl.GLVolumeItem(rgba, smooth=True)
#         self.vol_item.setGLOptions('translucent')
#         self.view3d.addItem(self.vol_item)
#         self.stack.setCurrentIndex(1) 

#     def export_rgba(self):
#         """把最近一次生成的 RGBA 体素导出，供右侧 3D 大图使用。"""
#         return self._last_rgba

#     # ---------- helpers ----------
#     def _normalize_volume(self, vol: np.ndarray) -> np.ndarray:
#         L, W = self.volume.window_level, self.volume.window_width
#         if L is not None and W is not None:
#             lo, hi = float(L) - float(W) * 0.5, float(L) + float(W) * 0.5
#         else:
#             gmin = self.volume._gmin if self.volume._gmin is not None else float(np.nanmin(vol))
#             gmax = self.volume._gmax if self.volume._gmax is not None else float(np.nanmax(vol))
#             if not np.isfinite(gmin) or not np.isfinite(gmax) or gmax <= gmin:
#                 gmin, gmax = 0.0, 1.0
#             lo, hi = gmin, gmax
#         out = (vol.astype(np.float32) - lo) / max(1e-6, (hi - lo))
#         out = np.clip(out, 0.0, 1.0)
#         return np.nan_to_num(out, nan=0.0, posinf=1.0, neginf=0.0)

#     def _build_rgba_from_current(self) -> np.ndarray:
#         """根据 display_mode 构建 (X,Y,Z,4) RGBA 体素。"""
#         vol = self.volume._vol
#         norm = self._normalize_volume(vol)               # (Z,Y,X) in [0,1]
#         mode = self.volume.display_mode
#         has_mask = self.volume.has_mask()

#         if mode == "image_only" or (mode == "image_masked" and not has_mask):
#             # 灰度体素 + 强度alpha
#             rgb = (norm * 255).astype(np.ubyte)
#             a   = (norm * 0.6 * 255).astype(np.ubyte)

#             rgba = np.stack([rgb, rgb, rgb, a], axis=-1)         # (Z,Y,X,4)

#         elif mode == "image_masked" and has_mask:
#             lab = self.volume._mask
#             m = (lab > 0).astype(np.float32)
#             rgb = (norm * m * 255).astype(np.ubyte)
#             a   = (norm * m * 0.6 * 255).astype(np.ubyte)
#             rgba = np.stack([rgb, rgb, rgb, a], axis=-1)

#         elif mode == "mask_only" and has_mask:
#             lab = np.asarray(self.volume._mask, dtype=np.int32)   # (Z,Y,X)
#             lut = self.volume._lut_colors
#             K = lut.shape[0]
#             lab_clip = np.clip(lab, 0, K-1)
#             rgba = lut[lab_clip].astype(np.ubyte)                 # (Z,Y,X,4)
#             # 全局 alpha（叠一层系数，让体渲染更柔和）
#             if self.volume.mask_alpha < 1.0:
#                 a = (rgba[..., 3].astype(np.float32) * self.volume.mask_alpha).clip(0,255).astype(np.ubyte)
#                 rgba = rgba.copy()
#                 rgba[..., 3] = a

#         else:
#             # overlay: 用灰度作底，mask>0 区域微微上色/加透明度
#             rgb = (norm * 255).astype(np.ubyte)
#             a   = (norm * 0.4 * 255).astype(np.ubyte)      # 底层 alpha
#             rgba = np.stack([rgb, rgb, rgb, a], axis=-1)

#             if has_mask and self.volume._lut_colors is not None:
#                 lab = np.asarray(self.volume._mask, dtype=np.int32)
#                 K = self.volume._lut_colors.shape[0]
#                 lab_clip = np.clip(lab, 0, K-1)
#                 color = self.volume._lut_colors[lab_clip]  # (Z,Y,X,4)
#                 # 简单混色：mask像素提亮并增加alpha
#                 mask_on = (lab_clip > 0)
#                 rgba = rgba.copy()
#                 rgba[..., 0][mask_on] = np.maximum(rgba[..., 0][mask_on], color[..., 0][mask_on])
#                 rgba[..., 1][mask_on] = np.maximum(rgba[..., 1][mask_on], color[..., 1][mask_on])
#                 rgba[..., 2][mask_on] = np.maximum(rgba[..., 2][mask_on], color[..., 2][mask_on])
#                 add_a = int(255 * 0.25 * self.volume.mask_alpha)
#                 rgba[..., 3][mask_on] = np.clip(rgba[..., 3][mask_on].astype(np.int16) + add_a, 0, 255).astype(np.ubyte)

#         # (Z,Y,X,4) -> (X,Y,Z,4)
#         rgba = np.transpose(rgba, (2, 1, 0, 3)).copy(order='C')
#         return rgba

class Volume3DPreview(QtWidgets.QFrame):
    """真正的3D小预览：默认空（黑底，不渲染）；点击 Update 后在小窗直接体渲染。"""
    zoomRequested = QtCore.Signal(object)   # 传自身对象，便于 MainWindow 放大

    def __init__(self, volume: VolumeData, title="3D"):
        super().__init__()
        self.volume = volume
        self.setFrameShape(QtWidgets.QFrame.StyledPanel)
        self.setFrameShadow(QtWidgets.QFrame.Raised)

        # --- 标题行：3D | Update | 🔍 ---
        top = QtWidgets.QHBoxLayout()
        self.lbl = QtWidgets.QLabel(title)
        self.lbl.setStyleSheet("font-weight: 600;")

        self.btnUpdate = QtWidgets.QToolButton()
        self.btnUpdate.setText("Update")
        self.btnUpdate.setToolTip("Generate 3D preview with current mode")
        self.btnUpdate.clicked.connect(self.regenerate)

        self.btnZoom = QtWidgets.QToolButton()
        self.btnZoom.setText("🔍")
        self.btnZoom.setToolTip("Show this 3D view in the large view")
        self.btnZoom.clicked.connect(lambda: self.zoomRequested.emit(self))

        top.addWidget(self.lbl)
        top.addStretch(1)
        top.addWidget(self.btnUpdate)
        top.addWidget(self.btnZoom)

        # --- 3D 小窗（始终是 GLViewWidget） ---
        self.view3d = gl.GLViewWidget()
        self.view3d.opts["distance"] = 200
        self.view3d.setBackgroundColor(30, 30, 30)
        # 允许用户在小窗里用鼠标旋转/缩放
        self.view3d.setMinimumHeight(160)

        self.vol_item = None          # GLVolumeItem
        self._last_rgba = None        # 最近一次生成的 RGBA 体素 (X,Y,Z,4)

        lay = QtWidgets.QVBoxLayout(self)
        lay.setContentsMargins(6, 6, 6, 6)
        lay.addLayout(top)
        lay.addWidget(self.view3d)

        # 数据/参数变化：仅清空（按需渲染策略，不自动重建）
        self.volume.dataChanged.connect(self.clear)
        self.volume.paramsChanged.connect(self.clear)

    def clear(self):
        if self.vol_item is not None:
            try:
                self.view3d.removeItem(self.vol_item)
            except Exception:
                pass
            self.vol_item = None
        self._last_rgba = None
        # 保留黑底空视图即可

    def regenerate(self):
        """按当前 VolumeData 的 WL/WW + display_mode 生成 3D 体素，并显示在小窗。"""
        if not self.volume.is_loaded():
            self.clear()
            return

        rgba = self._build_rgba_from_current()
        self._last_rgba = rgba

        # 替换 GLVolumeItem
        if self.vol_item is not None:
            try:
                self.view3d.removeItem(self.vol_item)
            except Exception:
                pass
            self.vol_item = None

        self.vol_item = gl.GLVolumeItem(rgba, smooth=True)
        self.vol_item.setGLOptions('translucent')
        self.view3d.addItem(self.vol_item)

    def export_rgba(self):
        """把最近一次生成的 RGBA 体素导出，供右侧 3D 大图使用。"""
        return self._last_rgba

    # ---------- helpers ----------
    def _normalize_volume(self, vol: np.ndarray) -> np.ndarray:
        L, W = self.volume.window_level, self.volume.window_width
        if L is not None and W is not None:
            lo, hi = float(L) - float(W) * 0.5, float(L) + float(W) * 0.5
        else:
            gmin = self.volume._gmin if self.volume._gmin is not None else float(np.nanmin(vol))
            gmax = self.volume._gmax if self.volume._gmax is not None else float(np.nanmax(vol))
            if not np.isfinite(gmin) or not np.isfinite(gmax) or gmax <= gmin:
                gmin, gmax = 0.0, 1.0
            lo, hi = gmin, gmax
        out = (vol.astype(np.float32) - lo) / max(1e-6, (hi - lo))
        out = np.clip(out, 0.0, 1.0)
        return np.nan_to_num(out, nan=0.0, posinf=1.0, neginf=0.0)

    def _build_rgba_from_current(self) -> np.ndarray:
        """根据 display_mode 构建 (X,Y,Z,4) RGBA 体素。"""
        vol = self.volume._vol
        norm = self._normalize_volume(vol)               # (Z,Y,X) in [0,1]
        mode = self.volume.display_mode
        has_mask = self.volume.has_mask()

        if mode == "image_only" or (mode == "image_masked" and not has_mask):
            rgb = (norm * 255).astype(np.ubyte)
            a   = (norm * 0.6 * 255).astype(np.ubyte)
            rgba = np.stack([rgb, rgb, rgb, a], axis=-1)

        elif mode == "image_masked" and has_mask:
            lab = self.volume._mask
            m = (lab > 0).astype(np.float32)
            rgb = (norm * m * 255).astype(np.ubyte)
            a   = (norm * m * 0.6 * 255).astype(np.ubyte)
            rgba = np.stack([rgb, rgb, rgb, a], axis=-1)

        elif mode == "mask_only" and has_mask:
            lab = np.asarray(self.volume._mask, dtype=np.int32)
            lut = self.volume._lut_colors
            K = lut.shape[0]
            lab_clip = np.clip(lab, 0, K-1)
            rgba = lut[lab_clip].astype(np.ubyte)
            if self.volume.mask_alpha < 1.0:
                a = (rgba[..., 3].astype(np.float32) * self.volume.mask_alpha).clip(0,255).astype(np.ubyte)
                rgba = rgba.copy(); rgba[..., 3] = a

        else:  # overlay
            rgb = (norm * 255).astype(np.ubyte)
            a   = (norm * 0.4 * 255).astype(np.ubyte)
            rgba = np.stack([rgb, rgb, rgb, a], axis=-1)
            if has_mask and self.volume._lut_colors is not None:
                lab = np.asarray(self.volume._mask, dtype=np.int32)
                K = self.volume._lut_colors.shape[0]
                lab_clip = np.clip(lab, 0, K-1)
                color = self.volume._lut_colors[lab_clip]  # (Z,Y,X,4)
                mask_on = (lab_clip > 0)
                rgba = rgba.copy()
                rgba[..., 0][mask_on] = np.maximum(rgba[..., 0][mask_on], color[..., 0][mask_on])
                rgba[..., 1][mask_on] = np.maximum(rgba[..., 1][mask_on], color[..., 1][mask_on])
                rgba[..., 2][mask_on] = np.maximum(rgba[..., 2][mask_on], color[..., 2][mask_on])
                add_a = int(255 * 0.25 * self.volume.mask_alpha)
                rgba[..., 3][mask_on] = np.clip(rgba[..., 3][mask_on].astype(np.int16) + add_a, 0, 255).astype(np.ubyte)

        # (Z,Y,X,4) -> (X,Y,Z,4)
        rgba = np.transpose(rgba, (2, 1, 0, 3)).copy(order='C')
        return rgba


class DetailView(QtWidgets.QFrame):
    """Large right-side view that mirrors and controls the active preview."""
    def __init__(self, volume: VolumeData):
        super().__init__()
        self.setFrameShape(QtWidgets.QFrame.StyledPanel)
        self.setFrameShadow(QtWidgets.QFrame.Raised)
        self.volume = volume
        self.volume.paramsChanged.connect(self._rerender_active)
        self.active_plane = None  # which preview we are mirroring

        # Header
        top = QtWidgets.QHBoxLayout()
        self.title = QtWidgets.QLabel("Detail — (none)")
        self.title.setStyleSheet("font-weight: 600; font-size: 14px;")

        # 新增：放大/缩小按钮
        self.btnZoomIn = QtWidgets.QToolButton()
        self.btnZoomIn.setText("+")                       # 也可用图标
        self.btnZoomIn.setToolTip("Zoom In")
        self.btnZoomIn.clicked.connect(self._zoom_in)

        self.btnZoomOut = QtWidgets.QToolButton()
        self.btnZoomOut.setText("−")                      # 注意是字符 '−' 或者用 '-'
        self.btnZoomOut.setToolTip("Zoom Out")
        self.btnZoomOut.clicked.connect(self._zoom_out)


        self.btnReset = QtWidgets.QToolButton()
        self.btnReset.setText("Reset View")
        self.btnReset.clicked.connect(self._reset_view)


        top.addWidget(self.title)
        top.addStretch(1)
        top.addWidget(self.btnZoomOut)
        top.addWidget(self.btnZoomIn)
        top.addWidget(self.btnReset)


        # Graphics
        self.glw = pg.GraphicsLayoutWidget()
        self.view = self.glw.addViewBox(lockAspect=True, enableMenu=False)
        self.view.setMouseEnabled(x=True, y=True)
        self.img_item = pg.ImageItem()
        self.view.addItem(self.img_item)
        self.img_item.setLevels((0.0, 1.0))



        self.mask_item = pg.ImageItem()
        self.mask_item.setZValue(10)
        self.view.addItem(self.mask_item)
        
        # Layout
        lay = QtWidgets.QVBoxLayout(self)
        lay.setContentsMargins(6, 6, 6, 6)
        lay.addLayout(top)
        lay.addWidget(self.glw)

        # For syncing: a handle to the preview's ViewBox (set when promoted)
        self._linked_preview_vb = None
        # 键盘快捷键：Ctrl/⌘ + 加/减，和 Ctrl/⌘ + 0 重置
        QShortcut(QKeySequence.ZoomIn,  self, activated=self._zoom_in)
        QShortcut(QKeySequence.ZoomOut, self, activated=self._zoom_out)
        QShortcut(QKeySequence("Ctrl+0"), self, activated=self._reset_view)
        QShortcut(QKeySequence("Meta+0"), self, activated=self._reset_view)  # macOS


        # Connect range change to push back to preview
        self.view.sigRangeChanged.connect(self._on_range_changed)



    def _on_range_changed(self, vb, ranges):
        # Mirror the pan/zoom to the linked preview viewbox
        if self._linked_preview_vb is not None:
            x_rng, y_rng = self.view.viewRange()
            # Block signals to avoid feedback loops
            self._linked_preview_vb.blockSignals(True)
            self._linked_preview_vb.setXRange(*x_rng, padding=0)
            self._linked_preview_vb.setYRange(*y_rng, padding=0)
            self._linked_preview_vb.blockSignals(False)

    def _reset_view(self):
        self.view.autoRange()
        if self._linked_preview_vb is not None:
            self._linked_preview_vb.autoRange()

    def _zoom_in(self):
        self._zoom(factor=0.8)   # 数值 <1 表示放大（范围变小）

    def _zoom_out(self):
        self._zoom(factor=1.25)  # 数值 >1 表示缩小（范围变大）

    def _zoom(self, factor: float):
        # 基于当前可视范围按中心缩放，保持纵横比/同步到预览
        x_rng, y_rng = self.view.viewRange()      # [(xmin, xmax), (ymin, ymax)]
        cx = 0.5 * (x_rng[0] + x_rng[1])
        cy = 0.5 * (y_rng[0] + y_rng[1])
        w = (x_rng[1] - x_rng[0]) * factor
        h = (y_rng[1] - y_rng[0]) * factor
        self.view.setXRange(cx - w/2.0, cx + w/2.0, padding=0)
        self.view.setYRange(cy - h/2.0, cy + h/2.0, padding=0)


    def _rerender_active(self):
        if self.active_plane is None:
            return
        idx = self.volume.slices[self.active_plane]

        # # 原图
        # img = self.volume.render_slice(self.active_plane, idx)
        # if img is not None:
        #     self.img_item.setImage(img.T, autoLevels=False, levels=(0.0, 1.0))

        mode = self.volume.display_mode

        if mode == "image_masked":
            img = self.volume.render_image_masked_slice(self.active_plane, idx)
            if img is not None:
                self.img_item.setImage(img.T, autoLevels=False, levels=(0.0, 1.0))
            # 不显示彩色 mask
            self.mask_item.setVisible(False)
            self.img_item.setVisible(True)
            return
        else:
            # 原图（正常）
            img = self.volume.render_slice(self.active_plane, idx)
            if img is not None:
                self.img_item.setImage(img.T, autoLevels=False, levels=(0.0, 1.0))


        # mask
        mode = self.volume.display_mode
        has_mask = self.volume.has_mask()
        if mode == "image_only" or not has_mask:
            self.mask_item.setVisible(False)
            self.img_item.setVisible(True)
            return

        rgba = self.volume.render_mask_rgba_slice(self.active_plane, idx)
        if rgba is None:
            self.mask_item.setVisible(False)
            self.img_item.setVisible(True)
            return

        self.mask_item.setImage(rgba.transpose(1, 0, 2), autoLevels=False)
        self.mask_item.setVisible(True)
        if mode == "mask_only":
            self.img_item.setVisible(False)
        else:
            self.img_item.setVisible(True)


    def promote_from(self, plane: str, src_imgitem: pg.ImageItem, src_viewbox: pg.ViewBox):
        self.active_plane = plane
        self.title.setText(f"Detail — {plane}")
        # 只用统一渲染，保证与预览一致
        self._rerender_active()
        self.view.autoRange()
        # 链接范围
        self._linked_preview_vb = src_viewbox
        x_rng, y_rng = src_viewbox.viewRange()
        self.view.setXRange(*x_rng, padding=0)
        self.view.setYRange(*y_rng, padding=0)


class DetailView3D(QtWidgets.QFrame):
    """右侧 3D 大图：复用与小图相同的 RGBA 体素。"""
    def __init__(self):
        super().__init__()
        self.setFrameShape(QtWidgets.QFrame.StyledPanel)
        self.setFrameShadow(QtWidgets.QFrame.Raised)

        top = QtWidgets.QHBoxLayout()
        self.title = QtWidgets.QLabel("Detail — 3D")
        self.title.setStyleSheet("font-weight: 600; font-size: 14px;")

        # 新增：放大/缩小按钮
        self.btnZoomIn = QtWidgets.QToolButton()
        self.btnZoomIn.setText("+")
        self.btnZoomIn.setToolTip("Zoom In")
        self.btnZoomIn.clicked.connect(self._zoom_in)

        self.btnZoomOut = QtWidgets.QToolButton()
        self.btnZoomOut.setText("−")
        self.btnZoomOut.setToolTip("Zoom Out")
        self.btnZoomOut.clicked.connect(self._zoom_out)

        self.btnReset = QtWidgets.QToolButton()
        self.btnReset.setText("Reset View")
        self.btnReset.clicked.connect(self._reset_view)

        top.addWidget(self.title)
        top.addStretch(1)
        top.addWidget(self.btnZoomOut)
        top.addWidget(self.btnZoomIn)
        top.addWidget(self.btnReset)

        self.view = gl.GLViewWidget()
        self.view.opts["distance"] = 200
        self.view.setBackgroundColor(30, 30, 30)

        self.vol_item = None

        lay = QtWidgets.QVBoxLayout(self)
        lay.setContentsMargins(6, 6, 6, 6)
        lay.addLayout(top)
        lay.addWidget(self.view)

        # 键盘快捷键
        QShortcut(QKeySequence.ZoomIn,  self, activated=self._zoom_in)
        QShortcut(QKeySequence.ZoomOut, self, activated=self._zoom_out)
        QShortcut(QKeySequence("Ctrl+0"), self, activated=self._reset_view)
        QShortcut(QKeySequence("Meta+0"), self, activated=self._reset_view)  # macOS


    def _reset_view(self):
        self.view.opts["distance"] = 200
        self.view.orbit(0, 0)

    def show_rgba(self, rgba: np.ndarray):
        if rgba is None:
            return
        if self.vol_item is not None:
            try:
                self.view.removeItem(self.vol_item)
            except Exception:
                pass
            self.vol_item = None
        self.vol_item = gl.GLVolumeItem(rgba, smooth=True)
        self.vol_item.setGLOptions('translucent')
        self.view.addItem(self.vol_item)
        self._reset_view()
        
    def _reset_view(self):
        # 重置到合适的观察距离和姿态
        self.view.opts["distance"] = 200
        # orbit(azimuthDelta, elevationDelta) —— 归零姿态
        self.view.orbit(0, 0)

    def _zoom(self, factor: float):
        """通过调节相机 distance 实现放大/缩小。factor<1放大，>1缩小"""
        d0 = float(self.view.opts.get("distance", 200))
        # 约束距离范围，避免飞入或飞出过远
        MIN_D, MAX_D = 5.0, 5000.0
        d1 = float(np.clip(d0 * factor, MIN_D, MAX_D))
        self.view.setCameraPosition(distance=d1)   # 官方推荐接口
        # 也可以：self.view.opts["distance"] = d1

    def _zoom_in(self):
        self._zoom(0.8)    # 放大（distance 变小）

    def _zoom_out(self):
        self._zoom(1.25)   # 缩小（distance 变大）



class LeftToolbar(QtWidgets.QFrame):
    """Left-side vertical toolbar with file open and basic controls."""
    openRequested = QtCore.Signal()
    openMaskRequested = QtCore.Signal()


    def __init__(self):
        super().__init__()
        self.setFixedWidth(160)
        self.setFrameShape(QtWidgets.QFrame.StyledPanel)
        self.setFrameShadow(QtWidgets.QFrame.Raised)

        lay = QtWidgets.QVBoxLayout(self)
        lay.setContentsMargins(8, 8, 8, 8)

        title = QtWidgets.QLabel("Tools")
        title.setStyleSheet("font-weight: 700; font-size: 14px;")
        lay.addWidget(title)

        self.btnOpen = QtWidgets.QPushButton("Open Image")
        self.btnOpen.clicked.connect(self.openRequested.emit)
        lay.addWidget(self.btnOpen)

        self.btnOpenMask = QtWidgets.QPushButton("Open Mask")
        self.btnOpenMask.clicked.connect(lambda: self.openMaskRequested.emit())
        lay.addWidget(self.btnOpenMask)

        
        self.spnLevel = QtWidgets.QDoubleSpinBox()
        self.spnLevel.setRange(-2000, 3000); self.spnLevel.setValue(40.0); self.spnLevel.setDecimals(1)
        self.spnWidth = QtWidgets.QDoubleSpinBox()
        self.spnWidth.setRange(1.0, 5000.0); self.spnWidth.setValue(80.0); self.spnWidth.setDecimals(1)


        row1 = QtWidgets.QHBoxLayout()
        row1.addWidget(QtWidgets.QLabel("Level"))
        row1.addWidget(self.spnLevel, 1)
        lay.addLayout(row1)

        row2 = QtWidgets.QHBoxLayout()
        row2.addWidget(QtWidgets.QLabel("Width"))
        row2.addWidget(self.spnWidth, 1)
        lay.addLayout(row2)

        # 透明度
        rowA = QtWidgets.QHBoxLayout()
        rowA.addWidget(QtWidgets.QLabel("Mask α"))
        self.sldMaskAlpha = QtWidgets.QSlider(QtCore.Qt.Horizontal)
        self.sldMaskAlpha.setRange(0, 100)
        self.sldMaskAlpha.setValue(int(0.6 * 100))
        rowA.addWidget(self.sldMaskAlpha, 1)
        lay.addLayout(rowA)

        # 显示模式（单选）
        self.grpMode = QtWidgets.QButtonGroup(self)
        radOverlay  = QtWidgets.QRadioButton("Image+Mask")
        radImage    = QtWidgets.QRadioButton("Image Only")
        radMaskOnly = QtWidgets.QRadioButton("Mask Only")
        radImageMasked = QtWidgets.QRadioButton("Image (masked)")


        radOverlay.setChecked(True)
        self.grpMode.addButton(radOverlay, 0)
        self.grpMode.addButton(radImage, 1)
        self.grpMode.addButton(radMaskOnly, 2)
        self.grpMode.addButton(radImageMasked, 3)
        lay.addWidget(radOverlay)
        lay.addWidget(radImage)
        lay.addWidget(radMaskOnly)
        lay.addWidget(radImageMasked)

        lay.addStretch(1)

class MiddleColumn(QtWidgets.QFrame):
    """Center column: three previews stacked vertically + a single global slice slider + plane selector."""
    # zoomRequested = QtCore.Signal(str)  # plane
    zoomRequested = QtCore.Signal(object) 
    globalSliceChanged = QtCore.Signal(str, int)  # (plane, idx)

    def __init__(self, volume: VolumeData):
        super().__init__()
        self.setFrameShape(QtWidgets.QFrame.StyledPanel)
        self.setFrameShadow(QtWidgets.QFrame.Raised)
        self.volume = volume

        self.view_specs = [
            {"name": "3D",       "factory": lambda: Volume3DPreview(self.volume, "3D")},
            {"name": "Axial",    "factory": lambda: ImagePreview("Axial", self.volume)},
            {"name": "Coronal",  "factory": lambda: ImagePreview("Coronal", self.volume)},
            {"name": "Sagittal", "factory": lambda: ImagePreview("Sagittal", self.volume)},
        ]
        self.viewport_size = len(self.view_specs)  # 4      # 一屏显示几个预览
        self.view_offset = 0        # 当前窗口起始索引


        # 实例化全部预览，但不全都放进可见布局
        self.all_previews = [spec["factory"]() for spec in self.view_specs]
        self.preview3d = None
        for prev in self.all_previews:
            # 透传 🔍
            if hasattr(prev, "zoomRequested"):
                prev.zoomRequested.connect(self.zoomRequested)
            # 找 3D
            if isinstance(prev, Volume3DPreview):
                self.preview3d = prev

        # 把所有 2D 预览的 update3DRequested -> 3D.regenerate
        # for prev in self.all_previews:
        #     if isinstance(prev, ImagePreview):
        #         prev.update3DRequested.connect(lambda: self.preview3d and self.preview3d.regenerate())

        # 可见容器（垂直）
        self.container = QtWidgets.QWidget()
        self.container_vbox = QtWidgets.QVBoxLayout(self.container)
        self.container_vbox.setContentsMargins(6, 6, 6, 6)
        self.container_vbox.setSpacing(6)

        # 先填充可见窗口（前 viewport_size 个）
        self._rebuild_visible_previews()

        # 左侧滚动区域，包裹可见容器
        self.scroll = QtWidgets.QScrollArea()
        self.scroll.setWidgetResizable(True)
        self.scroll.setWidget(self.container)

        # ---- 右侧：竖直全局滑条，用于分页滚动预览列表 ----
        self.sldViewport = QtWidgets.QSlider(QtCore.Qt.Vertical)
        self.sldViewport.setMinimum(0)
        self._update_viewport_slider_range()        # 设置最大值
        self.sldViewport.setValue(self.view_offset)
        self.sldViewport.valueChanged.connect(self._on_viewport_slider)

        # # 顶部：平面选择 + （如果你暂时不需要可移除） + 这里不再作为“slice 全局滑条”
        # self.cmbPlane = QtWidgets.QComboBox()
        # self.cmbPlane.addItems([spec["name"] for spec in self.view_specs])
        # self.cmbPlane.setVisible(False)  # 如不需要可隐藏

        # ---- 中列总布局：左边 scroll（显示预览），右边竖直 slider ----
        col = QtWidgets.QHBoxLayout()
        col.setContentsMargins(0, 0, 0, 0)
        col.setSpacing(6)
        col.addWidget(self.scroll, 1)
        col.addWidget(self.sldViewport)  # 右侧

        lay = QtWidgets.QVBoxLayout(self)
        lay.setContentsMargins(6, 6, 6, 6)
        lay.addLayout(col, 1)

        # 数据改变时，刷新可见预览（而不是重建全部）
        self.volume.dataChanged.connect(self._refresh_visible)


    def _update_viewport_slider_range(self):
        n = len(self.all_previews)
        max_off = max(0, n - self.viewport_size)
        self.sldViewport.setMaximum(max_off)
        self.sldViewport.setEnabled(max_off > 0)
        self.sldViewport.setSingleStep(1)
        self.sldViewport.setPageStep(1)


    def _rebuild_visible_previews(self):
        # 先清空容器
        while self.container_vbox.count():
            item = self.container_vbox.takeAt(0)
            w = item.widget()
            if w is not None:
                w.setParent(None)

        # 取 [view_offset, view_offset + viewport_size) 这一段
        start = int(self.view_offset)
        end = min(start + self.viewport_size, len(self.all_previews))
        for i in range(start, end):
            self.container_vbox.addWidget(self.all_previews[i])
        self.container_vbox.addStretch(1)

    def _on_viewport_slider(self, v: int):
        self.view_offset = int(v)
        self._rebuild_visible_previews()

    def _refresh_visible(self):
        # 数据更新时，刷新“当前可见”的预览
        start = int(self.view_offset)
        end = min(start + self.viewport_size, len(self.all_previews))
        for i in range(start, end):
            self.all_previews[i].refresh()
        # 预览数量若发生变化（未来加 3D 视图），要更新右侧滑条范围
        self._update_viewport_slider_range()



class MainWindow(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Artery GUI — QtCompat Skeleton")
        self.resize(1480, 920)

        # Core data model
        self.volume = VolumeData()

        # Left toolbar
        self.left = LeftToolbar()
        self.left.openRequested.connect(self._open_file)
        
        self.left.spnLevel.valueChanged.connect(
            lambda v: self.volume.set_window(v, self.left.spnWidth.value())
        )
        self.left.spnWidth.valueChanged.connect(
            lambda v: self.volume.set_window(self.left.spnLevel.value(), v)
        )


        self.left.openMaskRequested.connect(self._open_mask)

        # Mask 透明度
        self.left.sldMaskAlpha.valueChanged.connect(
            lambda v: self.volume.set_mask_alpha(v / 100.0)
        )

        # 显示模式

        def _on_mode_changed(id_):
            mode = {
                0: "overlay",
                1: "image_only",
                2: "mask_only",
                3: "image_masked",
            }.get(id_, "overlay")
            self.volume.set_display_mode(mode)


        self.left.grpMode.idClicked.connect(_on_mode_changed)

        # Middle previews
        self.middle = MiddleColumn(self.volume)
        self.middle.zoomRequested.connect(self._promote_preview)

        # Right detail view
        # self.right = DetailView(self.volume)

        # 右侧：2D + 3D 堆叠
        self.right2d = DetailView(self.volume)
        self.right3d = DetailView3D()
        self.rightStack = QtWidgets.QStackedWidget()
        self.rightStack.addWidget(self.right2d)   # index 0
        self.rightStack.addWidget(self.right3d)   # index 1
        self.rightStack.setCurrentIndex(0)




        # Arrange columns using a central widget with a grid
        central = QtWidgets.QWidget()
        grid = QtWidgets.QGridLayout(central)
        grid.setContentsMargins(6, 6, 6, 6)
        grid.setHorizontalSpacing(8)
        grid.setVerticalSpacing(8)

        grid.addWidget(self.left, 0, 0)
        grid.addWidget(self.middle, 0, 1, 1, 1)
        # grid.addWidget(self.right, 0, 2, 1, 1)
        # 摆放时用 rightStack
        grid.addWidget(self.rightStack, 0, 2, 1, 1)
        grid.setColumnStretch(0, 0)
        grid.setColumnStretch(1, 1)
        grid.setColumnStretch(2, 2)

        self.setCentralWidget(central)

        # Status bar
        self.status = self.statusBar()
        self._update_status("Ready")

    # ---------- Slots ----------
    def _update_status(self, text: str):
        self.status.showMessage(text)

    def _open_file(self):
        path, _ = QtWidgets.QFileDialog.getOpenFileName(
            self, "Open Image file", os.getcwd(), "NIfTI (*.nii *.nii.gz)"
        )
        if not path:
            return
        try:
            self.volume.load_nii(path)
            z, y, x = self.volume.shape_zyx()
            self._update_status(f"Loaded: {os.path.basename(path)} — shape (Z,Y,X)=({z},{y},{x})")
        except Exception as e:
            QtWidgets.QMessageBox.critical(self, "Load Error", str(e))
            self._update_status("Load failed")

    def _open_mask(self):
        path, _ = QtWidgets.QFileDialog.getOpenFileName(
            self, "Open Mask NIfTI", os.getcwd(), "NIfTI (*.nii *.nii.gz)"
        )
        if not path:
            return
        try:
            self.volume.load_mask(path)
            self._update_status(f"Loaded mask: {os.path.basename(path)}")
        except Exception as e:
            QtWidgets.QMessageBox.critical(self, "Load Mask Error", str(e))
            self._update_status("Load mask failed")


    # def _promote_preview(self, preview):


    #     if hasattr(preview, "view") and isinstance(preview.view, gl.GLViewWidget):
    #         QtWidgets.QMessageBox.information(self, "3D Promote",
    #             "当前右侧大图为2D视图。\n下一步我可以为你加一个右侧3D大图模式（可缩放旋转）。")
    #         return
    #     self.right.promote_from(preview.plane, preview.img_item, preview.view)

    #     self._update_status(f"Detail view: {preview.plane}")
    def _promote_preview(self, preview):
        # 3D 预览：放大到右侧 3D
        if isinstance(preview, Volume3DPreview):
            rgba = preview.export_rgba()
            if rgba is None:
                QtWidgets.QMessageBox.information(self, "3D",
                    "3D 尚未生成，请先在任一 2D 预览中点击 Update。")
                return
            self.right3d.show_rgba(rgba)
            self.rightStack.setCurrentIndex(1)  # 切到 3D
            self._update_status("Detail view: 3D")
            return

        # 2D：走原有 promote
        self.right2d.promote_from(preview.plane, preview.img_item, preview.view)
        self.rightStack.setCurrentIndex(0)
        self._update_status(f"Detail view: {preview.plane}")

# ------------------------ Entry ------------------------
if __name__ == "__main__":
    # Better performance on high-DPI displays
    QtWidgets.QApplication.setAttribute(QtCore.Qt.AA_EnableHighDpiScaling, True)
    QtWidgets.QApplication.setAttribute(QtCore.Qt.AA_UseHighDpiPixmaps, True)

    app = QtWidgets.QApplication(sys.argv)
    pg.setConfigOptions(imageAxisOrder='row-major')  # ensures ImageItem uses (row, col)

    win = MainWindow()
    win.show()
    sys.exit(app.exec_())
