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

from utils.centerline import (
    CenterlineOptions, compute_centerline,
    save_centerline_mask_nii, save_centerline_yaml
)

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

        self._undo_stack = []          # list[dict] 每个动作的稀疏 diff
        self._redo_stack = []
        self._undo_capacity = 20        
        
        # 画笔当前目标标签（由 UI 设置；0=背景=删除）
        self.brush_label = 1              # 写成哪个标签
        self.apply_only_label = None      # None=All；否则仅当旧值==此标签时才改

        # self.centerline_mask = None      # (Z,Y,X) uint8, 0/1
        # self.centerline_snakes = []      # 用于导出 YAML
        # self.centerline_rois = {}        # 用于导出 YAML
        # self.centerline_display_mode = "off"   # 'off' | 'overlay' | 'only'
        self._centerline = None      # 3D 二值/整型体素，形状 = vol.shape；>0 表示 centerline
        self._cl_color = (255, 80, 80, 255)  # 叠加颜色（可做成可配置）    
        self.show_centerline = True    


    def set_centerline_visible(self, on: bool):
        on = bool(on)
        if on != self.show_centerline:
            self.show_centerline = on
            self.paramsChanged.emit()


    # === centerline 数据入口（体素版）===
    def set_centerline_mask(self, cl_vol: np.ndarray):
        """cl_vol: 与 _vol 同形状的 3D 数组（bool, uint8, int 都可），>0 为 centerline。"""
        if cl_vol is None:
            self._centerline = None
        else:
            arr = np.asarray(cl_vol)
            if self._vol is None or arr.shape != self._vol.shape:
                raise ValueError(f"Centerline shape {arr.shape} must match image shape {self._vol.shape if self._vol is not None else None}")
            self._centerline = (arr > 0).astype(np.uint8)
        self.paramsChanged.emit()

    def has_centerline(self) -> bool:
        return self._centerline is not None

    def get_centerline_slice(self, plane: str, idx: int) -> np.ndarray | None:
        if not self.has_centerline():
            return None
        z, y, x = self._centerline.shape
        if plane == "Axial":
            idx = np.clip(idx, 0, z-1); return self._centerline[idx, :, :]
        if plane == "Coronal":
            idx = np.clip(idx, 0, y-1); return self._centerline[:, idx, :]
        if plane == "Sagittal":
            idx = np.clip(idx, 0, x-1); return self._centerline[:, :, idx]
        return None

    def render_centerline_rgba_slice(self, plane: str, idx: int) -> np.ndarray | None:
        """
        把 centerline 切片渲染成 RGBA(H,W,4)，背景 alpha=0，线条用 self._cl_color。
        注意：返回值**不转置**；前端按你现在的习惯做 transpose。
        """
        sl = self.get_centerline_slice(plane, idx)
        if sl is None:
            return None
        h, w = sl.shape
        rgba = np.zeros((h, w, 4), dtype=np.uint8)
        if np.any(sl):
            r, g, b, a = self._cl_color
            rgba[..., 0][sl > 0] = r
            rgba[..., 1][sl > 0] = g
            rgba[..., 2][sl > 0] = b
            rgba[..., 3][sl > 0] = a
        return rgba

    # （可选）矢量版入口：把一条条 3D 折线投影到各切片，渲染成 2D 位图或直接返回点集。
    # 先占位：等你 YAML/rois 的坐标定义定了，我们一起接。
    def set_centerline_vectors(self, list_of_polylines_zyx):
        """list_of_polylines_zyx: [N_i x 3] 的多段，坐标单位=体素 index (z,y,x)。先占位。"""
        self._centerline_vectors = list_of_polylines_zyx
        self.paramsChanged.emit()



    def set_brush_label(self, lab: int):
        lab = int(max(0, lab))
        if getattr(self, "brush_label", None) != lab:
            self.brush_label = lab
            self.paramsChanged.emit()
            
    def set_apply_only_label(self, lab: int | None):
        """lab 为 None 表示 All；否则仅修改原值==lab 的像素"""
        if lab is not None:
            lab = int(max(0, lab))
        if getattr(self, "apply_only_label", None) != lab:
            self.apply_only_label = lab
            self.paramsChanged.emit()
            
    # ====== 工具：把 view 坐标索引转换封装到调用侧做，这里只改切片 ======
    def apply_brush_disk(self, plane: str, idx: int, cx: int, cy: int, radius: int):
        if not self.has_mask() or not self.is_loaded(): return
        sl = self.get_mask_slice(plane, idx)
        if sl is None: return
        h, w = sl.shape
        r = max(1, int(radius))
        y, x = np.ogrid[:h, :w]
        region = (x - cx)**2 + (y - cy)**2 <= r*r

        # —— 关键：按 ApplyTo 过滤 —— #
        if self.apply_only_label is not None:
            region = region & (sl == self.apply_only_label)

        if not np.any(region): 
            return

        old = sl[region]
        new = np.full(old.shape, self.brush_label, dtype=np.int32)
        changed = (old != new)
        if not np.any(changed): 
            return

        coords = np.argwhere(region)[changed]
        old_vals = old[changed].copy()

        sl[region] = new
        if plane == "Axial":   self._mask[idx, :, :] = sl
        elif plane == "Coronal": self._mask[:, idx, :] = sl
        elif plane == "Sagittal": self._mask[:, :, idx] = sl

        self._push_undo({
            "type": "disk", "plane": plane, "idx": int(idx),
            "coords": coords, "old": old_vals,
            "new": np.full_like(old_vals, self.brush_label, dtype=np.int32),
        })
        self._redo_stack.clear()
        self.paramsChanged.emit()

    # def apply_polygon_fill(self, plane: str, idx: int, poly_rc: np.ndarray):
    #     """
    #     poly_rc: N x 2 的 (row, col) 浮点或整点，表示闭合多边形顶点（最后一个不必等于第一个）。
    #     用 QImage 填充生成二维布尔蒙版，再按 brush_label 改写。
    #     """
    #     if not self.has_mask() or not self.is_loaded():
    #         return
    #     sl = self.get_mask_slice(plane, idx)
    #     if sl is None:
    #         return
    #     h, w = sl.shape

    #     # # 用 QImage/QPainter 填充多边形 -> 二值图
    #     # img = QtGui.QImage(w, h, QtGui.QImage.Format_Grayscale8)
    #     # img.fill(0)
    #     # painter = QtGui.QPainter(img)
    #     # painter.setPen(QtCore.Qt.NoPen)
    #     # painter.setBrush(QtGui.QBrush(QtCore.Qt.white))
    #     # qpoly = QtGui.QPolygonF([QtCore.QPointF(c, r) for (r, c) in poly_rc])
    #     # painter.drawPolygon(qpoly)
    #     # painter.end()

    #     # ptr = img.bits(); ptr.setsize(img.byteCount())
    #     # mask = np.frombuffer(ptr, dtype=np.uint8).reshape((h, w)) > 0

    #     img = QtGui.QImage(w, h, QtGui.QImage.Format_Grayscale8)
    #     img.fill(0)
    #     painter = QtGui.QPainter(img)
    #     painter.setPen(QtCore.Qt.NoPen)
    #     painter.setBrush(QtGui.QBrush(QtCore.Qt.white))
    #     qpoly = QtGui.QPolygonF([QtCore.QPointF(c, r) for (r, c) in poly_rc])
    #     painter.drawPolygon(qpoly)
    #     painter.end()

    #     # ptr = img.bits(); ptr.setsize(img.byteCount())
    #     # stride = img.bytesPerLine()              # 每行实际字节数（含对齐填充）
    #     # buf = np.frombuffer(ptr, dtype=np.uint8)
    #     # arr = buf.reshape((h, stride))[:, :w]    # 裁掉行尾填充
    #     # mask = arr > 0
    #     ptr = img.bits()  # memoryview
    #     # Qt6 有 sizeInBytes()/bytesPerLine()，直接用它们，不需要 setsize
    #     nbytes = int(img.sizeInBytes())
    #     stride = int(img.bytesPerLine())
    #     buf = np.frombuffer(ptr, dtype=np.uint8, count=nbytes)
    #     arr = buf.reshape((h, stride))[:, :w]   # 裁去行尾对齐填充
    #     mask = arr > 0



    #     old = sl[mask]
    #     new = np.full(old.shape, self.brush_label, dtype=np.int32)
    #     changed = (old != new)
    #     if not np.any(changed):
    #         return
    #     coords = np.argwhere(mask)[changed]
    #     old_vals = old[changed].copy()

    #     sl[mask] = new
    #     if plane == "Axial":
    #         self._mask[idx, :, :] = sl
    #     elif plane == "Coronal":
    #         self._mask[:, idx, :] = sl
    #     elif plane == "Sagittal":
    #         self._mask[:, :, idx] = sl

    #     self._push_undo({
    #         "type": "poly",
    #         "plane": plane,
    #         "idx": int(idx),
    #         "coords": coords,
    #         "old": old_vals,
    #         "new": np.full_like(old_vals, self.brush_label, dtype=np.int32),
    #     })
    #     self._redo_stack.clear()
    #     self.paramsChanged.emit()

    def apply_polygon_fill(self, plane: str, idx: int, poly_rc: np.ndarray):
        if not self.has_mask() or not self.is_loaded(): return
        sl = self.get_mask_slice(plane, idx)
        if sl is None: return
        h, w = sl.shape

        img = QtGui.QImage(w, h, QtGui.QImage.Format_Grayscale8)
        img.fill(0)
        p = QtGui.QPainter(img); p.setPen(QtCore.Qt.NoPen); p.setBrush(QtGui.QBrush(QtCore.Qt.white))
        qpoly = QtGui.QPolygonF([QtCore.QPointF(c, r) for (r, c) in poly_rc])
        p.drawPolygon(qpoly); p.end()

        ptr = img.bits()
        stride = img.bytesPerLine()
        buf = np.frombuffer(ptr, dtype=np.uint8, count=h*stride)
        arr = buf.reshape((h, stride))[:, :w]
        region = (arr > 0)

        # —— 关键：按 ApplyTo 过滤 —— #
        if self.apply_only_label is not None:
            region = region & (sl == self.apply_only_label)

        if not np.any(region): 
            return

        old = sl[region]
        new = np.full(old.shape, self.brush_label, dtype=np.int32)
        changed = (old != new)
        if not np.any(changed): 
            return

        coords = np.argwhere(region)[changed]
        old_vals = old[changed].copy()

        sl[region] = new
        if plane == "Axial":   self._mask[idx, :, :] = sl
        elif plane == "Coronal": self._mask[:, idx, :] = sl
        elif plane == "Sagittal": self._mask[:, :, idx] = sl

        self._push_undo({
            "type": "poly", "plane": plane, "idx": int(idx),
            "coords": coords, "old": old_vals,
            "new": np.full_like(old_vals, self.brush_label, dtype=np.int32),
        })
        self._redo_stack.clear()
        self.paramsChanged.emit()

    def _push_undo(self, diff: dict):
        self._undo_stack.append(diff)
        if len(self._undo_stack) > self._undo_capacity:
            self._undo_stack.pop(0)

    def undo(self):
        if not self._undo_stack:
            return False
        diff = self._undo_stack.pop()
        self._apply_diff(diff, use_old=True)
        self._redo_stack.append(diff)
        self.paramsChanged.emit()
        return True

    def redo(self):
        if not self._redo_stack:
            return False
        diff = self._redo_stack.pop()
        self._apply_diff(diff, use_old=False)
        self._undo_stack.append(diff)
        self.paramsChanged.emit()
        return True

    def _apply_diff(self, diff: dict, use_old: bool):
        plane, idx = diff["plane"], diff["idx"]
        sl = self.get_mask_slice(plane, idx)
        if sl is None:
            return
        coords = diff["coords"]
        vals = diff["old"] if use_old else diff["new"]
        sl[coords[:,0], coords[:,1]] = vals
        if plane == "Axial":
            self._mask[idx, :, :] = sl
        elif plane == "Coronal":
            self._mask[:, idx, :] = sl
        elif plane == "Sagittal":
            self._mask[:, :, idx] = sl

    def save_mask(self, path: str):
        if self._mask is None:
            raise ValueError("No mask to save.")
        affine = self._affine if self._affine is not None else np.eye(4)
        img = nib.Nifti1Image(self._mask.astype(np.int32, copy=False), affine)
        nib.save(img, path)



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
        if mode not in ("overlay", "image_only", "mask_only", "image_masked", "centerline_only"):  # + centerline_only
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
        self.btnZoom.clicked.connect(lambda: self.zoomRequested.emit(self))

        title.addWidget(self.lbl)

        # 中间的细长 slider
        title.addWidget(self.hdrSlider, 1)

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

        self.centerline_item = pg.ImageItem()
        self.centerline_item.setZValue(15)
        self.view.addItem(self.centerline_item)
        self.centerline_item.setVisible(False)


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



    def _update_centerline_layer(self, idx: int):
        # 复选框控制
        if not self.volume.show_centerline:
            self.centerline_item.setVisible(False)
            return
        rgba = self.volume.render_centerline_rgba_slice(self.plane, idx)
        if rgba is None or rgba[...,3].max() == 0:
            self.centerline_item.setVisible(False)
            return
        # 统一遵循你现在的显示约定：可视时做转置
        self.centerline_item.setImage(rgba.transpose(1,0,2), autoLevels=False)
        self.centerline_item.setVisible(True)


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
                # self.img_item.setImage(img, autoLevels=False, levels=(0.0, 1.0))

                self.view.autoRange()
            # 该模式下不显示彩色 mask 覆盖
            self.mask_item.setVisible(False)
        elif mode == "centerline_only":
            # 仍然更新底图（用于坐标映射和视域），但隐藏它
            base = self.volume.render_slice(self.plane, idx)
            if base is not None:
                self.img_item.setImage(base.T, autoLevels=False, levels=(0.0, 1.0))
            self.img_item.setVisible(False)
            self.mask_item.setVisible(False)

        else:
            # 正常原图
            img = self.volume.render_slice(self.plane, idx)
            if img is not None:
                self.img_item.setImage(img.T, autoLevels=False, levels=(0.0, 1.0))
                # self.img_item.setImage(img, autoLevels=False, levels=(0.0, 1.0))
                self.view.autoRange()
            # 叠加/仅 mask / 仅图 由这个函数处理
            self._update_mask_layer(idx)
        self._update_centerline_layer(idx)

        
        
        
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
        # self.mask_item.setImage(rgba, autoLevels=False)

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
                # self.img_item.setImage(img, autoLevels=False, levels=(0.0, 1.0))

                self.view.autoRange()
            # 该模式下不显示彩色 mask 覆盖
            self.mask_item.setVisible(False)

        elif mode == "centerline_only":
            # 仍然更新底图（用于坐标映射和视域），但隐藏它
            base = self.volume.render_slice(self.plane, idx)
            if base is not None:
                self.img_item.setImage(base.T, autoLevels=False, levels=(0.0, 1.0))
            self.img_item.setVisible(False)
            self.mask_item.setVisible(False)

        else:
            # 正常原图
            img = self.volume.render_slice(self.plane, idx)
            if img is not None:
                self.img_item.setImage(img.T, autoLevels=False, levels=(0.0, 1.0))
                # self.img_item.setImage(img, autoLevels=False, levels=(0.0, 1.0))

                self.view.autoRange()
            # 叠加/仅 mask / 仅图 由这个函数处理
            self._update_mask_layer(idx)
        self._update_centerline_layer(idx)

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
        has_cl = self.volume.has_centerline()

        if mode == "image_only" or (mode == "image_masked" and not has_mask):
            rgb = (norm * 255).astype(np.ubyte)
            a   = (norm * 0.6 * 255).astype(np.ubyte)
            rgba = np.stack([rgb, rgb, rgb, a], axis=-1)

        elif mode == "centerline_only" and has_cl:
            cl = (self.volume._centerline > 0)
            r, g, b, a = self.volume._cl_color
            R = np.zeros_like(norm, dtype=np.ubyte)
            G = np.zeros_like(norm, dtype=np.ubyte)
            B = np.zeros_like(norm, dtype=np.ubyte)
            A = np.zeros_like(norm, dtype=np.ubyte)
            R[cl] = r; G[cl] = g; B[cl] = b; A[cl] = 255
            rgba = np.stack([R, G, B, A], axis=-1)  # (Z,Y,X,4)

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
        self._tool_mode = 0  # 0=BRUSH, 1=POLY
        

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
        
        # self.img_item = pg.ImageItem()
        # self.view.addItem(self.img_item)
        # self.img_item.setLevels((0.0, 1.0))


        # self.mask_item = pg.ImageItem()
        # self.mask_item.setZValue(10)
        # self.view.addItem(self.mask_item)

        self.img_item  = pg.ImageItem()
        self.mask_item = pg.ImageItem(); self.mask_item.setZValue(10)
        self.centerline_item = pg.ImageItem(); self.centerline_item.setZValue(15)

        self.view.addItem(self.img_item)
        self.view.addItem(self.mask_item)
        self.view.addItem(self.centerline_item)
        self.centerline_item.setVisible(False)



        # 让视图能接收键盘焦点
        self.setFocusPolicy(QtCore.Qt.StrongFocus)
        self.glw.setFocusPolicy(QtCore.Qt.StrongFocus)
        self.view.setFocusPolicy(QtCore.Qt.StrongFocus)

     
                
        # 编辑状态
        self._edit_enabled = False
        self._brush_radius = 8
        self._poly_points = []        # [(x,y) in image coords]
        self._poly_curve = pg.PlotDataItem(pen=pg.mkPen((0, 200, 255), width=2))
        self.view.addItem(self._poly_curve)
        self._poly_curve.setZValue(20)
        self._poly_curve.setVisible(False)

        # 显示笔刷位置的小圈圈
        self._brush_cursor = pg.ScatterPlotItem(size=0, pen=pg.mkPen('y'), brush=pg.mkBrush(0,0,0,0))
        self._brush_cursor.setZValue(15)
        self.view.addItem(self._brush_cursor)
                
        
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
        # 关闭/提交多边形的快捷键兜底
        QShortcut(QKeySequence(QtCore.Qt.Key_Return), self, activated=self._commit_polygon)
        QShortcut(QKeySequence(QtCore.Qt.Key_Enter),  self, activated=self._commit_polygon)
        QShortcut(QKeySequence(QtCore.Qt.Key_Escape), self, activated=self._cancel_polygon)


        # Connect range change to push back to preview
        self.view.sigRangeChanged.connect(self._on_range_changed)
        # 捕获场景事件
        # self.view.scene().installEventFilter(self) # TODO
        self.glw.scene().installEventFilter(self)



    def _update_centerline_layer(self, idx: int):
        if not self.volume.show_centerline:
            self.centerline_item.setVisible(False)
            return
        # rgba = self.volume.render_centerline_rgba_slice(self.active_plane, idx)
        
        rgba = self.volume.render_centerline_rgba_slice(self.active_plane, idx)
        if rgba is None or rgba[...,3].max() == 0:
            self.centerline_item.setVisible(False)
            return
        self.centerline_item.setImage(rgba.transpose(1,0,2), autoLevels=False)
        self.centerline_item.setVisible(True)

    def set_tool_mode(self, mode: int):
        self._tool_mode = 0 if int(mode) == 0 else 1
        # 切到多边形时，清掉画刷光标；切回刷子时，清多边形
        if self._tool_mode == 1:
            self._brush_cursor.setData([])
        else:
            self._poly_points.clear()
            self._poly_curve.setVisible(False)

    def _cancel_polygon(self):
        self._poly_points.clear()
        self._poly_curve.setVisible(False)


    def eventFilter(self, obj, ev):
        if self._edit_enabled and self.active_plane is not None:
            et = ev.type()

            if et == QtCore.QEvent.GraphicsSceneMouseMove:
                if self._tool_mode == 0:  # BRUSH
                    pos = ev.scenePos()
                    self._update_brush_cursor(pos)
                    if ev.buttons() & QtCore.Qt.LeftButton:
                        self._paint_at(pos)
                return False

            elif et == QtCore.QEvent.GraphicsSceneMousePress and ev.button() == QtCore.Qt.LeftButton:
                if self._tool_mode == 1:  # POLY
                    self._append_poly_point(ev.scenePos())
                else:                      # BRUSH
                    self._paint_at(ev.scenePos())
                return True

            elif et == QtCore.QEvent.KeyPress:
                if ev.key() in (QtCore.Qt.Key_Return, QtCore.Qt.Key_Enter):
                    if self._tool_mode == 1:   # 只有多边形模式时提交
                        self._commit_polygon()
                    return True
                elif ev.key() == QtCore.Qt.Key_Escape:
                    self._cancel_polygon()
                    return True

            elif et == QtCore.QEvent.GraphicsSceneMouseDoubleClick:
                if self._tool_mode == 1 and self._poly_points:
                    self._commit_polygon()
                    return True

        return super().eventFilter(obj, ev)



    def _scene_to_image_rc(self, scenePos):
        p = self.img_item.mapFromScene(scenePos)
        # 因为 setImage(img.T)，显示坐标 x 对应 row，y 对应 col
        r = int(round(p.x()))
        c = int(round(p.y()))
        return r, c


    def _paint_at(self, scenePos):
        if self.active_plane is None:
            return
        r, c = self._scene_to_image_rc(scenePos)
        idx = self.volume.slices[self.active_plane]
        # 边界保护
        z,y,x = self.volume.shape_zyx()
        h,w = None,None
        sl = self.volume.get_slice(self.active_plane, idx)
        if sl is None: return
        h,w = sl.shape
        if not (0 <= r < h and 0 <= c < w):
            return
        self.volume.apply_brush_disk(self.active_plane, idx, c, r, self._brush_radius)  # 注意顺序：cx,cy
        # 刷新当前帧
        self._rerender_active()

    def _append_poly_point(self, scenePos):
        # 以图像坐标收集点
        r, c = self._scene_to_image_rc(scenePos)
        sl = self.volume.get_slice(self.active_plane, self.volume.slices[self.active_plane])
        if sl is None: return
        h,w = sl.shape
        if 0 <= r < h and 0 <= c < w:
            self._poly_points.append((r,c))

            xs = [p[0] for p in self._poly_points]  # row -> x
            ys = [p[1] for p in self._poly_points]  # col -> y

            self._poly_curve.setData(xs, ys)
            self._poly_curve.setVisible(True)

    def _commit_polygon(self):
        if not self._poly_points or self.active_plane is None:
            return
        idx = self.volume.slices[self.active_plane]
        poly = np.array(self._poly_points, dtype=np.float32)
        self.volume.apply_polygon_fill(self.active_plane, idx, poly)
        self._poly_points.clear()
        self._poly_curve.setVisible(False)
        self._rerender_active()

    # def _update_brush_cursor(self, scenePos):
    #     # 仅显示一个环（用 ScatterPlotItem 的 size 模拟）
    #     r, c = self._scene_to_image_rc(scenePos)
    #     self._brush_cursor.setData([{'pos': (c, r), 'size': self._brush_radius*2, 'brush': None, 'pen': pg.mkPen('y', width=1)}])

    def _update_brush_cursor(self, scenePos):
        r, c = self._scene_to_image_rc(scenePos)
        # 画点时，scene里的 (x,y) = (row, col)
        self._brush_cursor.setData([{
            'pos': (r, c),
            'size': self._brush_radius*2,
            'brush': None,
            'pen': pg.mkPen('y', width=1)
        }])




    def set_edit_enabled(self, on: bool):
        self._edit_enabled = bool(on)
        self._poly_points.clear()
        self._poly_curve.setVisible(False)
        self._brush_cursor.setData([])
        if self._edit_enabled:
            self.view.setFocus()   # 让回车直接进来



    def set_brush_radius(self, r: int):
        self._brush_radius = max(1, int(r))


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
                # self.img_item.setImage(img, autoLevels=False, levels=(0.0, 1.0))

            # 不显示彩色 mask
            self.mask_item.setVisible(False)
            self.img_item.setVisible(True)
            self._update_centerline_layer(idx)
            return
        
        elif mode == "centerline_only":
            base = self.volume.render_slice(self.active_plane, idx)
            if base is not None:
                self.img_item.setImage(base.T, autoLevels=False, levels=(0.0, 1.0))
            # 仅显示中心线
            self.img_item.setVisible(False)
            self.mask_item.setVisible(False)
            self._update_centerline_layer(idx)
            return

        else:
            # 原图（正常）
            img = self.volume.render_slice(self.active_plane, idx)
            if img is not None:
                self.img_item.setImage(img.T, autoLevels=False, levels=(0.0, 1.0))
                # self.img_item.setImage(img, autoLevels=False, levels=(0.0, 1.0))


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
        # self.mask_item.setImage(rgba, autoLevels=False)

        self.mask_item.setVisible(True)
        if mode == "mask_only":
            self.img_item.setVisible(False)
        else:
            self.img_item.setVisible(True)
            
        self._update_centerline_layer(idx)

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

    editModeToggled = QtCore.Signal(bool)   # 开/关编辑
    brushSizeChanged = QtCore.Signal(int)   # 半径像素
    brushLabelChanged = QtCore.Signal(int)  # 作用标签（颜色）
    saveMaskRequested = QtCore.Signal()
    undoRequested = QtCore.Signal()
    redoRequested = QtCore.Signal()
    toolModeChanged = QtCore.Signal(int)
    applyFilterChanged = QtCore.Signal(object)   # 传 None 或 int
    
    saveCenterlineYamlRequested = QtCore.Signal()
    saveCenterlineNiftiRequested = QtCore.Signal()

    # --- Centerline 相关 ---
    computeCenterlineRequested = QtCore.Signal(str)   # method
    centerlineVisibilityChanged = QtCore.Signal(bool) # show/hide

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

        # --- 编辑工具 ---
        sep = QtWidgets.QFrame(); sep.setFrameShape(QtWidgets.QFrame.HLine)
        lay.addWidget(sep)

        self.chkEdit = QtWidgets.QCheckBox("Edit Mask")
        self.chkEdit.toggled.connect(self.editModeToggled.emit)
        lay.addWidget(self.chkEdit)

        # 工具模式
        self.grpTool = QtWidgets.QButtonGroup(self)
        radBrush = QtWidgets.QRadioButton("Brush")
        radPoly  = QtWidgets.QRadioButton("Polygon")
        radBrush.setChecked(True)
        self.grpTool.addButton(radBrush, 0)   # 0=BRUSH
        self.grpTool.addButton(radPoly, 1)    # 1=POLY
        lay.addWidget(radBrush)
        lay.addWidget(radPoly)

        # self.toolModeChanged = QtCore.Signal(int)  # <- 类属性最上面声明
        # self.grpTool.idClicked.connect(self.toolModeChanged.emit)
        self.grpTool.idClicked.connect(lambda i: self.toolModeChanged.emit(int(i)))

        
        # 笔刷半径
        rowB = QtWidgets.QHBoxLayout()
        rowB.addWidget(QtWidgets.QLabel("Brush R"))
        self.sldBrush = QtWidgets.QSlider(QtCore.Qt.Horizontal)
        self.sldBrush.setRange(1, 50); self.sldBrush.setValue(8)
        self.sldBrush.valueChanged.connect(self.brushSizeChanged.emit)
        rowB.addWidget(self.sldBrush, 1)
        lay.addLayout(rowB)

        # 两行颜色选择：1) 笔刷显示色（仅指示），2) 作用标签（决定加/删）
        # 用下拉列出 “Blank(0), Label 1, Label 2, ...”
        self.cmbBrushColor = QtWidgets.QComboBox()
        self.cmbApplyLabel = QtWidgets.QComboBox()
        
        lay.addWidget(QtWidgets.QLabel("Brush Label"))   # 原 Brush Color
        lay.addWidget(self.cmbBrushColor)
        lay.addWidget(QtWidgets.QLabel("Apply To"))
        lay.addWidget(self.cmbApplyLabel)

        self.cmbBrushColor.currentIndexChanged.connect(
            lambda: self.brushLabelChanged.emit(self.cmbBrushColor.currentData())
        )
        self.cmbApplyLabel.currentIndexChanged.connect(
            lambda: self.applyFilterChanged.emit(self.cmbApplyLabel.currentData())
        )


        # 撤销/重做/保存
        btnRow = QtWidgets.QHBoxLayout()
        self.btnUndo = QtWidgets.QPushButton("Undo")
        self.btnRedo = QtWidgets.QPushButton("Redo")
        self.btnSave = QtWidgets.QPushButton("Save Mask")
        self.btnUndo.clicked.connect(self.undoRequested.emit)
        self.btnRedo.clicked.connect(self.redoRequested.emit)
        self.btnSave.clicked.connect(self.saveMaskRequested.emit)
        btnRow.addWidget(self.btnUndo); btnRow.addWidget(self.btnRedo); btnRow.addWidget(self.btnSave)
        lay.addLayout(btnRow)

        
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
        radCenterlineOnly = QtWidgets.QRadioButton("Centerline Only")

        radOverlay.setChecked(True)
        self.grpMode.addButton(radOverlay, 0)
        self.grpMode.addButton(radImage, 1)
        self.grpMode.addButton(radMaskOnly, 2)
        self.grpMode.addButton(radImageMasked, 3)
        self.grpMode.addButton(radCenterlineOnly, 4)      # 新增 id=4

        lay.addWidget(radOverlay)
        lay.addWidget(radImage)
        lay.addWidget(radMaskOnly)
        lay.addWidget(radImageMasked)
        lay.addWidget(radCenterlineOnly)

        # ---- Centerline ----
        sep2 = QtWidgets.QFrame(); sep2.setFrameShape(QtWidgets.QFrame.HLine)
        lay.addWidget(sep2)

        clTitle = QtWidgets.QLabel("Centerline"); clTitle.setStyleSheet("font-weight: 700;")
        lay.addWidget(clTitle)

        self.cmbCLMethod = QtWidgets.QComboBox()
        self.cmbCLMethod.addItem("Baseline (3D skeletonize)", userData="baseline")
        self.cmbCLMethod.addItem("VMTK (vmtkcenterlines)",     userData="vmtk")
        lay.addWidget(self.cmbCLMethod)

        self.btnComputeCL = QtWidgets.QPushButton("Compute Centerline")
        lay.addWidget(self.btnComputeCL)

        self.chkShowCL = QtWidgets.QCheckBox("Show Centerline")
        self.chkShowCL.setChecked(True)
        lay.addWidget(self.chkShowCL)

        # 连接信号
        self.btnComputeCL.clicked.connect(
            lambda: self.computeCenterlineRequested.emit(self.cmbCLMethod.currentData())
        )
        self.chkShowCL.toggled.connect(self.centerlineVisibilityChanged.emit)

        rowSave = QtWidgets.QHBoxLayout()
        self.btnSaveCLYaml = QtWidgets.QPushButton("Save CL (YAML)")
        self.btnSaveCLNii  = QtWidgets.QPushButton("Save CL (NIfTI)")
        rowSave.addWidget(self.btnSaveCLYaml)
        rowSave.addWidget(self.btnSaveCLNii)
        lay.addLayout(rowSave)

        # 连接信号
        self.btnSaveCLYaml.clicked.connect(self.saveCenterlineYamlRequested.emit)
        self.btnSaveCLNii.clicked.connect(self.saveCenterlineNiftiRequested.emit)


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

        # Middle previews
        self.middle = MiddleColumn(self.volume)
        
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

        # ---------- 3) 信号连接（统一放在这里，顺序清晰） ----------

        self.left.openRequested.connect(self._open_file)
        self.left.openMaskRequested.connect(self._open_mask)

        # WL/WW 
        self.left.spnLevel.valueChanged.connect(
            lambda v: self.volume.set_window(v, self.left.spnWidth.value())
        )
        self.left.spnWidth.valueChanged.connect(
            lambda v: self.volume.set_window(self.left.spnLevel.value(), v)
        )

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
                4: "centerline_only", 
            }.get(id_, "overlay")
            self.volume.set_display_mode(mode)
        self.left.grpMode.idClicked.connect(_on_mode_changed)


        # —— 左侧编辑工具联动 —— 
        # 1) 开关编辑
        self.left.editModeToggled.connect(self._on_edit_toggled)
        # 2) 画笔半径（现在 right2d 已存在，连接是安全的）
        self.left.brushSizeChanged.connect(lambda r: self.right2d.set_brush_radius(r))
        # 3) Brush = 写成哪个标签
        self.left.brushLabelChanged.connect(self.volume.set_brush_label)
        # self.left.brushLabelChanged.connect(self._on_brush_label_changed)
        # 4) ApplyTo = 只作用于哪个旧标签（None=All）
        self.left.applyFilterChanged.connect(self.volume.set_apply_only_label)
        # 5) 工具模式（Brush / Polygon 等）
        self.left.toolModeChanged.connect(self.right2d.set_tool_mode)
        # 6) 撤销/重做/保存
        self.left.undoRequested.connect(lambda: self.volume.undo())
        self.left.redoRequested.connect(lambda: self.volume.redo())
        self.left.saveMaskRequested.connect(self._on_save_mask)
        # 中列预览 -> 右侧放大
        self.middle.zoomRequested.connect(self._promote_preview)
        # Centerline：触发与显示
        self.left.computeCenterlineRequested.connect(self._on_compute_centerline)
        self.left.centerlineVisibilityChanged.connect(self.volume.set_centerline_visible)
        # saving
        self.left.saveCenterlineYamlRequested.connect(self._on_save_centerline_yaml)
        self.left.saveCenterlineNiftiRequested.connect(self._on_save_centerline_nii)

        # 初始把“作用标签”下拉填充（在加载 mask 后会再次刷新）
        self._refresh_label_combos()


    def _on_save_centerline_yaml(self):
        if not self.volume.has_centerline():
            QtWidgets.QMessageBox.information(self, "Centerline", "没有可保存的 centerline，请先计算。")
            return
        path, _ = QtWidgets.QFileDialog.getSaveFileName(
            self, "Save Centerline (YAML)", os.getcwd(), "YAML (*.yaml *.yml)"
        )
        if not path:
            return
        try:
            # 你 utils.centerline.save_centerline_yaml 的签名若不同，请在这里适配。
            # 最常见：save_centerline_yaml(cl_mask, affine, yaml_path, radius_vox=2)
            save_centerline_yaml(self.volume._centerline, self.volume._affine, path)
            self._update_status(f"Saved centerline YAML: {os.path.basename(path)}")
        except Exception as e:
            QtWidgets.QMessageBox.critical(self, "Save YAML Error", str(e))

    def _on_save_centerline_nii(self):
        if not self.volume.has_centerline():
            QtWidgets.QMessageBox.information(self, "Centerline", "没有可保存的 centerline，请先计算。")
            return
        path, _ = QtWidgets.QFileDialog.getSaveFileName(
            self, "Save Centerline (NIfTI)", os.getcwd(), "NIfTI (*.nii *.nii.gz)"
        )
        if not path:
            return
        try:
            save_centerline_mask_nii(self.volume._centerline, self.volume._affine, path)
            self._update_status(f"Saved centerline NIfTI: {os.path.basename(path)}")
        except Exception as e:
            QtWidgets.QMessageBox.critical(self, "Save NIfTI Error", str(e))



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
            self._refresh_label_combos() 
            self._update_status(f"Loaded mask: {os.path.basename(path)}")
        except Exception as e:
            QtWidgets.QMessageBox.critical(self, "Load Mask Error", str(e))
            self._update_status("Load mask failed")


    def _on_compute_centerline(self, method: str):
        if not self.volume.has_mask():
            QtWidgets.QMessageBox.information(self, "Centerline", "请先加载/生成 mask。")
            return
        try:
            from utils.centerline import extract_centerline   # 你新建的单文件模块
        except Exception as e:
            QtWidgets.QMessageBox.critical(self, "Centerline", f"导入 centerline 模块失败：{e}")
            return
        try:
            cl = extract_centerline(self.volume._mask, method=method)
            self.volume.set_centerline_mask(cl)
            self._update_status(f"Centerline computed by {method}.")
        except Exception as e:
            QtWidgets.QMessageBox.critical(self, "Centerline", f"计算失败：{e}")


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


    def _on_edit_toggled(self, on: bool):
        self.right2d.set_edit_enabled(on)

    def _on_brush_label_changed(self, lab: int):
        self.volume.set_brush_label(lab)

    def _on_save_mask(self):
        path, _ = QtWidgets.QFileDialog.getSaveFileName(
            self, "Save Mask as NIfTI", os.getcwd(), "NIfTI (*.nii *.nii.gz)"
        )
        if not path:
            return
        try:
            self.volume.save_mask(path)
            self._update_status(f"Saved mask: {os.path.basename(path)}")
        except Exception as e:
            QtWidgets.QMessageBox.critical(self, "Save Mask Error", str(e))


    def _refresh_label_combos(self):
        lut = self.volume._lut_colors
        self.left.cmbBrushColor.clear()
        self.left.cmbApplyLabel.clear()

        # Brush：列出 0..K-1（0=Blank 擦除）
        if lut is None:
            items = [(0, (0,0,0,0)), (1, (255,0,0,255))]
        else:
            items = [(i, tuple(lut[i])) for i in range(lut.shape[0])]

        for i, rgba in items:
            pix = QtGui.QPixmap(16,16); pix.fill(QtGui.QColor(*rgba))
            icon = QtGui.QIcon(pix)
            text = "Blank (0)" if i==0 else f"Label {i}"
            self.left.cmbBrushColor.addItem(icon, text, userData=i)

        # ApplyTo：首项 All labels（userData=None），之后 0..K-1
        self.left.cmbApplyLabel.addItem("All labels", userData=None)
        for i, rgba in items:
            pix = QtGui.QPixmap(16,16); pix.fill(QtGui.QColor(*rgba))
            icon = QtGui.QIcon(pix)
            text = "Blank (0)" if i==0 else f"Label {i}"
            self.left.cmbApplyLabel.addItem(icon, text, userData=i)

        # 默认：Brush=1，ApplyTo=All
        idx1 = self.left.cmbBrushColor.findData(1)
        if idx1 >= 0: 
            self.left.cmbBrushColor.setCurrentIndex(idx1)
            self.volume.set_brush_label(1)

        self.left.cmbApplyLabel.setCurrentIndex(0)   # All labels
        self.volume.set_apply_only_label(None)


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
