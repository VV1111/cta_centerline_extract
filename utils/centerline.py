# centerline.py
# -*- coding: utf-8 -*-
from __future__ import annotations
import uuid
import math
from dataclasses import dataclass, field
from typing import List, Tuple, Dict, Optional, Iterable

import numpy as np
import nibabel as nib

# ---- 可选依赖（VMTK、scikit-image、scipy） ----
_has_vmtk = False
try:
    # vmtk 常见入口（有的环境是 vmtk, 有的是 vmtk.vmtkscripts）
    from vmtk import vmtkscripts  # type: ignore
    _has_vmtk = True
except Exception:
    _has_vmtk = False

_has_ski = False
try:
    from skimage.morphology import skeletonize_3d  # type: ignore
    _has_ski = True
except Exception:
    _has_ski = False

_has_scipy = False
try:
    from scipy.ndimage import distance_transform_edt  # type: ignore
    _has_scipy = True
except Exception:
    _has_scipy = False


# ------------------ 数据结构 ------------------

@dataclass
class CenterlineOptions:
    """统一的中心线参数；method='vmtk'优先，其次'baseline'。"""
    method: str = "vmtk"                    # 'vmtk' | 'baseline'
    use_labels: Optional[Iterable[int]] = None  # None=label>0；否则只用这些标签
    prune_length_mm: float = 4.0            # Baseline：端枝剪枝长度
    resample_step_mm: float = 1.0           # snakes 采样间距
    output_coord_space: str = "world"       # 'world' | 'voxel'
    centerline_label_value: int = 99        # 可把中心线写作一个特定类别（保存NIfTI时用）


@dataclass
class Snake:
    id: str
    points: List[Tuple[float, float, float]]      # world-mm 或 voxel，根据 options
    radius_mm: List[float]                        # 与 points 对齐
    src_label: Optional[int] = None               # 原始血管标签（多类时可用）


@dataclass
class ROI:
    id: str
    position: Tuple[float, float, float]          # world-mm 或 voxel
    snake_parents: List[str]
    radius_mm: float
    type: str                                     # 'endpoint' | 'bifurcation'


@dataclass
class CenterlineResult:
    centerline_mask: np.ndarray                   # (Z,Y,X) uint8 or int
    snakes: List[Snake] = field(default_factory=list)
    rois: Dict[str, ROI] = field(default_factory=dict)


# ------------------ 顶层入口 ------------------

def compute_centerline(
    mask_zyx: np.ndarray,
    affine: np.ndarray,
    spacing_zyx: Tuple[float, float, float],
    options: CenterlineOptions = CenterlineOptions(),
) -> CenterlineResult:
    """
    统一入口。所有计算都在体素顺序 (Z,Y,X) 上进行。
    显示层是否转置.T 完全不影响本函数。
    """
    if mask_zyx.ndim != 3:
        raise ValueError("mask must be 3D array (Z,Y,X)")

    # 选取有效体素（兼容二值/多类）
    if options.use_labels is None:
        work = (mask_zyx.astype(np.int32) > 0)
    else:
        labs = np.array(list(options.use_labels), dtype=np.int32)
        work = np.isin(mask_zyx.astype(np.int32), labs)

    if not np.any(work):
        raise ValueError("No foreground voxels to extract centerline from.")

    m = options.method.lower()
    if m == "vmtk":
        if not _has_vmtk:
            # 回落
            return _compute_centerline_baseline(work, affine, spacing_zyx, options)
        return _compute_centerline_vmtk(work, affine, spacing_zyx, options)
    elif m == "baseline":
        return _compute_centerline_baseline(work, affine, spacing_zyx, options)
    else:
        raise ValueError(f"Unknown method: {options.method}")


# ------------------ VMTK 实现（简版骨架） ------------------

def _compute_centerline_vmtk(
    bin_mask_zyx: np.ndarray,
    affine: np.ndarray,
    spacing_zyx: Tuple[float, float, float],
    options: CenterlineOptions,
) -> CenterlineResult:
    """
    用 VMTK：mask→表面→中心线；然后落回体素 + 生成 snakes/rois。
    说明：VMTK 最佳入口是“表面网格”，这里给出一条常用低耦合流程。
    """
    # 1) 把 mask 存成 NIfTI（临时）并转 VTK Image，随后跑 vmtk
    # 为避免大量磁盘IO，这里走内存最小化流程：先保存 NIfTI（简单可靠）。
    import tempfile, os
    tmpdir = tempfile.mkdtemp()
    niip = os.path.join(tmpdir, "mask.nii.gz")
    nib.save(nib.Nifti1Image(bin_mask_zyx.astype(np.uint8), affine), niip)

    # 2) vmtkimagevolumeviewer/vmtkmarchingcubes → 表面
    # 直接用 vmtkmarchingcubes 从阈值生成表面
    mc = vmtkscripts.vmtkMarchingCubes()
    mc.InputImageName = niip
    mc.OtsuThreshold = 0                # 不用 Otsu；我们已经是二值
    mc.Level = 0.5                      # 二值边界
    mc.Execute()
    surface = mc.Surface

    # 3) 平滑/清理（可选）
    clean = vmtkscripts.vmtkSurfaceCleaner()
    clean.Surface = surface
    clean.Execute()
    surface = clean.Surface

    # 4) 中心线（需要 inlet/outlet；若不指定，vmtk 可自动找端点）
    cl = vmtkscripts.vmtkCenterlines()
    cl.Surface = surface
    cl.AppendEndPoints = 1
    cl.SeedSelectorName = 'openprofiles'  # 自动开口作为端点
    cl.Execute()
    centerlines = cl.Centerlines

    # 5) 计算半径属性
    attr = vmtkscripts.vmtkCenterlineAttributes()
    attr.Centerlines = centerlines
    attr.Execute()
    centerlines = attr.Centerlines

    # 6) 将中心线采样到体素掩膜（粗化：把线段上 mm 坐标落到体素并置1）
    # 同时构建 snakes/rois（用 PolyData 拿出每条 cell 的点）
    from vtk import vtkPolyData, vtkIdList
    import vtk
    pd: vtkPolyData = centerlines
    cl_mask = np.zeros_like(bin_mask_zyx, dtype=np.uint8)

    def world_to_voxel(xyz_mm):
        # 逆变换：voxel = inv(affine) @ [x,y,z,1]
        v = np.linalg.inv(affine) @ np.array([xyz_mm[0], xyz_mm[1], xyz_mm[2], 1.0])
        return (v[2], v[1], v[0])  # 返回 (Z,Y,X) 顺序的体素坐标（浮点）

    snakes: List[Snake] = []
    rois: Dict[str, ROI] = {}

    n_cells = pd.GetNumberOfCells()
    for ci in range(n_cells):
        cell = pd.GetCell(ci)
        pts = cell.GetPoints()
        npts = pts.GetNumberOfPoints()
        if npts < 2:
            continue
        sid = str(uuid.uuid4())
        pts_out: List[Tuple[float, float, float]] = []
        r_out: List[float] = []

        for i in range(npts):
            x, y, z = pts.GetPoint(i)  # 世界坐标（mm）
            # 半径属性常叫 "MaximumInscribedSphereRadius"
            r = pd.GetPointData().GetArray("MaximumInscribedSphereRadius").GetTuple1(cell.GetPointId(i)) \
                if pd.GetPointData().GetArray("MaximumInscribedSphereRadius") else 0.0

            if options.output_coord_space == "world":
                pts_out.append((x, y, z))
            else:
                vz, vy, vx = world_to_voxel((x, y, z))
                pts_out.append((vx, vy, vz))
            r_out.append(float(r))

            # 落回体素 mask
            vz, vy, vx = world_to_voxel((x, y, z))
            iz, iy, ix = int(round(vz)), int(round(vy)), int(round(vx))
            if 0 <= iz < cl_mask.shape[0] and 0 <= iy < cl_mask.shape[1] and 0 <= ix < cl_mask.shape[2]:
                cl_mask[iz, iy, ix] = 1

        snakes.append(Snake(id=sid, points=pts_out, radius_mm=r_out, src_label=None))

    # 7) ROI（端点/分叉）。VMTK 的 centerlines 有拓扑信息，这里简化：每段两端作为 endpoint。
    for s in snakes:
        if not s.points:
            continue
        for k, role in [(0, 'endpoint'), (-1, 'endpoint')]:
            rid = str(uuid.uuid4())
            p = s.points[k]
            r = s.radius_mm[k] if s.radius_mm else 0.0
            rois[rid] = ROI(id=rid, position=p, snake_parents=[s.id], radius_mm=float(r), type=role)

    return CenterlineResult(centerline_mask=cl_mask, snakes=snakes, rois=rois)


# ------------------ Baseline 实现 ------------------

def _compute_centerline_baseline(
    bin_mask_zyx: np.ndarray,
    affine: np.ndarray,
    spacing_zyx: Tuple[float, float, float],
    options: CenterlineOptions,
) -> CenterlineResult:
    if not _has_ski or not _has_scipy:
        raise RuntimeError(
            "Baseline method requires scikit-image and scipy. "
            "Please install: pip install scikit-image scipy"
        )

    sz, sy, sx = spacing_zyx

    # 1) EDT（毫米）
    D = distance_transform_edt(bin_mask_zyx, sampling=(sz, sy, sx)).astype(np.float32)

    # 2) 3D 骨架
    skel = skeletonize_3d(bin_mask_zyx).astype(np.uint8)

    # 3) 剪枝
    Lmin_vox = max(1, int(round(options.prune_length_mm / max(1e-6, (sx + sy + sz) / 3.0))))
    skel = _prune_skeleton_simple(skel, Lmin_vox)

    # 4) 图与分段
    nodes, edges = _skeleton_to_graph(skel)  # nodes: dict[id] -> (z,y,x), edges: list[list[(z,y,x)]]
    cl_mask = (skel > 0).astype(np.uint8)

    # 5) 生成 snakes（按弧长采样），半径来自 D
    snakes: List[Snake] = []
    for epts in edges:
        if len(epts) < 2:
            continue
        sid = str(uuid.uuid4())
        pts_samp, rad_samp = _resample_edge_voxels(epts, D, spacing_zyx, step_mm=options.resample_step_mm)

        if options.output_coord_space == "world":
            pts_out = [_vox_to_world_xyz(p, affine) for p in pts_samp]  # (x,y,z)mm
        else:
            pts_out = [(float(x), float(y), float(z)) for (z, y, x) in pts_samp]  # 注意顺序

        snakes.append(Snake(id=sid, points=pts_out, radius_mm=[float(r) for r in rad_samp]))

    # 6) ROIs（端点/分叉）
    rois: Dict[str, ROI] = {}
    for n_id, (nz, ny, nx) in nodes.items():
        p_vox = (nz, ny, nx)
        r = float(D[p_vox]) if 0 <= nz < D.shape[0] and 0 <= ny < D.shape[1] and 0 <= nx < D.shape[2] else 0.0
        p_out = _vox_to_world_xyz(p_vox, affine) if options.output_coord_space == "world" else (float(nx), float(ny), float(nz))
        degree = len(n_id) if isinstance(n_id, (list, tuple)) else 0  # 占位，无需真实度（后面填正确）
        # 根据邻接边数判断端/叉
        # 计算真实度：
        deg = _node_degree_from_edges(p_vox, edges)
        rtype = 'endpoint' if deg == 1 else ('bifurcation' if deg >= 3 else 'junction')
        rid = str(uuid.uuid4())
        # 关联到经过该点的 snake
        sp = _incident_snake_ids_for_vox(p_vox, edges, snakes, options, affine)
        rois[rid] = ROI(id=rid, position=p_out, snake_parents=sp, radius_mm=r, type=rtype)

    return CenterlineResult(centerline_mask=cl_mask, snakes=snakes, rois=rois)


# ------------------ Baseline 辅助函数 ------------------

def _prune_skeleton_simple(skel: np.ndarray, min_len_vox: int) -> np.ndarray:
    """非常简单的端枝剪枝：从端点往里走，不足阈值的叶子支路移除。"""
    sz, sy, sx = skel.shape
    sk = skel.copy()

    def neighbors(z, y, x):
        for dz in (-1, 0, 1):
            for dy in (-1, 0, 1):
                for dx in (-1, 0, 1):
                    if dz == 0 and dy == 0 and dx == 0: 
                        continue
                    zz, yy, xx = z+dz, y+dy, x+dx
                    if 0 <= zz < sz and 0 <= yy < sy and 0 <= xx < sx and sk[zz,yy,xx]:
                        yield (zz,yy,xx)

    # 找端点
    endpoints = []
    it = np.argwhere(sk > 0)
    for z,y,x in it:
        deg = sum(1 for _ in neighbors(z,y,x))
        if deg == 1:
            endpoints.append((z,y,x))

    # 对每个端点向里走，直到遇到分叉/交点；若长度 < min_len_vox，则删除该路径
    for ez,ey,ex in endpoints:
        path = [(ez,ey,ex)]
        prev = (None, None, None)
        cur = (ez,ey,ex)
        while True:
            neigh = [p for p in neighbors(*cur) if p != prev]
            if len(neigh) != 1:  # 到分叉/交点/尽头
                break
            nxt = neigh[0]
            path.append(nxt)
            prev, cur = cur, nxt
            # 安全长度上限，避免死循环
            if len(path) > 10000: 
                break
        if len(path) < min_len_vox:
            for (zz,yy,xx) in path:
                sk[zz,yy,xx] = 0
    return sk


def _skeleton_to_graph(skel: np.ndarray):
    """把骨架转换为 (nodes, edges)。nodes: dict[node_id]->(z,y,x)；edges: list[list[(z,y,x)]]"""
    sz, sy, sx = skel.shape
    vox_on = set(map(tuple, np.argwhere(skel > 0)))
    nodes: Dict[Tuple[int,int,int], Tuple[int,int,int]] = {}
    edges: List[List[Tuple[int,int,int]]] = []

    def neighs(v):
        z,y,x = v
        out = []
        for dz in (-1,0,1):
            for dy in (-1,0,1):
                for dx in (-1,0,1):
                    if dz==0 and dy==0 and dx==0: 
                        continue
                    u = (z+dz, y+dy, x+dx)
                    if u in vox_on:
                        out.append(u)
        return out

    # 节点=度数!=2 的体素
    for v in list(vox_on):
        deg = len(neighs(v))
        if deg != 2:
            nodes[v] = v

    # 从节点出发沿链条走到下一个节点，形成一条边
    visited = set()
    for s in nodes.keys():
        for t in neighs(s):
            if (s,t) in visited or (t,s) in visited or t not in vox_on:
                continue
            path = [s, t]
            prev, cur = s, t
            while True:
                nb = [u for u in neighs(cur) if u != prev]
                if len(nb) != 1:
                    break
                nxt = nb[0]
                path.append(nxt)
                prev, cur = cur, nxt
                if cur in nodes:
                    break
            visited.add((s, path[1]))
            edges.append(path)
    # 给 nodes 一个稳定的 id
    nodes_dict = {f"{z}-{y}-{x}": (z,y,x) for (z,y,x) in nodes.keys()}
    return nodes_dict, edges


def _resample_edge_voxels(
    epts: List[Tuple[int,int,int]],
    D_mm: np.ndarray,
    spacing_zyx: Tuple[float,float,float],
    step_mm: float = 1.0
):
    """沿边按弧长（mm）采样点，并取半径（来自 D_mm）。返回：[(z,y,x)], [r_mm]"""
    if len(epts) == 0:
        return [], []
    sz, sy, sx = spacing_zyx
    # 体素坐标转 mm 的弧长
    def seg_len_mm(a,b):
        dz = (a[0]-b[0]) * sz
        dy = (a[1]-b[1]) * sy
        dx = (a[2]-b[2]) * sx
        return math.sqrt(dx*dx + dy*dy + dz*dz)

    # 保留路径上的原始体素点，并在累计到 step_mm 时取样（简单邻近，不插值）
    pts_out = [epts[0]]
    rad_out = [float(D_mm[epts[0]])]
    acc = 0.0
    for i in range(1, len(epts)):
        a, b = epts[i-1], epts[i]
        d = seg_len_mm(a,b)
        acc += d
        if acc >= step_mm:
            pts_out.append(b)
            rad_out.append(float(D_mm[b]))
            acc = 0.0
    # 尾点
    if pts_out[-1] != epts[-1]:
        pts_out.append(epts[-1])
        rad_out.append(float(D_mm[epts[-1]]))
    return pts_out, rad_out


def _vox_to_world_xyz(p_zyx: Tuple[int,int,int], affine: np.ndarray) -> Tuple[float,float,float]:
    z,y,x = p_zyx
    v = affine @ np.array([x, y, z, 1.0], dtype=np.float64)
    return (float(v[0]), float(v[1]), float(v[2]))


def _node_degree_from_edges(node_vox, edges):
    deg = 0
    for path in edges:
        if node_vox in (path[0], path[-1]):
            deg += 1
    return deg


def _incident_snake_ids_for_vox(node_vox, edges, snakes: List[Snake], options: CenterlineOptions, affine: np.ndarray):
    ids = []
    # 粗配：在 edges 中做首尾匹配，再按顺序对应 snakes（两者生成顺序一致）
    # 工程上如需更稳，可建哈希映射。这里足够用。
    s_idx = 0
    for path in edges:
        if node_vox in (path[0], path[-1]):
            if s_idx < len(snakes):
                ids.append(snakes[s_idx].id)
        s_idx += 1
    return ids


# ------------------ 导出函数 ------------------

def save_centerline_mask_nii(path: str, cl_mask_zyx: np.ndarray, affine: np.ndarray, label_value: int | None = None):
    """
    保存中心线 NIfTI：
    - 若 label_value 提供，则输出为一个“类标签”mask：中心线处=label_value；否则为 0/1。
    """
    arr = cl_mask_zyx.astype(np.int16, copy=True)
    if label_value is not None:
        arr = (arr > 0).astype(np.int16) * int(label_value)
    nib.save(nib.Nifti1Image(arr, affine), path)


def save_centerline_yaml(
    path: str,
    snakes: List[Snake],
    rois: Dict[str, ROI],
    spacing_zyx: Tuple[float,float,float],
    version: str = "3.2.1",
    are_snakes_key_point_based: bool = True,
):
    import yaml  # pyyaml
    data = {
        "METADATA": {
            "version": version,
            "are_snakes_key_point_based": bool(are_snakes_key_point_based),
            "spacing_mm": [float(spacing_zyx[2]), float(spacing_zyx[1]), float(spacing_zyx[0])]  # (sx,sy,sz) for reference
        },
        "snakes": [
            {
                "id": s.id,
                "points": [list(map(float, p)) for p in s.points],
                "radius_mm": [float(r) for r in s.radius_mm],
                **({"src_label": int(s.src_label)} if s.src_label is not None else {})
            }
            for s in snakes
        ],
        "rois": {
            rid: {
                "position": list(map(float, roi.position)),
                "snake_parents": list(roi.snake_parents),
                "radius_mm": float(roi.radius_mm),
                "type": str(roi.type),
            }
            for rid, roi in rois.items()
        }
    }
    with open(path, "w") as f:
        yaml.safe_dump(data, f, sort_keys=False)
