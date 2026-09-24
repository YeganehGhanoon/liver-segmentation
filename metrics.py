
import numpy as np
from scipy.spatial import cKDTree

def dice_3d(gt, pred):
    gt = gt.astype(bool)
    pred = pred.astype(bool)
    intersection = np.logical_and(gt, pred).sum()
    denominator = gt.sum() + pred.sum()
    if denominator == 0:
        return 1.0
    return 2.0 * intersection / denominator

def iou_3d(gt, pred):
    gt = gt.astype(bool)
    pred = pred.astype(bool)
    intersection = np.logical_and(gt, pred).sum()
    union = np.logical_or(gt, pred).sum()
    if union == 0:
        return 1.0
    return intersection / union

def precision_3d(gt, pred):
    gt = gt.astype(bool)
    pred = pred.astype(bool)
    tp = np.logical_and(gt, pred).sum()
    fp = np.logical_and(~gt, pred).sum()
    if tp + fp == 0:
        return 1.0
    return tp / (tp + fp)

def recall_3d(gt, pred):
    gt = gt.astype(bool)
    pred = pred.astype(bool)
    tp = np.logical_and(gt, pred).sum()
    fn = np.logical_and(gt, ~pred).sum()
    if tp + fn == 0:
        return 1.0
    return tp / (tp + fn)

def extract_surface(mask):
    mask = mask.astype(bool)
    if not mask.any():
        return np.empty((0, 3), dtype=np.int32)
    eroded = mask.copy()
    eroded[1:, :, :] &= mask[:-1, :, :]
    eroded[:-1, :, :] &= mask[1:, :, :]
    eroded[:, 1:, :] &= mask[:, :-1, :]
    eroded[:, :-1, :] &= mask[:, 1:, :]
    eroded[:, :, 1:] &= mask[:, :, :-1]
    eroded[:, :, :-1] &= mask[:, :, 1:]
    surface = mask & ~eroded
    return np.argwhere(surface)

def reduce_points(points, max_points=200000):
    if len(points) <= max_points:
        return points
    indices = np.linspace(0, len(points)-1, max_points).astype(np.int64)
    return points[indices]

def calculate_hd95_physical(gt, pred, spacing_zyx):
    gt = gt.astype(bool)
    pred = pred.astype(bool)
    if not gt.any() and not pred.any():
        return 0.0
    if not gt.any() or not pred.any():
        return np.inf
    gt_surf = extract_surface(gt)
    pred_surf = extract_surface(pred)
    if len(gt_surf) == 0 or len(pred_surf) == 0:
        return np.inf
    gt_surf = reduce_points(gt_surf)
    pred_surf = reduce_points(pred_surf)
    spacing = np.asarray(spacing_zyx, dtype=np.float64)
    gt_mm = gt_surf.astype(np.float64) * spacing
    pred_mm = pred_surf.astype(np.float64) * spacing
    tree_pred = cKDTree(pred_mm)
    tree_gt = cKDTree(gt_mm)
    d1, _ = tree_pred.query(gt_mm, k=1)
    d2, _ = tree_gt.query(pred_mm, k=1)
    distances = np.concatenate([d1, d2])
    return float(np.percentile(distances, 95))

def calculate_hausdorff_physical(gt, pred, spacing_zyx):
    gt = gt.astype(bool)
    pred = pred.astype(bool)
    if not gt.any() and not pred.any():
        return 0.0
    if not gt.any() or not pred.any():
        return np.inf
    gt_surf = extract_surface(gt)
    pred_surf = extract_surface(pred)
    if len(gt_surf) == 0 or len(pred_surf) == 0:
        return np.inf
    gt_surf = reduce_points(gt_surf)
    pred_surf = reduce_points(pred_surf)
    spacing = np.asarray(spacing_zyx, dtype=np.float64)
    gt_mm = gt_surf.astype(np.float64) * spacing
    pred_mm = pred_surf.astype(np.float64) * spacing
    tree_pred = cKDTree(pred_mm)
    tree_gt = cKDTree(gt_mm)
    d1, _ = tree_pred.query(gt_mm, k=1)
    d2, _ = tree_gt.query(pred_mm, k=1)
    return float(max(np.max(d1), np.max(d2)))
