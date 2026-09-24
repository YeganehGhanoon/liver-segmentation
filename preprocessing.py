
import re
import numpy as np
import pydicom
import cv2
from scipy.ndimage import zoom

TARGET_SPACING = 1.0
CROP_TOP, CROP_BOTTOM, CROP_LEFT, CROP_RIGHT = 100, 80, 80, 80
MEDIAN_BLUR_KERNEL = 3
WINDOW_MIN, WINDOW_MAX = -45, 105
CLAHE_CLIP_LIMIT = 2.0
CLAHE_GRID_SIZE = (8, 8)
FINAL_SIZE = (128, 128)

def extract_number(filename):
    numbers = re.findall(r'\d+', filename)
    return int(numbers[0]) if numbers else 0

def read_dicom_with_spacing(file_path):
    ds = pydicom.dcmread(file_path)
    img = ds.pixel_array.astype(np.float32)
    if hasattr(ds, 'RescaleSlope') and hasattr(ds, 'RescaleIntercept'):
        slope = float(ds.RescaleSlope)
        intercept = float(ds.RescaleIntercept)
        img = img * slope + intercept
    spacing_x, spacing_y, spacing_z = 0.98, 0.98, 2.5
    if hasattr(ds, 'PixelSpacing'):
        spacing = ds.PixelSpacing
        if len(spacing) >= 2:
            spacing_x = float(spacing[0])
            spacing_y = float(spacing[1])
    elif hasattr(ds, 'ImagerPixelSpacing'):
        spacing = ds.ImagerPixelSpacing
        if len(spacing) >= 2:
            spacing_x = float(spacing[0])
            spacing_y = float(spacing[1])
    if hasattr(ds, 'SliceThickness'):
        spacing_z = float(ds.SliceThickness)
    elif hasattr(ds, 'SpacingBetweenSlices'):
        spacing_z = float(ds.SpacingBetweenSlices)
    return img, spacing_x, spacing_y, spacing_z

def apply_median_filter(img, kernel_size=3):
    if kernel_size % 2 == 0:
        kernel_size += 1
    if img.dtype == np.uint8:
        return cv2.medianBlur(img, kernel_size)
    else:
        img_uint8 = np.clip(img, 0, 255).astype(np.uint8)
        return cv2.medianBlur(img_uint8, kernel_size)

def apply_window(img, window_min=WINDOW_MIN, window_max=WINDOW_MAX):
    """
    Apply CT windowing (HU windowing).
    Cast to float32 first to avoid uint8 overflow with negative window_min.
    """
    img = np.asarray(img, dtype=np.float32)
    w_min = float(window_min)
    w_max = float(window_max)
    if w_max <= w_min:
        w_max = w_min + 1.0
    img = np.clip(img, w_min, w_max)
    img = (img - w_min) / (w_max - w_min) * 255.0
    img = np.clip(img, 0, 255)
    return img.astype(np.uint8)

def resample_volume_3d(volume, factor_x, factor_y, factor_z):
    if factor_x <= 0 or factor_y <= 0 or factor_z <= 0:
        return volume
    return zoom(volume, (factor_z, factor_y, factor_x), order=1)

def resample_mask_volume_3d(mask_volume, factor_x, factor_y, factor_z):
    if factor_x <= 0 or factor_y <= 0 or factor_z <= 0:
        return mask_volume
    return (zoom(mask_volume, (factor_z, factor_y, factor_x), order=0) > 0.5).astype(np.uint8) * 255

def apply_clahe(img, clip_limit=CLAHE_CLIP_LIMIT, grid_size=CLAHE_GRID_SIZE):
    if img.dtype != np.uint8:
        img = np.clip(img, 0, 255).astype(np.uint8)
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=grid_size)
    return clahe.apply(img)

def resize_or_pad_to_target(img, target_size=FINAL_SIZE):
    target_h, target_w = target_size
    h, w = img.shape[:2]
    scale = min(target_h / h, target_w / w)
    new_h, new_w = int(h * scale), int(w * scale)
    if img.dtype == np.uint8:
        resized = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)
    else:
        resized = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
    pad_h = target_h - new_h
    pad_w = target_w - new_w
    top = pad_h // 2
    bottom = pad_h - top
    left = pad_w // 2
    right = pad_w - left
    padded = cv2.copyMakeBorder(resized, top, bottom, left, right,
                                cv2.BORDER_CONSTANT, value=0)
    return padded

def resize_or_pad_to_target_mask(mask, target_size=FINAL_SIZE):
    target_h, target_w = target_size
    h, w = mask.shape[:2]
    scale = min(target_h / h, target_w / w)
    new_h, new_w = int(h * scale), int(w * scale)
    resized = cv2.resize(mask, (new_w, new_h), interpolation=cv2.INTER_NEAREST)
    pad_h = target_h - new_h
    pad_w = target_w - new_w
    top = pad_h // 2
    bottom = pad_h - top
    left = pad_w // 2
    right = pad_w - left
    padded = cv2.copyMakeBorder(resized, top, bottom, left, right,
                                cv2.BORDER_CONSTANT, value=0)
    return padded

def process_slice_basic(img, verbose=False):
    h, w = img.shape
    if h - CROP_BOTTOM - CROP_TOP > 0 and w - CROP_LEFT - CROP_RIGHT > 0:
        img = img[CROP_TOP:h-CROP_BOTTOM, CROP_LEFT:w-CROP_RIGHT]
    img = apply_median_filter(img, MEDIAN_BLUR_KERNEL)
    img = apply_window(img, WINDOW_MIN, WINDOW_MAX)
    return img

def process_slice_final(img, verbose=False):
    img = apply_clahe(img)
    img = resize_or_pad_to_target(img, FINAL_SIZE)
    return img

def preprocess_volume(img_volume, spacing_zyx):
    spacing_z, spacing_y, spacing_x = spacing_zyx
    factor_x = spacing_x / TARGET_SPACING
    factor_y = spacing_y / TARGET_SPACING
    factor_z = spacing_z / TARGET_SPACING
    img_volume_resampled = resample_volume_3d(img_volume, factor_x, factor_y, factor_z)

    processed_slices = []
    for i in range(img_volume_resampled.shape[0]):
        img = img_volume_resampled[i]
        img = process_slice_basic(img)
        img = process_slice_final(img)
        processed_slices.append(img)
    processed_volume = np.stack(processed_slices, axis=0)

    original_y = img_volume.shape[1]
    original_x = img_volume.shape[2]
    resampled_y = int(round(original_y * spacing_y / TARGET_SPACING))
    resampled_x = int(round(original_x * spacing_x / TARGET_SPACING))
    crop_y = resampled_y - CROP_TOP - CROP_BOTTOM
    crop_x = resampled_x - CROP_LEFT - CROP_RIGHT
    scale = min(FINAL_SIZE[0] / crop_y, FINAL_SIZE[1] / crop_x)
    final_y = TARGET_SPACING / scale
    final_x = TARGET_SPACING / scale
    final_z = TARGET_SPACING
    final_spacing = (final_z, final_y, final_x)

    return processed_volume, final_spacing
