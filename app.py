import tempfile
import shutil
import base64
import io
import html as html_lib
import time
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.backends.backend_agg import FigureCanvasAgg
from scipy.ndimage import gaussian_filter
from skimage import measure
import tensorflow as tf
from PIL import Image
from shiny import App, ui, render, reactive
import shinyswatch

import preprocessing as prep
import metrics as mt

try:
    import pydicom
    _HAS_PYDICOM = True
except ImportError:
    _HAS_PYDICOM = False

# ================== مدل ==================
import os
import requests
from tensorflow.keras.models import load_model

MODEL_URL = "https://github.com/YeganehGhanoon/liver-segmentation/releases/download/model-v1/best_unet_model.h5"
MODEL_PATH = "best_unet_model.h5"

try:
    if not os.path.exists(MODEL_PATH):
        print("Downloading model, please wait...")
        r = requests.get(MODEL_URL, stream=True, timeout=600)
        r.raise_for_status()  # اگر خطای HTTP بدهد، متوقف می‌شود
        with open(MODEL_PATH, "wb") as f:
            for chunk in r.iter_content(chunk_size=8192):
                f.write(chunk)
        print("Model downloaded successfully.")
except Exception as e:
    print(f"Error downloading model: {e}")
    raise

model = load_model(MODEL_PATH, compile=False)
# ================== عکس کبد ==================
def _load_liver_logo() -> str:
    candidates = [
        Path(__file__).parent / "www" / "liver.png",
        Path(__file__).parent / "liver.png",
        Path(__file__).parent / "www" / "liver.jpg",
        Path(__file__).parent / "liver.jpg",
        Path(__file__).parent / "www" / "liver.jpeg",
        Path(__file__).parent / "liver.jpeg",
    ]
    for p in candidates:
        if p.exists():
            suffix = p.suffix.lower()
            mime = "image/png" if suffix == ".png" else "image/jpeg"
            b64 = base64.b64encode(p.read_bytes()).decode("ascii")
            print(f"[logo] loaded: {p}")
            return f"data:{mime};base64,{b64}"
    print("[logo] WARNING: liver image not found.")
    return ""


def _find_image(base_name: str) -> str:
    for ext in (".png", ".jpg", ".jpeg", ".webp"):
        for folder in (Path(__file__).parent / "www", Path(__file__).parent):
            p = folder / f"{base_name}{ext}"
            if p.exists():
                if ext == ".png":
                    mime = "image/png"
                elif ext == ".webp":
                    mime = "image/webp"
                else:
                    mime = "image/jpeg"
                b64 = base64.b64encode(p.read_bytes()).decode("ascii")
                print(f"[gallery] loaded: {p}")
                return f"data:{mime};base64,{b64}"
    print(f"[gallery] WARNING: {base_name} not found.")
    return ""


def _load_liver_gallery() -> list:
    return [_find_image(f"liver{i}") for i in range(1, 5)]


LIVER_LOGO_SRC = _load_liver_logo()
LIVER_GALLERY = [s for s in _load_liver_gallery() if s]


# ================== ترجمه‌ها ==================
TEXT = {
    "en": {
        "please_upload": "Please upload DICOM files and click Run Segmentation.",
        "model_not_loaded": "Model not loaded. Check MODEL_PATH.",
        "no_dicom": "No DICOM image files uploaded.",
        "no_valid_dicom": "No valid DICOM files found.",
        "no_gt_uploaded": "No ground truth masks uploaded. Metrics not computed.",
        "no_gt_3d": "No ground truth masks uploaded.",
        "volume_empty": "Volume is empty; no 3D surface to show.",
        "volume_uniform": "Volume has uniform value; cannot render surface.",
        "no_volume": "No volume available.",
        "no_3d_vertices": "Marching cubes returned no vertices.",
        "3d_error": "3D rendering error",
        "slice": "Slice",
        "dice": "Dice", "iou": "IoU", "precision": "Precision", "recall": "Recall",
        "hd95": "HD95 (mm)", "hausdorff": "AHD (mm)", "hausdorff_2d": "AHD 2D (mm)",
        "slice_num": "Slice #",
        "proc_time": "Processing Time",
        "segmentation_results": "Segmentation Results",
        "slice_metrics_results": "Slice-based Results",
        "original_image": "Original Image",
        "predicted_overlay": "Predicted Mask Overlay",
        "gt_overlay": "Ground Truth Overlay",
        "combined_overlay": "Reference vs Prediction",
        "slice_navigation": "Slice Navigation",
        "go": "Go",
        "legend_tp": "TP (True Positive)",
        "legend_fp": "FP (False Positive)",
        "legend_fn": "FN (False Negative)",
        "legend_liver": "Liver",
        "legend_background": "Background",
        "info_slices": "Slices",
        "info_slices_orig": "Slices (Original)",
        "info_slices_processed": "Slices (Processed)",
        "info_size": "Size", "info_modality": "Modality",
        "upload_data": "Upload Data",
        "select_dicom": "Select DICOM image files",
        "select_mask": "Select DICOM mask files",
        "optional": "Optional",
        "hint_dicom": "Drop your CT scan series here",
        "hint_mask": "Drop your ground-truth masks here",
        "run_seg": "Run Segmentation",
        "view_3d": "3D View",
        "view_3d_title": "3D Liver Reconstruction",
        "view_3d_hint": "Smoothed 3D surface: Ground Truth · Prediction · Difference",
        "show_prediction": "Predicted Volume",
        "show_gt": "Ground Truth Volume",
        "opacity": "Opacity",
        "no_3d_data": "Run segmentation first to see the 3D view.",
        "3d_pred_missing": "Prediction mask is empty — no surface to render.",
        "3d_gt_missing": "Ground-truth mask not available for 3D comparison.",
        "3d_render_title": "Rendered Surface",
        "3d_controls": "Display Controls",
        "3d_info": "Volume Info",
        "voxel_count": "Voxels",
        "surface_area": "Surface (vox²)",
        "hint_3d": "Drag to rotate · Scroll to zoom · Right-drag to pan",
        "3d_gt_title": "Ground Truth",
        "3d_pred_title": "Prediction",
        "3d_diff_title": "Difference",
        "3d_main_title": "3D Qualitative Comparison of Liver Segmentation",
        "smooth_label": "Smoothing (σ)",
        "3d_need_gt": "Ground truth masks are required for the 3D comparison. Please upload them.",
        "3d_show_diff": "Show Difference",
        "menu": "Menu",
        "nav_segmentation": "Segmentation",
        "nav_3d": "3D View",
        "nav_language": "Language",
        "nav_settings": "Settings",
        "nav_about": "About",
        "appearance": "Appearance",
        "language": "Language",
        "theme_light": "Light",
        "theme_dark": "Dark",
        "pred_voxels": "Predicted Voxels",
        "est_volume": "Est. Volume",
        "slices_with_liver": "Slices w/ Liver",
        "coverage": "Coverage",
        "pred_pixels": "Predicted Pixels",
        "area": "Area",
        "na": "N/A",
        "metrics_pred_only": "Showing prediction statistics (no reference mask uploaded for evaluation).",
        "slice_pred_only": "Prediction-only slice statistics.",
        "theme_title": "Theme",
        "lang_title": "Language",
        "diff_title": "Comparison Display",
        "smooth_title": "Surface Smoothing",
        "opacity_title": "Surface Opacity",
        "metrics_3d_title": "Evaluation Metrics",
        "splash_welcome": "Welcome",
        "splash_subtitle": "AI-Powered 3D Liver Analysis",
        "splash_preparing": "Preparing workspace",
        "splash_quote": "Precision today, for a healthier tomorrow",

        # ---- About page ----
        "about_title": "About This Application",
        "about_subtitle": "AI-Powered Liver Segmentation & 3D Analysis",
        "about_description": "Automatic liver segmentation from CT DICOM using a 2D U-Net deep learning model — with quantitative evaluation metrics and 3D surface reconstruction.",
        "about_colors_title": "Color Legend",
        "about_color_tp": "True Positive (overlap)",
        "about_color_fp": "False Positive (predicted only)",
        "about_color_fn": "False Negative (reference only)",
        "about_color_bg": "Background (neither mask)",
        "about_features_title": "Key Features",
        "about_feat_1": "Automatic 2D U-Net segmentation on CT series",
        "about_feat_2": "Quantitative metrics: Dice, IoU, Precision, Recall, HD95, AHD",
        "about_feat_3": "Smooth 3D surface reconstruction via Marching Cubes",
        "about_feat_4": "Interactive slice navigation and zoom",
        "about_dev_title": "Developer",
        "about_dev_name_en": "Yeganeh Ghanoon",
        "about_dev_name_fa": "یگانه قانون",
        "about_dev_role_en": "Designer & Developer",
        "about_dev_role_fa": "طراح و توسعه‌دهنده",
        "about_dev_footer": "Crafted with precision for medical imaging research",
        "about_version": "Version",
        "about_version_value": "1.0.0",
        "about_year": "Year",
        "about_year_value": "2024 – 2026",
    },
    "fa": {
        "please_upload": "لطفاً فایل‌های DICOM را بارگذاری کرده و روی دکمه «اجرای تقسیم‌بندی» کلیک کنید.",
        "model_not_loaded": "مدل بارگذاری نشده است. مسیر MODEL_PATH را بررسی کنید.",
        "no_dicom": "هیچ فایل DICOM بارگذاری نشده است.",
        "no_valid_dicom": "فایل DICOM معتبری یافت نشد.",
        "no_gt_uploaded": "ماسک Ground Truth بارگذاری نشده است. معیارها محاسبه نمی‌شوند.",
        "no_gt_3d": "ماسک Ground Truth بارگذاری نشده است.",
        "volume_empty": "حجم خالی است.",
        "volume_uniform": "حجم یکنواخت است.",
        "no_volume": "حجمی موجود نیست.",
        "no_3d_vertices": "رأسی برنگرداند.",
        "3d_error": "خطای رندر سه‌بعدی",
        "slice": "برش",
        "dice": "دایس", "iou": "IoU", "precision": "دقت", "recall": "بازخوانی",
        "hd95": "HD95 (میلی‌متر)", "hausdorff": "AHD (میلی‌متر)",
        "hausdorff_2d": "AHD دوبعدی (میلی‌متر)",
        "slice_num": "شماره برش",
        "proc_time": "زمان پردازش",
        "segmentation_results": "نتایج تقسیم‌بندی",
        "slice_metrics_results": "نتایج برش‌محور",
        "original_image": "تصویر اصلی",
        "predicted_overlay": "روکش ماسک پیش‌بینی",
        "gt_overlay": "روکش ماسک مرجع",
        "combined_overlay": "مرجع در مقابل پیش‌بینی",
        "slice_navigation": "پیمایش برش‌ها",
        "go": "برو",
        "legend_tp": "TP (مثبت واقعی)",
        "legend_fp": "FP (مثبت کاذب)",
        "legend_fn": "FN (منفی کاذب)",
        "legend_liver": "کبد",
        "legend_background": "پس‌زمینه",
        "info_slices": "تعداد برش‌ها",
        "info_slices_orig": "برش‌های اصلی",
        "info_slices_processed": "برش‌های پیش‌پردازش",
        "info_size": "اندازه", "info_modality": "مودالیتی",
        "upload_data": "بارگذاری داده",
        "select_dicom": "انتخاب فایل‌های تصویری DICOM",
        "select_mask": "انتخاب فایل‌های ماسک DICOM",
        "optional": "اختیاری",
        "hint_dicom": "سری اسکن CT را اینجا رها کنید",
        "hint_mask": "ماسک‌های مرجع را اینجا رها کنید",
        "run_seg": "اجرای تقسیم‌بندی",
        "view_3d": "نمای سه‌بعدی",
        "view_3d_title": "بازسازی سه‌بعدی کبد",
        "view_3d_hint": "سطح سه‌بعدی هموار: مرجع · پیش‌بینی · تفاوت",
        "show_prediction": "حجم پیش‌بینی",
        "show_gt": "حجم مرجع",
        "opacity": "شفافیت",
        "no_3d_data": "ابتدا تقسیم‌بندی را اجرا کنید تا نمای سه‌بعدی نمایش داده شود.",
        "3d_pred_missing": "ماسک پیش‌بینی خالی است — سطحی برای رندر وجود ندارد.",
        "3d_gt_missing": "ماسک مرجع برای مقایسهٔ سه‌بعدی در دسترس نیست.",
        "3d_render_title": "سطح رندرشده",
        "3d_controls": "کنترل‌های نمایش",
        "3d_info": "اطلاعات حجم",
        "voxel_count": "وکسل‌ها",
        "surface_area": "سطح (وکسل²)",
        "hint_3d": "برای چرخش بکشید · برای بزرگ‌نمایی اسکرول کنید",
        "3d_gt_title": "مرجع",
        "3d_pred_title": "پیش‌بینی",
        "3d_diff_title": "تفاوت",
        "3d_main_title": "مقایسهٔ کیفی سه‌بعدی تقسیم‌بندی کبد",
        "smooth_label": "هموارسازی (σ)",
        "3d_need_gt": "برای مقایسهٔ سه‌بعدی، ماسک مرجع لازم است. لطفاً بارگذاری کنید.",
        "3d_show_diff": "نمایش تفاوت",
        "menu": "منو",
        "nav_segmentation": "تقسیم‌بندی",
        "nav_3d": "نمای سه‌بعدی",
        "nav_language": "زبان",
        "nav_settings": "تنظیمات",
        "nav_about": "درباره",
        "appearance": "ظاهر",
        "language": "زبان",
        "theme_light": "روشن",
        "theme_dark": "تاریک",
        "pred_voxels": "وکسل‌های پیش‌بینی",
        "est_volume": "حجم تخمینی",
        "slices_with_liver": "برش‌های دارای کبد",
        "coverage": "پوشش",
        "pred_pixels": "پیکسل‌های پیش‌بینی",
        "area": "مساحت",
        "na": "ناموجود",
        "metrics_pred_only": "نمایش آمار پیش‌بینی (ماسک مرجع برای ارزیابی بارگذاری نشده است).",
        "slice_pred_only": "آمار برش بر اساس پیش‌بینی.",
        "theme_title": "پوسته",
        "lang_title": "زبان",
        "diff_title": "نمایش مقایسه",
        "smooth_title": "هموارسازی سطح",
        "opacity_title": "شفافیت سطح",
        "metrics_3d_title": "معیارهای ارزیابی",
        "splash_welcome": "خوش آمدید",
        "splash_subtitle": "تحلیل سه‌بعدی کبد با هوش مصنوعی",
        "splash_preparing": "در حال آماده‌سازی محیط کاری",
        "splash_quote": "دقت امروز، برای فردایی سالم‌تر",

        # ---- About page ----
        "about_title": "درباره این برنامه",
        "about_subtitle": "تقسیم‌بندی کبد و تحلیل سه‌بعدی با هوش مصنوعی",
        "about_description": "تقسیم‌بندی خودکار کبد از تصاویر CT DICOM با مدل یادگیری عمیق U-Net دوبعدی — همراه با معیارهای ارزیابی کمی و بازسازی سطح سه‌بعدی.",
        "about_colors_title": "راهنمای رنگ‌ها",
        "about_color_tp": "مثبت واقعی (هم‌پوشانی)",
        "about_color_fp": "مثبت کاذب (فقط پیش‌بینی)",
        "about_color_fn": "منفی کاذب (فقط مرجع)",
        "about_color_bg": "پس‌زمینه (هیچ‌کدام)",
        "about_features_title": "ویژگی‌های کلیدی",
        "about_feat_1": "تقسیم‌بندی خودکار ۲بعدی با U-Net روی سری CT",
        "about_feat_2": "معیارهای کمی: دایس، IoU، دقت، بازخوانی، HD95 و AHD",
        "about_feat_3": "بازسازی سطح سه‌بعدی هموار با Marching Cubes",
        "about_feat_4": "پیمایش و بزرگ‌نمایی تعاملی برش‌ها",
        "about_dev_title": "توسعه‌دهنده",
        "about_dev_name_en": "Yeganeh Ghanoon",
        "about_dev_name_fa": "یگانه قانون",
        "about_dev_role_en": "Designer & Developer",
        "about_dev_role_fa": "طراح و توسعه‌دهنده",
        "about_dev_footer": "با دقت برای پژوهش‌های تصویربرداری پزشکی ساخته شده است",
        "about_version": "نسخه",
        "about_version_value": "1.0.0",
        "about_year": "سال",
        "about_year_value": "۱۴۰۳ – ۱۴۰۵",
    },
}


def bilingual(en_text: str, fa_text: str):
    return ui.HTML(
        f'<span class="en-text">{html_lib.escape(en_text)}</span>'
        f'<span class="fa-text">{html_lib.escape(fa_text)}</span>'
    )


# ================== آیکون‌ها ==================
METRICS_ICON_SVG = """<svg class="metrics-icon-svg" viewBox="0 0 24 24"
    width="22" height="22" fill="none" stroke="currentColor"
    stroke-width="2.2" stroke-linecap="round" stroke-linejoin="round">
    <line x1="6" y1="20" x2="6" y2="14"></line>
    <line x1="12" y1="20" x2="12" y2="8"></line>
    <line x1="18" y1="20" x2="18" y2="4"></line>
    <line x1="3" y1="20" x2="21" y2="20"></line>
</svg>"""

SLICE_NAV_ICON_SVG = """<svg viewBox="0 0 24 24" width="22" height="22"
    fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round"
    stroke-linejoin="round">
    <rect x="3" y="3" width="18" height="18" rx="2"></rect>
    <circle cx="8.5" cy="8.5" r="1.5"></circle>
    <polyline points="21 15 16 10 5 21"></polyline>
</svg>"""

INFO_ICON_SVG = """<svg viewBox="0 0 24 24" width="20" height="20"
    fill="none" stroke="currentColor" stroke-width="2.2"
    stroke-linecap="round" stroke-linejoin="round">
    <circle cx="12" cy="12" r="10"></circle>
    <line x1="12" y1="16" x2="12" y2="12"></line>
    <line x1="12" y1="8" x2="12.01" y2="8"></line>
</svg>"""

ZOOM_IN_SVG = """<svg viewBox="0 0 24 24" width="14" height="14" fill="none"
    stroke="currentColor" stroke-width="2.4" stroke-linecap="round"
    stroke-linejoin="round"><circle cx="11" cy="11" r="7"></circle>
    <line x1="21" y1="21" x2="16.65" y2="16.65"></line>
    <line x1="11" y1="8" x2="11" y2="14"></line>
    <line x1="8" y1="11" x2="14" y2="11"></line></svg>"""

ZOOM_OUT_SVG = """<svg viewBox="0 0 24 24" width="14" height="14" fill="none"
    stroke="currentColor" stroke-width="2.4" stroke-linecap="round"
    stroke-linejoin="round"><circle cx="11" cy="11" r="7"></circle>
    <line x1="21" y1="21" x2="16.65" y2="16.65"></line>
    <line x1="8" y1="11" x2="14" y2="11"></line></svg>"""

FIT_SVG = """<svg viewBox="0 0 24 24" width="14" height="14" fill="none"
    stroke="currentColor" stroke-width="2.2" stroke-linecap="round"
    stroke-linejoin="round"><path d="M3 7V5a2 2 0 0 1 2-2h2"></path>
    <path d="M17 3h2a2 2 0 0 1 2 2v2"></path>
    <path d="M21 17v2a2 2 0 0 1-2 2h-2"></path>
    <path d="M7 21H5a2 2 0 0 1-2-2v-2"></path></svg>"""

EXPAND_SVG = """<svg viewBox="0 0 24 24" width="14" height="14" fill="none"
    stroke="currentColor" stroke-width="2.2" stroke-linecap="round"
    stroke-linejoin="round"><polyline points="15 3 21 3 21 9"></polyline>
    <polyline points="9 21 3 21 3 15"></polyline>
    <line x1="21" y1="3" x2="14" y2="10"></line>
    <line x1="3" y1="21" x2="10" y2="14"></line></svg>"""

UPLOAD_TITLE_ICON_SVG = """<svg viewBox="0 0 24 24" width="24" height="24"
    fill="none" stroke="currentColor" stroke-width="2.2"
    stroke-linecap="round" stroke-linejoin="round">
    <path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4"></path>
    <polyline points="17 8 12 3 7 8"></polyline>
    <line x1="12" y1="3" x2="12" y2="15"></line>
</svg>"""

DICOM_IMG_ICON_SVG = """<svg viewBox="0 0 24 24" width="26" height="26"
    fill="none" stroke="currentColor" stroke-width="1.9"
    stroke-linecap="round" stroke-linejoin="round">
    <rect x="3" y="3" width="18" height="18" rx="2.5"></rect>
    <circle cx="8.5" cy="8.5" r="1.6"></circle>
    <polyline points="21 15 16 10 5 21"></polyline>
</svg>"""

MASK_ICON_SVG = """<svg viewBox="0 0 24 24" width="26" height="26"
    fill="none" stroke="currentColor" stroke-width="1.9"
    stroke-linecap="round" stroke-linejoin="round">
    <path d="M3 12c0-4.5 3.5-8 8-8s8 3.5 8 8"></path>
    <circle cx="8.5" cy="14" r="1.5"></circle>
    <circle cx="15.5" cy="14" r="1.5"></circle>
    <path d="M12 20c4.5 0 8-2.5 8-5"></path>
    <path d="M4 20c0-2 1.5-4 4-4"></path>
    <path d="M20 20c0-2-1.5-4-4-4"></path>
</svg>"""

PLAY_ICON_SVG = """<svg viewBox="0 0 24 24" width="20" height="20"
    fill="none" stroke="currentColor" stroke-width="2.4"
    stroke-linecap="round" stroke-linejoin="round">
    <polygon points="6 3 20 12 6 21 6 3" fill="currentColor" stroke="none"></polygon>
</svg>"""

CUBE_ICON_SVG = """<svg viewBox="0 0 24 24" width="24" height="24"
    fill="none" stroke="currentColor" stroke-width="1.9"
    stroke-linecap="round" stroke-linejoin="round">
    <path d="M21 16V8a2 2 0 0 0-1-1.73l-7-4a2 2 0 0 0-2 0l-7 4A2 2 0 0 0 3 8v8a2 2 0 0 0 1 1.73l7 4a2 2 0 0 0 2 0l7-4A2 2 0 0 0 21 16z"></path>
    <polyline points="3.27 6.96 12 12.01 20.73 6.96"></polyline>
    <line x1="12" y1="22.08" x2="12" y2="12"></line>
</svg>"""

ROTATE_ICON_SVG = """<svg viewBox="0 0 24 24" width="22" height="22"
    fill="none" stroke="currentColor" stroke-width="1.9"
    stroke-linecap="round" stroke-linejoin="round">
    <polyline points="23 4 23 10 17 10"></polyline>
    <path d="M20.49 15a9 9 0 1 1-2.12-9.36L23 10"></path>
</svg>"""

HAMBURGER_ICON_SVG = """<svg viewBox="0 0 24 24" width="24" height="24"
    fill="none" stroke="currentColor" stroke-width="2.4"
    stroke-linecap="round" stroke-linejoin="round">
    <line x1="3" y1="6" x2="21" y2="6"></line>
    <line x1="3" y1="12" x2="21" y2="12"></line>
    <line x1="3" y1="18" x2="21" y2="18"></line>
</svg>"""

CLOSE_ICON_SVG = """<svg viewBox="0 0 24 24" width="20" height="20"
    fill="none" stroke="currentColor" stroke-width="2.4"
    stroke-linecap="round" stroke-linejoin="round">
    <line x1="18" y1="6" x2="6" y2="18"></line>
    <line x1="6" y1="6" x2="18" y2="18"></line>
</svg>"""

CHEVRON_DOWN_SVG = """<svg viewBox="0 0 24 24" width="16" height="16"
    fill="none" stroke="currentColor" stroke-width="2.4"
    stroke-linecap="round" stroke-linejoin="round">
    <polyline points="6 9 12 15 18 9"></polyline>
</svg>"""

SEGMENTATION_NAV_ICON_SVG = """<svg viewBox="0 0 24 24" width="22" height="22"
    fill="none" stroke="currentColor" stroke-width="2"
    stroke-linecap="round" stroke-linejoin="round">
    <line x1="6" y1="20" x2="6" y2="14"></line>
    <line x1="12" y1="20" x2="12" y2="8"></line>
    <line x1="18" y1="20" x2="18" y2="4"></line>
    <line x1="3" y1="20" x2="21" y2="20"></line>
</svg>"""

GLOBE_NAV_ICON_SVG = """<svg viewBox="0 0 24 24" width="22" height="22"
    fill="none" stroke="currentColor" stroke-width="2"
    stroke-linecap="round" stroke-linejoin="round">
    <circle cx="12" cy="12" r="10"></circle>
    <line x1="2" y1="12" x2="22" y2="12"></line>
    <path d="M12 2a15.3 15.3 0 0 1 4 10 15.3 15.3 0 0 1-4 10 15.3 15.3 0 0 1-4-10 15.3 15.3 0 0 1 4-10z"></path>
</svg>"""

SETTINGS_NAV_ICON_SVG = """<svg viewBox="0 0 24 24" width="22" height="22"
    fill="none" stroke="currentColor" stroke-width="2"
    stroke-linecap="round" stroke-linejoin="round">
    <circle cx="12" cy="12" r="3"></circle>
    <path d="M19.4 15a1.65 1.65 0 0 0 .33 1.82l.06.06a2 2 0 0 1 0 2.83 2 2 0 0 1-2.83 0l-.06-.06a1.65 1.65 0 0 0-1.82-.33 1.65 1.65 0 0 0-1 1.51V21a2 2 0 0 1-2 2 2 2 0 0 1-2-2v-.09A1.65 1.65 0 0 0 9 19.4a1.65 1.65 0 0 0-1.82.33l-.06.06a2 2 0 0 1-2.83 0 2 2 0 0 1 0-2.83l.06-.06a1.65 1.65 0 0 0 .33-1.82 1.65 1.65 0 0 0-1.51-1H3a2 2 0 0 1-2-2 2 2 0 0 1 2-2h.09A1.65 1.65 0 0 0 4.6 9a1.65 1.65 0 0 0-.33-1.82l-.06-.06a2 2 0 0 1 0-2.83 2 2 0 0 1 2.83 0l.06.06a1.65 1.65 0 0 0 1.82.33H9a1.65 1.65 0 0 0 1-1.51V3a2 2 0 0 1 2-2 2 2 0 0 1 2 2v.09a1.65 1.65 0 0 0 1 1.51 1.65 1.65 0 0 0 1.82-.33l.06-.06a2 2 0 0 1 2.83 0 2 2 0 0 1 0 2.83l-.06.06a1.65 1.65 0 0 0-.33 1.82V9a1.65 1.65 0 0 0 1.51 1H21a2 2 0 0 1 2 2 2 2 0 0 1-2 2h-.09a1.65 1.65 0 0 0-1.51 1z"></path>
</svg>"""

INFO_NAV_ICON_SVG = """<svg viewBox="0 0 24 24" width="22" height="22"
    fill="none" stroke="currentColor" stroke-width="2"
    stroke-linecap="round" stroke-linejoin="round">
    <circle cx="12" cy="12" r="10"></circle>
    <line x1="12" y1="16" x2="12" y2="12"></line>
    <line x1="12" y1="8" x2="12.01" y2="8"></line>
</svg>"""

ICON_DICE = """<svg viewBox="0 0 24 24" width="16" height="16" fill="none"
    stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
    <circle cx="9" cy="12" r="6"></circle><circle cx="15" cy="12" r="6"></circle></svg>"""
ICON_IOU = """<svg viewBox="0 0 24 24" width="16" height="16" fill="none"
    stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
    <rect x="3" y="3" width="12" height="12" rx="2"></rect>
    <rect x="9" y="9" width="12" height="12" rx="2"></rect></svg>"""
ICON_PRECISION = """<svg viewBox="0 0 24 24" width="16" height="16" fill="none"
    stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
    <circle cx="12" cy="12" r="9"></circle><circle cx="12" cy="12" r="4"></circle>
    <line x1="12" y1="3" x2="12" y2="8"></line></svg>"""
ICON_RECALL = """<svg viewBox="0 0 24 24" width="16" height="16" fill="none"
    stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
    <path d="M3 12a9 9 0 1 0 3-6.7"></path><polyline points="3 4 3 9 8 9"></polyline></svg>"""
ICON_HD95 = """<svg viewBox="0 0 24 24" width="16" height="16" fill="none"
    stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
    <line x1="3" y1="12" x2="21" y2="12"></line>
    <polyline points="8 7 3 12 8 17"></polyline>
    <polyline points="16 7 21 12 16 17"></polyline></svg>"""
ICON_AHD = """<svg viewBox="0 0 24 24" width="16" height="16" fill="none"
    stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
    <polyline points="3 17 9 11 13 15 21 7"></polyline>
    <polyline points="15 7 21 7 21 13"></polyline></svg>"""
ICON_SLICE = """<svg viewBox="0 0 24 24" width="16" height="16" fill="none"
    stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
    <rect x="3" y="3" width="18" height="18" rx="2"></rect>
    <line x1="3" y1="9" x2="21" y2="9"></line>
    <line x1="3" y1="15" x2="21" y2="15"></line></svg>"""
ICON_TIME = """<svg viewBox="0 0 24 24" width="16" height="16" fill="none"
    stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
    <circle cx="12" cy="12" r="9"></circle>
    <polyline points="12 7 12 12 15 14"></polyline></svg>"""


# ================== CSS ==================
CSS = """
:root {
    --slate-50:  #f8fafc; --slate-100: #f1f5f9; --slate-200: #e2e8f0;
    --slate-300: #cbd5e1; --slate-400: #94a3b8; --slate-500: #64748b;
    --slate-600: #475569; --slate-700: #334155; --slate-800: #1e293b;
    --slate-900: #0f172a; --slate-950: #020617;

    --brand-900: #0c4a6e; --brand-800: #075985; --brand-700: #0e7490;
    --brand-600: #0891b2; --brand-500: #06b6d4; --brand-400: #22d3ee;
    --brand-300: #67e8f9; --brand-200: #a5f3fc;

    --coral-700: #be123c; --coral-600: #e11d48; --coral-500: #f43f5e;
    --coral-400: #fb7185; --coral-300: #fda4af;

    --indigo-600: #4f46e5; --indigo-500: #6366f1; --indigo-400: #818cf8;
    --violet-600: #7c3aed; --violet-500: #8b5cf6; --violet-400: #a78bfa;

    --bg-body: #f8fafc;
    --bg-panel-left: #ffffff;
    --bg-panel-right: #ffffff;
    --text-main: #0f172a;
    --text-muted: #64748b;
    --border: #e2e8f0;
    --border-strong: #cbd5e1;
    --shadow-panel: 0 4px 24px rgba(15, 23, 42, 0.06);

    --mc-bg: #ffffff;
    --mc-border: rgba(15, 23, 42, 0.08);
    --mc-border-hover: rgba(6, 182, 212, 0.45);

    --up-card-bg: #ffffff;
    --up-card-border: #e2e8f0;
    --up-card-hover: rgba(6, 182, 212, 0.55);
    --up-hint: #94a3b8;
    --up-icon: #0e7490;
    --up-badge-bg: rgba(6, 182, 212, 0.12);
    --up-badge-text: #0e7490;

    --grad-btn: linear-gradient(135deg, #0e7490 0%, #06b6d4 100%);
    --grad-btn-hover: linear-gradient(135deg, #0891b2 0%, #22d3ee 100%);
    --grad-header: linear-gradient(135deg, #0c4a6e 0%, #075985 40%, #0e7490 75%, #0891b2 100%);
    --grad-header-glow: radial-gradient(circle, rgba(251, 113, 133, 0.30), transparent 65%);
}

body.theme-dark {
    --bg-body: #020617;
    --bg-panel-left: #0f172a;
    --bg-panel-right: #0f172a;
    --text-main: #e2e8f0;
    --text-muted: #94a3b8;
    --border: #1e293b;
    --border-strong: #334155;
    --shadow-panel: 0 8px 32px rgba(0, 0, 0, 0.5);

    --mc-bg: linear-gradient(145deg, #0f172a 0%, #0b1220 100%);
    --mc-border: rgba(34, 211, 238, 0.16);
    --mc-border-hover: rgba(34, 211, 238, 0.48);

    --up-card-bg: linear-gradient(145deg, #0f172a 0%, #0b1220 100%);
    --up-card-border: #1e293b;
    --up-card-hover: rgba(34, 211, 238, 0.45);
    --up-hint: #64748b;
    --up-icon: #22d3ee;
    --up-badge-bg: rgba(34, 211, 238, 0.14);
    --up-badge-text: #67e8f9;

    --grad-btn: linear-gradient(135deg, #0891b2 0%, #22d3ee 100%);
    --grad-btn-hover: linear-gradient(135deg, #06b6d4 0%, #67e8f9 100%);
    --grad-header: linear-gradient(135deg, #020617 0%, #0c4a6e 45%, #0e7490 100%);
    --grad-header-glow: radial-gradient(circle, rgba(251, 113, 133, 0.22), transparent 65%);
}

.lang-en .fa-text { display: none !important; }
.lang-fa .en-text { display: none !important; }

body.lang-fa .left-panel h4,
body.lang-fa .right-panel h4,
body.lang-fa .metrics-panel h4,
body.lang-fa .input-label,
body.lang-fa .btn-run,
body.lang-fa .about-body { direction: rtl; text-align: right; }

.file-input-wrapper, .file-input-wrapper *,
.shiny-input-container input[type="file"] {
    direction: ltr !important; text-align: left !important;
}

body {
    background-color: var(--bg-body) !important;
    color: var(--text-main);
    transition: background-color 0.35s ease, color 0.35s ease;
    font-family: 'Inter', 'Segoe UI', system-ui, -apple-system, sans-serif;
}
.container-fluid { background-color: var(--bg-body); padding: 18px 22px; }

/* ============ HEADER ============ */
.app-header {
    position: relative;
    background: var(--grad-header);
    color: white;
    padding: 26px 90px 26px 30px;
    border-radius: 18px; margin-bottom: 25px;
    text-align: left;
    box-shadow: 0 12px 34px rgba(12, 74, 110, 0.30);
    display: flex; align-items: center; justify-content: flex-start;
    min-height: 128px;
    overflow: hidden;
}
.app-header::after {
    content: '';
    position: absolute;
    right: -80px; top: -80px;
    width: 300px; height: 300px;
    background: var(--grad-header-glow);
    pointer-events: none;
}
.app-header::before {
    content: '';
    position: absolute;
    left: -100px; bottom: -100px;
    width: 280px; height: 280px;
    background: radial-gradient(circle, rgba(103, 232, 249, 0.20), transparent 70%);
    pointer-events: none;
}
.header-content {
    display: flex; align-items: center; justify-content: flex-start;
    gap: 22px; flex-wrap: nowrap; width: 100%;
    position: relative; z-index: 1;
}
.header-liver-icon {
    width: 78px; height: 78px;
    flex-shrink: 0; object-fit: contain;
    filter: drop-shadow(0 4px 18px rgba(103, 232, 249, 0.55));
    animation: liverPulse 3.5s ease-in-out infinite;
}
@keyframes liverPulse {
    0%, 100% { transform: scale(1); }
    50%      { transform: scale(1.06); }
}
.header-titles {
    display: flex; flex-direction: column;
    align-items: flex-start; justify-content: center; text-align: left;
}
.app-header h1 {
    font-weight: 800; font-size: 2.15rem; margin: 0;
    color: #ffffff; line-height: 1.15; letter-spacing: 0.3px;
    text-shadow: 0 2px 14px rgba(0, 0, 0, 0.35);
}
.app-subtitle {
    font-size: 1.05rem; font-weight: 500; margin-top: 8px;
    color: #a5f3fc;
    letter-spacing: 0.8px;
    display: inline-flex;
    align-items: center;
    gap: 8px;
}
.app-subtitle::before {
    content: '';
    width: 8px; height: 8px;
    border-radius: 50%;
    background: #fb7185;
    box-shadow: 0 0 12px #fb7185;
    flex-shrink: 0;
}
body.lang-fa .header-content { flex-direction: row-reverse; }
body.lang-fa .header-titles { align-items: flex-end; text-align: right; }
body.lang-fa .app-header {
    text-align: right;
    padding: 26px 30px 26px 90px;
}
@media (max-width: 700px) {
    .app-header { padding: 20px 70px 20px 18px; min-height: 100px; }
    body.lang-fa .app-header { padding: 20px 18px 20px 70px; }
    .header-liver-icon { width: 56px; height: 56px; }
    .app-header h1 { font-size: 1.55rem; }
    .app-subtitle { font-size: 0.85rem; letter-spacing: 0.3px; }
    .header-content { gap: 14px; }
}

/* ============ HAMBURGER BUTTON ============ */
.hamburger-btn {
    position: absolute;
    top: 22px; right: 22px;
    width: 48px; height: 48px;
    background: rgba(255, 255, 255, 0.15);
    color: #ffffff;
    border: 1.5px solid rgba(255, 255, 255, 0.45);
    border-radius: 50%;
    cursor: pointer;
    display: flex; align-items: center; justify-content: center;
    padding: 0; outline: none;
    z-index: 1040;
    backdrop-filter: blur(8px);
    -webkit-backdrop-filter: blur(8px);
    transition: background 0.2s, border-color 0.2s, transform 0.15s, box-shadow 0.2s;
}
.hamburger-btn:hover {
    background: rgba(255, 255, 255, 0.28);
    border-color: #a5f3fc;
    transform: scale(1.06);
    box-shadow: 0 0 20px rgba(165, 243, 252, 0.45);
}
.hamburger-btn:active { transform: scale(0.98); }
body.lang-fa .hamburger-btn { right: auto; left: 22px; }

/* ============ SIDEBAR ============ */
.sidebar-overlay {
    position: fixed; inset: 0;
    background: rgba(2, 6, 23, 0.6);
    backdrop-filter: blur(4px);
    -webkit-backdrop-filter: blur(4px);
    opacity: 0; visibility: hidden;
    transition: opacity 0.3s ease, visibility 0.3s ease;
    z-index: 1090;
}
.sidebar-overlay.open { opacity: 1; visibility: visible; }

.sidebar {
    position: fixed;
    top: 0; right: 0;
    width: 320px; height: 100vh;
    background: var(--bg-panel-right);
    border-left: 1px solid var(--border);
    box-shadow: -10px 0 40px rgba(2, 6, 23, 0.35);
    transform: translateX(100%);
    transition: transform 0.35s cubic-bezier(0.4, 0, 0.2, 1);
    z-index: 1100;
    display: flex; flex-direction: column;
    overflow: hidden;
}
.sidebar.open { transform: translateX(0); }
body.lang-fa .sidebar {
    right: auto; left: 0;
    border-left: none;
    border-right: 1px solid var(--border);
    box-shadow: 10px 0 40px rgba(2, 6, 23, 0.35);
    transform: translateX(-100%);
}
body.lang-fa .sidebar.open { transform: translateX(0); }

.sidebar-header {
    display: flex; align-items: center; justify-content: space-between;
    padding: 22px 22px;
    background: var(--grad-header);
    color: #ffffff;
    flex-shrink: 0;
    position: relative;
    overflow: hidden;
}
.sidebar-header::after {
    content: '';
    position: absolute;
    right: -60px; top: -60px;
    width: 200px; height: 200px;
    background: var(--grad-header-glow);
    pointer-events: none;
}
.sidebar-title {
    font-size: 1.2rem;
    font-weight: 700;
    letter-spacing: 0.5px;
    display: inline-flex; align-items: center; gap: 10px;
    position: relative; z-index: 1;
}
.sidebar-title::before {
    content: '';
    width: 8px; height: 8px;
    border-radius: 50%;
    background: #fb7185;
    box-shadow: 0 0 12px #fb7185;
    flex-shrink: 0;
}
.sidebar-close {
    background: rgba(255, 255, 255, 0.15);
    border: 1px solid rgba(255, 255, 255, 0.35);
    color: #ffffff;
    width: 36px; height: 36px;
    border-radius: 50%;
    cursor: pointer;
    display: flex; align-items: center; justify-content: center;
    padding: 0;
    flex-shrink: 0;
    transition: background 0.2s, transform 0.25s, border-color 0.2s;
    position: relative; z-index: 1;
}
.sidebar-close:hover {
    background: rgba(255, 255, 255, 0.30);
    border-color: #a5f3fc;
    transform: rotate(90deg);
}

.sidebar-nav {
    padding: 14px 12px;
    display: flex; flex-direction: column;
    gap: 4px;
    flex: 1 1 auto;
    overflow-y: auto;
}

.sidebar-item {
    display: flex; align-items: center;
    gap: 14px;
    width: 100%;
    padding: 14px 18px;
    background: transparent;
    border: none;
    border-radius: 10px;
    color: var(--text-main);
    font-size: 1rem;
    font-weight: 500;
    font-family: inherit;
    cursor: pointer;
    text-align: left;
    transition: background 0.15s, color 0.15s, transform 0.15s;
    position: relative;
    letter-spacing: 0.2px;
}
body.lang-fa .sidebar-item { text-align: right; flex-direction: row-reverse; }
.sidebar-item:hover {
    background: rgba(6, 182, 212, 0.10);
    color: var(--brand-700);
}
body.theme-dark .sidebar-item:hover { color: #22d3ee; }
.sidebar-item.active {
    background: var(--grad-btn);
    color: #ffffff;
    box-shadow: 0 4px 14px rgba(6, 182, 212, 0.35);
}
.sidebar-item.active:hover {
    color: #ffffff;
    background: var(--grad-btn-hover);
}
.sidebar-item .item-icon {
    display: inline-flex;
    align-items: center; justify-content: center;
    flex-shrink: 0;
    width: 22px; height: 22px;
}
.sidebar-item .item-icon svg {
    width: 20px !important;
    height: 20px !important;
}
.sidebar-item .item-text {
    flex: 1;
    font-size: 1rem;
    display: inline-flex;
    align-items: center;
}
.sidebar-item .item-chevron {
    display: inline-flex;
    align-items: center; justify-content: center;
    transition: transform 0.25s ease;
    opacity: 0.7;
    flex-shrink: 0;
}
.sidebar-item .item-chevron svg {
    width: 14px !important;
    height: 14px !important;
}
.sidebar-item.expanded .item-chevron {
    transform: rotate(180deg);
}

.sidebar-group { display: contents; }

.sidebar-submenu {
    max-height: 0;
    overflow: hidden;
    transition: max-height 0.3s ease;
    margin: 0 4px;
    border-radius: 10px;
}
.sidebar-submenu.open { max-height: 400px; }

.submenu-item {
    display: flex; align-items: center; gap: 10px;
    width: 100%;
    padding: 11px 20px 11px 54px;
    background: transparent;
    border: none;
    border-radius: 8px;
    color: var(--text-muted);
    font-size: 0.95rem;
    font-weight: 500;
    font-family: inherit;
    cursor: pointer;
    text-align: left;
    transition: background 0.15s, color 0.15s;
    position: relative;
}
body.lang-fa .submenu-item {
    padding: 11px 54px 11px 20px;
    text-align: right;
    flex-direction: row-reverse;
}
.submenu-item:hover {
    background: rgba(6, 182, 212, 0.10);
    color: var(--brand-700);
}
body.theme-dark .submenu-item:hover { color: #67e8f9; }
.submenu-item.active {
    background: rgba(6, 182, 212, 0.16);
    color: var(--brand-700);
    font-weight: 700;
}
body.theme-dark .submenu-item.active { color: #22d3ee; }
.submenu-item.active::after {
    content: '✓';
    position: absolute;
    right: 20px;
    color: var(--brand-600);
    font-weight: 800;
    font-size: 1rem;
}
body.lang-fa .submenu-item.active::after {
    right: auto; left: 20px;
}
.submenu-item .opt-icon {
    font-size: 1.1rem; line-height: 1;
    display: inline-flex; flex-shrink: 0;
}

.sidebar-submenu-sub {
    max-height: 0;
    overflow: hidden;
    transition: max-height 0.3s ease;
    margin: 0 4px 0 16px;
    border-left: 2px solid var(--border);
}
.sidebar-submenu-sub.open { max-height: 150px; }
.submenu-item.nested-toggle {
    padding-left: 40px;
    justify-content: space-between;
}
body.lang-fa .submenu-item.nested-toggle {
    padding-left: 20px;
    padding-right: 40px;
}
.submenu-item.nested-toggle .item-chevron {
    display: inline-flex;
    transition: transform 0.25s ease;
    opacity: 0.7;
}
.submenu-item.nested-toggle.expanded .item-chevron {
    transform: rotate(180deg);
}
.sidebar-submenu-sub .submenu-item {
    padding-left: 60px;
}
body.lang-fa .sidebar-submenu-sub .submenu-item {
    padding-left: 20px;
    padding-right: 60px;
}

@media (max-width: 500px) {
    .sidebar { width: 85vw; }
}

/* ============ Panels ============ */
.left-panel, .right-panel {
    background: var(--bg-panel-left);
    color: var(--text-main);
    border-radius: 16px; padding: 22px;
    box-shadow: var(--shadow-panel);
    border: 1px solid var(--border);
}
.right-panel { background: var(--bg-panel-right); }

/* ============ UPLOAD ============ */
.panel-title-row {
    display: flex; align-items: center; gap: 12px;
    margin-bottom: 20px; padding-bottom: 14px;
    border-bottom: 1px solid var(--border);
}
.panel-title-row .panel-title-icon {
    color: var(--brand-700);
    display: flex; align-items: center; justify-content: center;
    flex-shrink: 0;
    filter: drop-shadow(0 2px 6px rgba(14, 116, 144, 0.35));
}
body.theme-dark .panel-title-row .panel-title-icon {
    color: #22d3ee;
    filter: drop-shadow(0 2px 10px rgba(34, 211, 238, 0.5));
}
body.lang-fa .panel-title-row { flex-direction: row-reverse; }
.panel-title-row h4 {
    margin: 0 !important;
    font-size: 1.35rem !important;
    color: var(--brand-700) !important;
    font-weight: 700; letter-spacing: 0.3px;
}
body.theme-dark .panel-title-row h4 { color: #22d3ee !important; }

.file-card {
    display: block; width: 100%;
    margin: 0 0 16px 0;
    padding: 16px 18px 18px 18px;
    background: var(--up-card-bg);
    border: 1.5px dashed var(--up-card-border);
    border-radius: 14px;
    transition: border-color 0.25s, background 0.25s, transform 0.2s, box-shadow 0.25s;
    position: relative;
}
.file-card:hover {
    border-color: var(--up-card-hover);
    box-shadow: 0 8px 24px rgba(6, 182, 212, 0.14);
    transform: translateY(-1px);
}
.file-card.drag-over {
    border-color: var(--brand-500);
    background: rgba(6, 182, 212, 0.06);
    transform: scale(1.01);
}
.file-card-head {
    display: flex; align-items: center; gap: 12px; margin-bottom: 12px;
}
body.lang-fa .file-card-head { flex-direction: row-reverse; }
.file-card-icon-box {
    width: 44px; height: 44px; border-radius: 12px;
    background: linear-gradient(135deg, rgba(6, 182, 212, 0.14) 0%,
                                        rgba(14, 116, 144, 0.20) 100%);
    display: flex; align-items: center; justify-content: center;
    color: var(--up-icon);
    flex-shrink: 0;
    border: 1px solid rgba(6, 182, 212, 0.28);
}
.file-card-titles {
    display: flex; flex-direction: column; gap: 3px;
    min-width: 0; flex: 1;
}
body.lang-fa .file-card-titles { align-items: flex-end; text-align: right; }
.file-card-title {
    font-weight: 700; font-size: 1rem;
    color: var(--text-main); letter-spacing: 0.2px;
    display: flex; align-items: center; gap: 8px; flex-wrap: wrap;
}
.file-card-hint {
    font-size: 0.82rem; color: var(--up-hint);
    font-weight: 400; letter-spacing: 0.15px;
}
.badge-optional {
    display: inline-block;
    padding: 2px 9px;
    font-size: 0.7rem; font-weight: 700;
    letter-spacing: 0.5px;
    text-transform: uppercase;
    color: var(--up-badge-text);
    background: var(--up-badge-bg);
    border-radius: 999px;
    border: 1px solid rgba(6, 182, 212, 0.28);
}
.file-card .shiny-input-container {
    width: 100% !important; margin: 0 !important; padding: 0 !important;
}
.file-card .input-group {
    display: flex !important; width: 100% !important;
    border-radius: 10px; overflow: hidden;
    border: 1px solid var(--border);
    background: var(--bg-panel-right);
    transition: border-color 0.2s, box-shadow 0.2s;
}
.file-card .input-group:focus-within {
    border-color: var(--brand-600);
    box-shadow: 0 0 0 3px rgba(6, 182, 212, 0.16);
}
.file-card .input-group-btn { display: flex !important; width: auto !important; }
.file-card .btn-file {
    display: inline-flex !important;
    align-items: center; justify-content: center;
    gap: 7px;
    background: var(--grad-btn) !important;
    color: #ffffff !important;
    border: none !important;
    padding: 0 20px !important;
    font-weight: 600 !important;
    font-size: 0.92rem !important;
    letter-spacing: 0.3px;
    cursor: pointer;
    height: 100%; min-height: 44px;
    transition: filter 0.2s, transform 0.15s;
    border-radius: 0 !important;
    white-space: nowrap;
}
.file-card .btn-file:hover { filter: brightness(1.1); color: #ffffff !important; }
.file-card .btn-file:active { transform: scale(0.98); }
.file-card .btn-file::before {
    content: '';
    display: inline-block; width: 16px; height: 16px;
    background-image: url("data:image/svg+xml;utf8,<svg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 24 24' fill='none' stroke='white' stroke-width='2.4' stroke-linecap='round' stroke-linejoin='round'><path d='M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4'/><polyline points='17 8 12 3 7 8'/><line x1='12' y1='3' x2='12' y2='15'/></svg>");
    background-size: contain; background-repeat: no-repeat; background-position: center;
    flex-shrink: 0;
}
.file-card .btn-file input[type="file"] {
    position: absolute; left: 0; top: 0; width: 100%; height: 100%;
    opacity: 0; cursor: pointer;
}
.file-card .form-control,
.file-card .form-control[readonly] {
    background: transparent !important;
    border: none !important;
    box-shadow: none !important;
    color: var(--text-main) !important;
    font-size: 0.92rem !important;
    padding: 12px 14px !important;
    height: auto !important; min-height: 44px;
    border-radius: 0 !important;
    flex: 1 1 auto; min-width: 0;
    outline: none !important;
}
.file-card .progress {
    margin-top: 12px !important; margin-bottom: 0 !important;
    height: 24px !important;
    background: rgba(6, 182, 212, 0.10) !important;
    border-radius: 999px !important;
    overflow: hidden;
    box-shadow: inset 0 1px 3px rgba(0,0,0,0.08);
    border: 1px solid rgba(6, 182, 212, 0.15);
}
.file-card .progress-bar {
    background: var(--grad-btn) !important;
    color: #ffffff !important;
    font-weight: 600 !important;
    font-size: 0.8rem !important;
    line-height: 24px !important;
    text-align: center !important;
    letter-spacing: 0.4px;
    border-radius: 999px !important;
    transition: width 0.4s ease;
    box-shadow: 0 1px 6px rgba(6, 182, 212, 0.4);
}
.file-card .progress-bar[aria-valuenow="100"] {
    background: linear-gradient(90deg, #059669 0%, #34d399 100%) !important;
    box-shadow: 0 1px 8px rgba(52, 211, 153, 0.55);
}
.file-card .progress-bar[aria-valuenow="100"]::before {
    content: '✓  '; font-weight: 800;
}

.btn-run-wrap { display: flex; margin: 4px 0 6px 0; }
.btn-run {
    display: inline-flex !important;
    align-items: center; justify-content: center; gap: 12px;
    width: 100%;
    background: var(--grad-btn) !important;
    color: white !important;
    border: none !important;
    padding: 15px 32px !important;
    font-size: 1.1rem !important;
    font-weight: 700 !important;
    letter-spacing: 0.4px;
    border-radius: 999px !important;
    cursor: pointer;
    transition: transform 0.18s, box-shadow 0.3s, filter 0.2s;
    box-shadow: 0 8px 22px rgba(8, 145, 178, 0.42);
    position: relative; overflow: hidden;
}
.btn-run:hover {
    transform: translateY(-2px);
    color: white !important;
    box-shadow: 0 14px 30px rgba(8, 145, 178, 0.55);
    filter: brightness(1.05);
    background: var(--grad-btn-hover) !important;
}
.btn-run:active { transform: translateY(0); }
.btn-run svg { flex-shrink: 0; }

/* ============ METRICS PANEL ============ */
.metrics-panel {
    margin-top: 20px;
    border-radius: 16px;
    padding: 22px;
    color: var(--text-main);
    border: 1px solid var(--border);
    background: var(--bg-panel-right);
    box-shadow: var(--shadow-panel);
    position: relative; overflow: hidden;
}
.metrics-panel::before {
    content: '';
    position: absolute; top: 0; left: 0;
    width: 100%; height: 4px;
    background: linear-gradient(90deg, #0891b2, #06b6d4, #22d3ee, #67e8f9);
    opacity: 0.9;
}
.metrics-header {
    display: flex; align-items: center; justify-content: flex-start;
    gap: 10px; margin-bottom: 18px;
    padding-bottom: 12px;
    border-bottom: 1px solid var(--border);
}
body.lang-fa .metrics-header { flex-direction: row-reverse; }
.metrics-header .metrics-icon-svg {
    color: var(--brand-600);
    flex-shrink: 0;
    filter: drop-shadow(0 2px 4px rgba(8, 145, 178, 0.3));
}
.metrics-header h4 {
    margin: 0; font-weight: 700;
    color: var(--text-main);
    font-size: 1.15rem; letter-spacing: 0.3px;
}
.metric-cards-grid {
    display: grid;
    grid-template-columns: repeat(3, minmax(0, 1fr));
    gap: 14px;
    margin-bottom: 22px;
}
.metric-cards-grid:last-child { margin-bottom: 0; }
.metric-card {
    position: relative;
    background: var(--bg-panel-right);
    border: 1px solid var(--border);
    border-radius: 14px;
    padding: 16px 16px 14px 16px;
    display: flex; flex-direction: column;
    gap: 6px;
    overflow: hidden;
    transition: transform 0.18s, border-color 0.2s, box-shadow 0.25s;
    box-shadow: 0 2px 8px rgba(15, 23, 42, 0.04);
}
body.theme-dark .metric-card {
    background: var(--mc-bg);
    border-color: var(--mc-border);
    box-shadow: 0 2px 12px rgba(0, 0, 0, 0.35);
}
.metric-card::before {
    content: '';
    position: absolute; top: 0; left: 0;
    width: 4px; height: 100%;
    background: var(--mc-accent-bar, linear-gradient(180deg, #06b6d4, #0891b2));
    opacity: 0.85;
}
.metric-card:hover {
    transform: translateY(-3px);
    box-shadow: 0 10px 28px rgba(6, 182, 212, 0.18);
    border-color: var(--mc-border-hover, rgba(6, 182, 212, 0.55));
}
.metric-card.variant-similarity {
    --mc-accent-bar: linear-gradient(180deg, #22d3ee, #0891b2);
    --mc-accent-icon-bg: linear-gradient(135deg, rgba(6,182,212,0.16), rgba(8,145,178,0.22));
    --mc-accent-icon-border: rgba(6,182,212,0.30);
    --mc-accent-icon-color: #0891b2;
    --mc-accent-value: #0e7490;
    --mc-accent-bar-fill: linear-gradient(90deg, #22d3ee, #0891b2);
    --mc-accent-bar-bg: rgba(6, 182, 212, 0.12);
}
body.theme-dark .metric-card.variant-similarity {
    --mc-accent-icon-color: #22d3ee;
    --mc-accent-value: #67e8f9;
}
.metric-card.variant-detection {
    --mc-accent-bar: linear-gradient(180deg, #a5b4fc, #4f46e5);
    --mc-accent-icon-bg: linear-gradient(135deg, rgba(99,102,241,0.16), rgba(79,70,229,0.22));
    --mc-accent-icon-border: rgba(99,102,241,0.30);
    --mc-accent-icon-color: #4f46e5;
    --mc-accent-value: #4338ca;
    --mc-accent-bar-fill: linear-gradient(90deg, #a5b4fc, #4f46e5);
    --mc-accent-bar-bg: rgba(99, 102, 241, 0.12);
}
body.theme-dark .metric-card.variant-detection {
    --mc-accent-icon-color: #a5b4fc;
    --mc-accent-value: #c7d2fe;
}
.metric-card.variant-distance {
    --mc-accent-bar: linear-gradient(180deg, #fda4af, #e11d48);
    --mc-accent-icon-bg: linear-gradient(135deg, rgba(244,63,94,0.16), rgba(225,29,72,0.22));
    --mc-accent-icon-border: rgba(244,63,94,0.30);
    --mc-accent-icon-color: #e11d48;
    --mc-accent-value: #be123c;
    --mc-accent-bar-fill: linear-gradient(90deg, #fda4af, #e11d48);
    --mc-accent-bar-bg: rgba(244, 63, 94, 0.12);
}
body.theme-dark .metric-card.variant-distance {
    --mc-accent-icon-color: #fda4af;
    --mc-accent-value: #fecdd3;
}
.metric-card.variant-slice {
    --mc-accent-bar: linear-gradient(180deg, #c4b5fd, #7c3aed);
    --mc-accent-icon-bg: linear-gradient(135deg, rgba(139,92,246,0.16), rgba(124,58,237,0.22));
    --mc-accent-icon-border: rgba(139,92,246,0.30);
    --mc-accent-icon-color: #7c3aed;
    --mc-accent-value: #6d28d9;
    --mc-accent-bar-fill: linear-gradient(90deg, #c4b5fd, #7c3aed);
    --mc-accent-bar-bg: rgba(139, 92, 246, 0.12);
}
body.theme-dark .metric-card.variant-slice {
    --mc-accent-icon-color: #c4b5fd;
    --mc-accent-value: #ddd6fe;
}
.metric-card-top {
    display: flex; align-items: center; justify-content: space-between;
    gap: 8px; margin-bottom: 2px;
}
.metric-card-icon {
    display: inline-flex; align-items: center; justify-content: center;
    width: 30px; height: 30px;
    border-radius: 9px;
    background: var(--mc-accent-icon-bg, linear-gradient(135deg, rgba(6,182,212,0.16), rgba(8,145,178,0.22)));
    color: var(--mc-accent-icon-color, #0891b2);
    flex-shrink: 0;
    border: 1px solid var(--mc-accent-icon-border, rgba(6,182,212,0.30));
}
.metric-card-label {
    flex: 1;
    font-size: 0.83rem; font-weight: 600;
    color: var(--text-muted);
    letter-spacing: 0.3px;
    text-align: right;
    white-space: nowrap;
    overflow: hidden; text-overflow: ellipsis;
}
body.lang-fa .metric-card-label { text-align: left; }
.metric-card-value {
    font-size: 1.75rem; font-weight: 800;
    color: var(--mc-accent-value, var(--text-main));
    line-height: 1.05;
    font-variant-numeric: tabular-nums;
    letter-spacing: -0.5px;
}
.metric-card-unit {
    font-size: 0.85rem; font-weight: 600;
    color: var(--text-muted);
    margin-left: 4px; opacity: 0.75;
}
.metric-card-bar {
    height: 6px;
    background: var(--mc-accent-bar-bg, rgba(6, 182, 212, 0.12));
    border-radius: 999px; overflow: hidden;
    margin-top: 4px;
}
.metric-card-bar-fill {
    height: 100%;
    background: var(--mc-accent-bar-fill, linear-gradient(90deg, #22d3ee, #0891b2));
    border-radius: 999px;
    transition: width 0.5s ease;
    box-shadow: 0 0 8px rgba(6, 182, 212, 0.5);
}
.metric-cards-empty {
    text-align: center; padding: 22px 12px;
    color: var(--text-muted); font-size: 0.98rem;
    font-style: italic;
    border: 1px dashed var(--border-strong);
    border-radius: 12px;
    background: rgba(6, 182, 212, 0.04);
}
@media (max-width: 700px) {
    .metric-cards-grid { grid-template-columns: repeat(2, 1fr); gap: 10px; }
    .metric-card { padding: 12px 12px 10px 12px; }
    .metric-card-label { font-size: 0.78rem; }
    .metric-card-value { font-size: 1.35rem; }
    .metric-card-icon { width: 24px; height: 24px; }
}

/* ============ DICOM INFO STRIP ============ */
.dicom-info-strip {
    display: flex; align-items: center; gap: 12px;
    background: linear-gradient(135deg, #0c4a6e 0%, #075985 60%, #0e7490 100%);
    border: 1px solid rgba(103, 232, 249, 0.28);
    border-radius: 12px; padding: 12px 18px;
    margin-bottom: 14px; flex-wrap: wrap;
    box-shadow: 0 6px 18px rgba(12, 74, 110, 0.28);
    justify-content: center;
    text-align: center;
}
.dicom-info-strip .info-icon {
    color: #67e8f9;
    display: flex; align-items: center; justify-content: center;
    flex-shrink: 0;
    filter: drop-shadow(0 0 8px rgba(103, 232, 249, 0.5));
}
body.lang-fa .dicom-info-strip { flex-direction: row-reverse; }
.dicom-info-strip .info-item {
    display: inline-flex; align-items: baseline; gap: 6px;
    color: #cffafe; font-size: 0.95rem; white-space: nowrap;
    text-align: center;
}
.dicom-info-strip .info-label { color: #a5f3fc; font-weight: 500; opacity: 0.85; }
.dicom-info-strip .info-value {
    color: #ffffff; font-weight: 700;
    font-variant-numeric: tabular-nums;
}
.dicom-info-strip .info-sep { color: #22d3ee; font-weight: 700; user-select: none; opacity: 0.5; }
.dicom-info-strip .info-empty {
    color: #a5f3fc; font-style: italic; font-size: 0.92rem;
    width: 100%; text-align: center; padding: 4px 0;
}
.dicom-info-strip .info-value.info-time {
    color: #fef08a;
    text-shadow: 0 0 10px rgba(254, 240, 138, 0.6);
}

/* ============ RESULT PANELS ============ */
.result-grid { margin-bottom: 8px; }
.result-grid [class*="col-"] { padding-left: 8px; padding-right: 8px; }
.result-panel {
    position: relative;
    background: #0f172a;
    border: 1px solid #1e293b;
    border-radius: 14px; overflow: hidden; margin-bottom: 16px;
    display: flex; flex-direction: column;
    box-shadow: 0 8px 24px rgba(2, 6, 23, 0.4);
}

.result-panel .panel-loading-overlay {
    position: absolute;
    inset: 0;
    display: none;
    align-items: center;
    justify-content: center;
    background: rgba(15, 23, 42, 0.55);
    backdrop-filter: blur(2px);
    -webkit-backdrop-filter: blur(2px);
    z-index: 30;
    pointer-events: none;
    border-radius: 14px;
}
.result-panel.is-loading .panel-loading-overlay {
    display: flex;
}

.result-panel .panel-spinner-lg {
    width: 52px;
    height: 52px;
    border: 4px solid rgba(34, 211, 238, 0.22);
    border-top-color: #22d3ee;
    border-radius: 50%;
    animation: spinCyan 0.75s linear infinite;
    flex-shrink: 0;
    box-shadow: 0 0 26px rgba(34, 211, 238, 0.65),
                0 0 6px rgba(34, 211, 238, 0.9) inset;
}

@keyframes spinCyan {
    to { transform: rotate(360deg); }
}

.result-panel .recalculating::after,
.result-panel .recalculating::before,
.result-panel-image .recalculating::after,
.result-panel-image .recalculating::before {
    display: none !important;
}
.result-panel .recalculating,
.result-panel-image .recalculating {
    background-image: none !important;
    opacity: 1 !important;
}

.result-panel-header {
    display: flex; align-items: center; justify-content: space-between;
    gap: 10px; padding: 10px 14px;
    background: linear-gradient(135deg, #1e293b 0%, #0f172a 100%);
    border-bottom: 1px solid #1e293b;
    min-height: 44px; flex-wrap: wrap;
}
.result-panel-title {
    font-size: 0.98rem; font-weight: 600;
    color: #e2e8f0; letter-spacing: 0.3px;
    flex-shrink: 0;
    display: inline-flex; align-items: center; gap: 8px;
}
.result-panel-title::before {
    content: '';
    width: 3px; height: 14px;
    border-radius: 2px;
    background: linear-gradient(180deg, #22d3ee, #0891b2);
    flex-shrink: 0;
}
.result-panel-legend {
    display: flex; align-items: center; justify-content: flex-end;
    gap: 10px; flex-wrap: wrap;
    margin-left: auto; flex: 1 1 auto;
}
.result-panel-legend:empty { display: none; }
.legend-item {
    display: inline-flex; align-items: center; gap: 5px;
    font-size: 0.72rem; color: #cbd5e1;
    white-space: nowrap; font-weight: 500;
    letter-spacing: 0.2px; line-height: 1;
}
.legend-dot {
    width: 9px; height: 9px; border-radius: 50%;
    display: inline-block; flex-shrink: 0;
    border: 1px solid rgba(255,255,255,0.4);
    box-shadow: 0 0 6px rgba(0,0,0,0.4);
}
.result-panel-close {
    background: transparent; border: none;
    color: #64748b; font-size: 1.4rem;
    cursor: pointer; padding: 0 4px; line-height: 1;
    flex-shrink: 0;
    transition: color 0.15s;
}
.result-panel-close:hover { color: #22d3ee; }
.result-panel-toolbar {
    display: flex; align-items: center; justify-content: space-between;
    gap: 12px; padding: 8px 14px;
    background: #0f172a;
    border-bottom: 1px solid rgba(30, 41, 59, 0.6);
}
.result-toolbar-left { display: flex; align-items: center; gap: 14px; }
.result-panel-controls { display: flex; gap: 6px; flex-shrink: 0; }
.result-toolbar-left .shiny-text-output {
    color: #ffffff !important;
    font-size: 0.95rem !important;
    font-weight: 600 !important;
    letter-spacing: 0.4px;
    font-variant-numeric: tabular-nums;
    text-shadow: 0 1px 3px rgba(0, 0, 0, 0.7);
    white-space: nowrap;
}
.ctrl-btn {
    background: rgba(34, 211, 238, 0.10);
    border: 1px solid rgba(34, 211, 238, 0.30);
    color: #67e8f9;
    width: 28px; height: 28px; border-radius: 6px;
    cursor: pointer;
    display: flex; align-items: center; justify-content: center;
    padding: 0;
    transition: background 0.15s, color 0.15s, border-color 0.15s, transform 0.1s;
}
.ctrl-btn:hover {
    background: rgba(34, 211, 238, 0.28);
    border-color: rgba(103, 232, 249, 0.85);
    color: #ffffff;
}
.ctrl-btn:active { transform: scale(0.94); }
.ctrl-btn svg { display: block; }

.result-panel-image {
    background: #0f172a;
    padding: 18px;
    display: flex; align-items: center; justify-content: center;
    width: 100%; min-height: 200px;
    overflow: hidden; position: relative; cursor: default;
}
.result-panel-image .shiny-html-output {
    display: flex; align-items: center; justify-content: center;
    width: 100% !important; padding: 0 !important; margin: 0 !important;
    overflow: hidden;
}
.slice-img {
    display: block; width: 100%;
    max-width: 380px; height: auto;
    border-radius: 8px; background: #000;
    transform-origin: center center;
    transition: transform 0.15s ease-out;
    will-change: transform;
    user-select: none;
    -webkit-user-drag: none;
    pointer-events: auto;
}
.slice-img.panning { transition: none !important; cursor: grabbing !important; }
.slice-error {
    color: #94a3b8; font-size: 0.95rem;
    text-align: center; padding: 60px 20px;
    font-style: italic; max-width: 380px;
}
.result-panel-image { min-height: 380px; }
.result-panel:fullscreen {
    background: #000;
    width: 100vw; height: 100vh;
    display: flex; flex-direction: column;
    border-radius: 0;
}
.result-panel:fullscreen .result-panel-image {
    flex: 1; min-height: 0; border-radius: 0;
}
.result-panel:fullscreen img.slice-img { max-width: none; max-height: 100%; }

/* ============ FILMSTRIP ============ */
.filmstrip-section {
    background: #0f172a;
    border: 1px solid #1e293b;
    border-radius: 14px;
    padding: 14px 18px 20px 18px;
    margin-bottom: 20px;
    box-shadow: 0 8px 22px rgba(2, 6, 23, 0.4);
}
.filmstrip-title {
    display: flex; align-items: center; justify-content: space-between;
    gap: 14px; margin-bottom: 18px;
    color: #e2e8f0; flex-wrap: wrap;
}
.filmstrip-title-left { display: flex; align-items: center; gap: 10px; }
.filmstrip-title-left .filmstrip-icon {
    color: #22d3ee; display: flex;
    filter: drop-shadow(0 0 6px rgba(34, 211, 238, 0.5));
}
.filmstrip-title-left span { font-size: 1rem; font-weight: 600; }
.jump-to-slice { display: flex; align-items: center; gap: 8px; }
.jump-to-slice .shiny-input-container {
    margin: 0 !important; padding: 0 !important; width: auto !important;
}
#jump_input {
    background: #0b1220 !important;
    color: #e2e8f0 !important;
    border: 1px solid #1e293b !important;
    border-radius: 8px !important;
    padding: 7px 12px !important;
    font-size: 0.92rem !important;
    width: 96px !important;
    text-align: center !important;
    margin: 0 !important;
    transition: border-color 0.15s, box-shadow 0.15s;
}
#jump_input::placeholder { color: #64748b !important; font-size: 0.85rem; }
#jump_input:focus {
    outline: none !important;
    border-color: #22d3ee !important;
    box-shadow: 0 0 0 2px rgba(34, 211, 238, 0.25) !important;
}
.jump-btn {
    background: rgba(34, 211, 238, 0.15) !important;
    border: 1px solid rgba(34, 211, 238, 0.40) !important;
    color: #67e8f9 !important;
    border-radius: 8px !important;
    padding: 7px 16px !important;
    font-size: 0.9rem !important;
    font-weight: 600 !important;
    cursor: pointer;
    margin: 0 !important;
    transition: background 0.15s, color 0.15s;
}
.jump-btn:hover {
    background: rgba(34, 211, 238, 0.35) !important;
    color: #ffffff !important;
}
.filmstrip { display: flex; align-items: center; gap: 10px; }
.filmstrip-arrow {
    background: transparent; border: none;
    color: #22d3ee; font-size: 42px; line-height: 1;
    cursor: pointer; padding: 0 16px;
    flex-shrink: 0; user-select: none;
    transition: color 0.15s, transform 0.15s;
}
.filmstrip-arrow:hover:not(:disabled) {
    color: #ffffff;
    transform: scale(1.2);
    filter: drop-shadow(0 0 8px rgba(34, 211, 238, 0.7));
}
.filmstrip-arrow:disabled { opacity: 0.25; cursor: not-allowed; }
.filmstrip-track {
    display: flex; gap: 14px; flex: 1;
    justify-content: center; overflow: hidden; padding: 6px 0;
}
.filmstrip-item {
    display: flex; flex-direction: column; align-items: center;
    gap: 10px; cursor: pointer; flex-shrink: 0;
    transition: transform 0.15s ease; user-select: none;
}
.filmstrip-item:hover { transform: translateY(-3px); }
.filmstrip-item img {
    width: 130px; height: 130px;
    border-radius: 10px; border: 2px solid #1e293b;
    display: block; object-fit: cover; background: #000;
    transition: border-color 0.15s;
}
.filmstrip-item:hover img { border-color: #22d3ee; }
.filmstrip-item.current img {
    border-color: #22d3ee;
    box-shadow: 0 0 0 2px rgba(34, 211, 238, 0.45),
                0 0 22px rgba(34, 211, 238, 0.75);
}
.filmstrip-num {
    font-size: 1.02rem; color: #94a3b8; font-weight: 500;
    font-variant-numeric: tabular-nums; line-height: 1;
}
.filmstrip-item.current .filmstrip-num { color: #22d3ee; font-weight: 700; }
.filmstrip-empty {
    text-align: center; color: #64748b;
    padding: 40px 12px; font-style: italic; flex: 1;
}
.filmstrip-ellipsis {
    display: flex; align-items: center; justify-content: center;
    width: 130px; height: 130px; border-radius: 10px;
    border: 2px dashed #1e293b;
    color: #475569; font-size: 1.8rem; font-weight: 700;
    flex-shrink: 0; user-select: none;
}
@media (max-width: 1300px) {
    .filmstrip-item img { width: 112px; height: 112px; }
    .filmstrip-ellipsis { width: 112px; height: 112px; font-size: 1.5rem; }
    .filmstrip-arrow { font-size: 36px; padding: 0 12px; }
}
@media (max-width: 1050px) {
    .filmstrip-item img { width: 92px; height: 92px; }
    .filmstrip-ellipsis { width: 92px; height: 92px; font-size: 1.3rem; }
    .filmstrip-arrow { font-size: 30px; padding: 0 8px; }
}
@media (max-width: 800px) {
    .filmstrip-item img { width: 72px; height: 72px; }
    .filmstrip-ellipsis { width: 72px; height: 72px; font-size: 1.1rem; }
    .filmstrip-arrow { font-size: 26px; padding: 0 6px; }
    .result-panel-image { padding: 10px; }
}

/* ============ 3D VIEW ============ */
.view3d-wrapper {
    position: relative;
    overflow: hidden;
    border-radius: 18px;
    padding: 22px;
    margin-bottom: 20px;
    transition: background 0.35s ease, border-color 0.35s ease, box-shadow 0.35s ease;
}
body.theme-light .view3d-wrapper {
    background: linear-gradient(135deg, #f8fafc 0%, #e0f2fe 100%);
    border: 1px solid #bae6fd;
    box-shadow: 0 14px 40px rgba(8, 145, 178, 0.10);
}
body.theme-light .view3d-wrapper::before {
    content: '';
    position: absolute;
    top: -100px; right: -100px;
    width: 300px; height: 300px;
    background: radial-gradient(circle, rgba(251, 113, 133, 0.14), transparent 70%);
    pointer-events: none;
}
body.theme-dark .view3d-wrapper {
    background: linear-gradient(135deg, #0f172a 0%, #0c4a6e 100%);
    border: 1px solid #1e293b;
    box-shadow: 0 14px 40px rgba(2, 6, 23, 0.55);
}
body.theme-dark .view3d-wrapper::before {
    content: '';
    position: absolute;
    top: -100px; right: -100px;
    width: 300px; height: 300px;
    background: radial-gradient(circle, rgba(251, 113, 133, 0.18), transparent 70%);
    pointer-events: none;
}
.view3d-header {
    display: flex; align-items: center; gap: 14px;
    padding-bottom: 16px; margin-bottom: 18px;
    flex-wrap: wrap;
    position: relative; z-index: 1;
    transition: border-color 0.35s ease;
}
body.theme-light .view3d-header { border-bottom: 1px solid #bae6fd; }
body.theme-dark  .view3d-header { border-bottom: 1px solid rgba(34, 211, 238, 0.22); }
body.lang-fa .view3d-header { flex-direction: row-reverse; }
.view3d-header-icon {
    width: 50px; height: 50px;
    border-radius: 14px;
    display: flex; align-items: center; justify-content: center;
    flex-shrink: 0;
    transition: all 0.35s ease;
}
body.theme-light .view3d-header-icon {
    background: linear-gradient(135deg, rgba(6, 182, 212, 0.18) 0%,
                                        rgba(251, 113, 133, 0.16) 100%);
    color: #0e7490;
    border: 1px solid rgba(6, 182, 212, 0.35);
    box-shadow: 0 4px 18px rgba(6, 182, 212, 0.22);
}
body.theme-dark .view3d-header-icon {
    background: linear-gradient(135deg, rgba(34, 211, 238, 0.22) 0%,
                                        rgba(251, 113, 133, 0.20) 100%);
    color: #67e8f9;
    border: 1px solid rgba(34, 211, 238, 0.40);
    box-shadow: 0 0 24px rgba(34, 211, 238, 0.30);
}
.view3d-header-text { display: flex; flex-direction: column; gap: 3px; }
body.lang-fa .view3d-header-text { align-items: flex-end; text-align: right; }
.view3d-header-text h3 {
    margin: 0; font-size: 1.35rem; font-weight: 700;
    letter-spacing: 0.3px;
    transition: color 0.35s ease;
}
body.theme-light .view3d-header-text h3 { color: #0c4a6e; }
body.theme-dark  .view3d-header-text h3 { color: #e2e8f0; }
.view3d-header-text .view3d-hint {
    font-size: 0.85rem; letter-spacing: 0.2px;
    transition: color 0.35s ease;
}
body.theme-light .view3d-header-text .view3d-hint { color: #0891b2; }
body.theme-dark  .view3d-header-text .view3d-hint { color: #67e8f9; }
.view3d-body {
    display: grid; grid-template-columns: 320px 1fr; gap: 20px;
    position: relative; z-index: 1;
}
@media (max-width: 1000px) { .view3d-body { grid-template-columns: 1fr; } }
.view3d-controls {
    border-radius: 14px;
    padding: 18px;
    display: flex; flex-direction: column; gap: 16px;
    max-height: calc(100vh - 220px);
    overflow-y: auto;
    transition: background 0.35s ease, border-color 0.35s ease;
}
body.theme-light .view3d-controls {
    background: linear-gradient(145deg, #ffffff 0%, #f0f9ff 100%);
    border: 1px solid #bae6fd;
    box-shadow: 0 4px 20px rgba(8, 145, 178, 0.08);
}
body.theme-dark .view3d-controls {
    background: linear-gradient(145deg, #0b1220 0%, #070d1a 100%);
    border: 1px solid rgba(34, 211, 238, 0.20);
}
.view3d-controls::-webkit-scrollbar { width: 8px; }
body.theme-light .view3d-controls::-webkit-scrollbar-track {
    background: rgba(186, 230, 253, 0.4); border-radius: 4px;
}
body.theme-light .view3d-controls::-webkit-scrollbar-thumb {
    background: rgba(8, 145, 178, 0.35); border-radius: 4px;
}
body.theme-light .view3d-controls::-webkit-scrollbar-thumb:hover {
    background: rgba(8, 145, 178, 0.6);
}
body.theme-dark .view3d-controls::-webkit-scrollbar-track {
    background: rgba(2, 6, 23, 0.4); border-radius: 4px;
}
body.theme-dark .view3d-controls::-webkit-scrollbar-thumb {
    background: rgba(34, 211, 238, 0.35); border-radius: 4px;
}
body.theme-dark .view3d-controls::-webkit-scrollbar-thumb:hover {
    background: rgba(34, 211, 238, 0.6);
}
.view3d-control-group {
    display: flex;
    flex-direction: column;
    gap: 6px;
    padding: 14px;
    border-radius: 12px;
    transition: all 0.35s ease;
}
body.theme-light .view3d-control-group {
    background: #ffffff;
    border: 1px solid #e0f2fe;
    box-shadow: 0 1px 3px rgba(8, 145, 178, 0.06);
}
body.theme-light .view3d-control-group:hover {
    border-color: rgba(8, 145, 178, 0.4);
    box-shadow: 0 4px 18px rgba(8, 145, 178, 0.12);
}
body.theme-dark .view3d-control-group {
    background: linear-gradient(145deg, rgba(2, 6, 23, 0.8), rgba(15, 23, 42, 0.6));
    border: 1px solid rgba(34, 211, 238, 0.15);
    box-shadow: inset 0 1px 3px rgba(0,0,0,0.2);
}
body.theme-dark .view3d-control-group:hover {
    border-color: rgba(34, 211, 238, 0.4);
    box-shadow: 0 4px 20px rgba(34, 211, 238, 0.1), inset 0 1px 3px rgba(0,0,0,0.2);
}
.view3d-control-group .control-label {
    font-size: 0.78rem;
    font-weight: 700;
    letter-spacing: 0.5px;
    text-transform: uppercase;
    margin-bottom: 4px;
    transition: color 0.35s ease;
}
body.theme-light .view3d-control-group .control-label { color: #0e7490; }
body.theme-dark  .view3d-control-group .control-label { color: #67e8f9; }
.view3d-toggle-row {
    display: flex; align-items: center; gap: 10px;
}
.view3d-toggle-row .form-check,
.view3d-toggle-row .shiny-input-container {
    margin: 0 !important; padding: 0 !important; width: 100% !important;
}
.view3d-toggle-row label {
    font-weight: 500 !important;
    font-size: 0.92rem !important;
    cursor: pointer; user-select: none;
    display: flex; align-items: center; gap: 10px;
    transition: color 0.35s ease;
}
body.theme-light .view3d-toggle-row label { color: #1e293b !important; }
body.theme-dark  .view3d-toggle-row label { color: #e2e8f0 !important; }
.view3d-toggle-row input[type="checkbox"] {
    width: 16px; height: 16px;
    accent-color: #06b6d4;
    cursor: pointer; margin: 0 !important;
}
.view3d-color-dot {
    display: inline-block;
    width: 12px; height: 12px;
    border-radius: 50%;
    flex-shrink: 0;
    box-shadow: 0 0 8px currentColor;
    border: 1px solid rgba(255,255,255,0.35);
}
body.theme-light .view3d-color-dot {
    border-color: rgba(15, 23, 42, 0.15);
}
.view3d-metrics-wrap { margin-top: 4px; }
.view3d-control-group .metric-cards-grid {
    grid-template-columns: repeat(2, minmax(0, 1fr));
    gap: 8px;
    margin-bottom: 0;
}
.view3d-control-group .metric-card {
    padding: 10px 10px 8px 10px;
    border-radius: 10px;
    transition: all 0.3s ease;
}
body.theme-light .view3d-control-group .metric-card {
    background: #f8fafc;
    border-color: #e2e8f0;
    box-shadow: 0 1px 2px rgba(15, 23, 42, 0.04);
}
body.theme-light .view3d-control-group .metric-card:hover {
    transform: translateY(-1px);
    background: #ffffff;
    box-shadow: 0 4px 14px rgba(8, 145, 178, 0.18);
}
body.theme-dark .view3d-control-group .metric-card {
    background: rgba(2, 6, 23, 0.55);
    border-color: rgba(34, 211, 238, 0.18);
}
body.theme-dark .view3d-control-group .metric-card:hover {
    transform: translateY(-1px);
    box-shadow: 0 4px 14px rgba(34, 211, 238, 0.22);
}
.view3d-control-group .metric-card-icon {
    width: 22px; height: 22px; border-radius: 6px;
}
.view3d-control-group .metric-card-icon svg {
    width: 12px !important; height: 12px !important;
}
.view3d-control-group .metric-card-label {
    font-size: 0.68rem; letter-spacing: 0.15px;
}
.view3d-control-group .metric-card-value {
    font-size: 1.05rem; letter-spacing: -0.2px;
}
.view3d-control-group .metric-card-unit { font-size: 0.65rem; }
.view3d-control-group .metric-card-bar { height: 4px; margin-top: 3px; }
.view3d-control-group .metric-cards-empty {
    padding: 14px 10px;
    font-size: 0.82rem;
}
.view3d-info {
    border-radius: 10px;
    padding: 12px 14px;
    display: flex; flex-direction: column; gap: 8px;
    transition: all 0.35s ease;
}
body.theme-light .view3d-info {
    background: #ffffff;
    border: 1px solid #e0f2fe;
}
body.theme-dark .view3d-info {
    background: rgba(2, 6, 23, 0.6);
    border: 1px solid rgba(34, 211, 238, 0.16);
}
.view3d-info-title {
    font-size: 0.78rem; font-weight: 700;
    letter-spacing: 0.5px;
    text-transform: uppercase; margin-bottom: 2px;
    transition: color 0.35s ease;
}
body.theme-light .view3d-info-title { color: #0e7490; }
body.theme-dark  .view3d-info-title { color: #67e8f9; }
.view3d-info-row {
    display: flex; justify-content: space-between;
    align-items: baseline; gap: 10px;
    font-size: 0.85rem;
    transition: color 0.35s ease;
}
body.theme-light .view3d-info-row .k { color: #64748b; }
body.theme-light .view3d-info-row .v { color: #0f172a; font-weight: 700; font-variant-numeric: tabular-nums; }
body.theme-dark  .view3d-info-row .k { color: #94a3b8; }
body.theme-dark  .view3d-info-row .v { color: #ffffff; font-weight: 700; font-variant-numeric: tabular-nums; }
.view3d-hint-rotate {
    display: flex; align-items: center; gap: 8px;
    font-size: 0.78rem;
    padding-top: 8px;
    font-style: italic;
    transition: all 0.35s ease;
}
body.theme-light .view3d-hint-rotate {
    color: #64748b;
    border-top: 1px solid #e0f2fe;
}
body.theme-dark .view3d-hint-rotate {
    color: #64748b;
    border-top: 1px solid rgba(34, 211, 238, 0.12);
}
.view3d-hint-rotate svg { color: #06b6d4; flex-shrink: 0; }
.view3d-canvas-wrap {
    border-radius: 14px;
    overflow: hidden;
    min-height: 500px;
    position: relative;
    display: flex; align-items: center; justify-content: center;
    padding: 12px;
    background: radial-gradient(circle at 50% 30%, #0c4a6e 0%, #020617 100%);
    border: 1px solid rgba(34, 211, 238, 0.22);
    box-shadow: inset 0 0 80px rgba(34, 211, 238, 0.08);
    transition: box-shadow 0.35s ease;
}
body.theme-light .view3d-canvas-wrap {
    box-shadow: 0 4px 20px rgba(8, 145, 178, 0.18),
                inset 0 0 80px rgba(34, 211, 238, 0.08);
}
.view3d-canvas-wrap .shiny-html-output {
    width: 100% !important;
    padding: 0 !important; margin: 0 !important;
    display: flex; justify-content: center; align-items: center;
}
.view3d-image {
    display: block; width: 100%; max-width: 100%; height: auto;
    border-radius: 10px; background: #020617;
    box-shadow: 0 4px 28px rgba(0,0,0,0.5);
}
.view3d-empty {
    display: flex; flex-direction: column;
    align-items: center; justify-content: center;
    min-height: 500px; gap: 14px;
    text-align: center; padding: 30px;
    transition: color 0.35s ease;
}
body.theme-light .view3d-empty { color: #64748b; }
body.theme-dark  .view3d-empty { color: #94a3b8; }
.view3d-empty svg { color: #06b6d4; opacity: 0.7; }
.view3d-empty .msg {
    font-size: 1rem; font-style: italic;
    max-width: 380px; line-height: 1.6;
}

/* ============ SPLASH SCREEN ============ */
.splash-screen {
    position: fixed;
    inset: 0;
    z-index: 9999;
    display: flex;
    flex-direction: column;
    align-items: center;
    justify-content: center;
    gap: 22px;
    overflow: hidden;
    transition: opacity 0.7s ease, visibility 0.7s ease;
    will-change: opacity;
}
body.theme-light .splash-screen {
    background: linear-gradient(135deg, #f8fafc 0%, #e0f2fe 55%, #bae6fd 100%);
}
body.theme-dark .splash-screen {
    background: linear-gradient(135deg, #020617 0%, #0c4a6e 55%, #0e7490 100%);
}
.splash-screen.hide {
    opacity: 0;
    visibility: hidden;
    pointer-events: none;
}
.splash-blob {
    position: absolute;
    border-radius: 50%;
    filter: blur(60px);
    pointer-events: none;
    animation: splashBlob 6s ease-in-out infinite alternate;
}
.splash-blob-1 {
    top: -120px; right: -120px;
    width: 420px; height: 420px;
}
.splash-blob-2 {
    bottom: -140px; left: -140px;
    width: 460px; height: 460px;
    animation-delay: -3s;
}
body.theme-light .splash-blob-1 { background: rgba(6, 182, 212, 0.30); }
body.theme-light .splash-blob-2 { background: rgba(251, 113, 133, 0.22); }
body.theme-dark  .splash-blob-1 { background: rgba(34, 211, 238, 0.22); }
body.theme-dark  .splash-blob-2 { background: rgba(251, 113, 133, 0.18); }
@keyframes splashBlob {
    0%   { transform: translate(0, 0) scale(1); }
    100% { transform: translate(30px, -20px) scale(1.12); }
}
.splash-content {
    position: relative;
    z-index: 2;
    display: flex;
    flex-direction: column;
    align-items: center;
    justify-content: center;
    text-align: center;
    gap: 18px;
    padding: 20px;
    animation: splashContentIn 0.9s cubic-bezier(0.22, 1, 0.36, 1) both;
}
@keyframes splashContentIn {
    0%   { opacity: 0; transform: translateY(18px) scale(0.94); }
    100% { opacity: 1; transform: translateY(0)    scale(1); }
}
.splash-title {
    font-size: 2.9rem;
    font-weight: 800;
    margin: 0;
    letter-spacing: 0.4px;
    line-height: 1.15;
    transition: color 0.35s ease;
}
body.theme-light .splash-title {
    color: #0c4a6e;
    text-shadow: 0 2px 24px rgba(8, 145, 178, 0.25);
}
body.theme-dark .splash-title {
    color: #ffffff;
    text-shadow: 0 2px 28px rgba(34, 211, 238, 0.45);
}
.splash-subtitle {
    font-size: 1.05rem;
    font-weight: 500;
    letter-spacing: 0.8px;
    margin: 0;
    transition: color 0.35s ease;
}
body.theme-light .splash-subtitle { color: #0891b2; }
body.theme-dark  .splash-subtitle { color: #a5f3fc; }
.splash-gallery {
    position: relative;
    width: min(620px, 68vmin);
    height: min(620px, 68vmin);
    margin-top: 16px;
    margin-bottom: 14px;
    display: flex;
    align-items: center;
    justify-content: center;
    perspective: 1600px;
}
.splash-gallery-item {
    position: absolute;
    inset: 0;
    border-radius: 46px;
    overflow: hidden;
    display: flex;
    align-items: center;
    justify-content: center;
    opacity: 0;
    transform: scale(0.70) rotateY(-30deg);
    transition: opacity 1.0s cubic-bezier(0.22, 1, 0.36, 1),
                transform 1.1s cubic-bezier(0.22, 1, 0.36, 1),
                box-shadow 0.8s ease,
                filter 0.8s ease;
    will-change: opacity, transform, box-shadow, filter;
    pointer-events: none;
    filter: blur(8px) saturate(0.65) brightness(0.82);
}
body.theme-light .splash-gallery-item {
    background: radial-gradient(circle at 50% 30%,
                                rgba(6, 182, 212, 0.10),
                                rgba(8, 145, 178, 0.24));
    border: 4px solid rgba(6, 182, 212, 0.45);
    box-shadow: 0 26px 80px rgba(8, 145, 178, 0.30),
                0 0 50px rgba(6, 182, 212, 0.22),
                inset 0 0 40px rgba(6, 182, 212, 0.16);
}
body.theme-dark .splash-gallery-item {
    background: radial-gradient(circle at 50% 30%,
                                rgba(34, 211, 238, 0.14),
                                rgba(8, 145, 178, 0.34));
    border: 4px solid rgba(34, 211, 238, 0.55);
    box-shadow: 0 26px 90px rgba(34, 211, 238, 0.42),
                0 0 70px rgba(34, 211, 238, 0.30),
                inset 0 0 46px rgba(34, 211, 238, 0.24);
}
.splash-gallery-item.active {
    opacity: 1;
    filter: blur(0) saturate(1.20) brightness(1.06);
    transform: scale(1) rotateY(0deg);
    pointer-events: auto;
    animation: galleryActivePulse 2.2s ease-in-out infinite;
}
@keyframes galleryActivePulse {
    0%, 100% {
        box-shadow: 0 26px 80px rgba(34, 211, 238, 0.45),
                    0 0 55px rgba(34, 211, 238, 0.30),
                    inset 0 0 40px rgba(34, 211, 238, 0.22);
        filter: blur(0) saturate(1.20) brightness(1.06);
    }
    50% {
        box-shadow: 0 34px 105px rgba(34, 211, 238, 0.75),
                    0 0 90px rgba(34, 211, 238, 0.50),
                    inset 0 0 54px rgba(34, 211, 238, 0.38);
        filter: blur(0) saturate(1.30) brightness(1.14);
    }
}
.splash-gallery-item img {
    width: 100%;
    height: 100%;
    object-fit: cover;
    display: block;
    user-select: none;
    -webkit-user-drag: none;
    pointer-events: none;
}
.splash-gallery-item::after {
    content: '';
    position: absolute;
    inset: 0;
    border-radius: 42px;
    background: linear-gradient(120deg, transparent 35%,
                                rgba(255, 255, 255, 0.30) 50%,
                                transparent 65%);
    transform: translateX(-140%);
    pointer-events: none;
}
.splash-gallery-item.active::after {
    animation: galleryShine 2.0s ease-out;
}
@keyframes galleryShine {
    0%   { transform: translateX(-140%); }
    100% { transform: translateX(140%); }
}
.splash-gallery-dots {
    display: inline-flex;
    gap: 14px;
    margin-top: 6px;
    margin-bottom: 6px;
}
.splash-gallery-dots span {
    width: 13px;
    height: 13px;
    border-radius: 50%;
    transition: all 0.45s cubic-bezier(0.22, 1, 0.36, 1);
}
body.theme-light .splash-gallery-dots span {
    background: rgba(8, 145, 178, 0.22);
    border: 1px solid rgba(8, 145, 178, 0.40);
}
body.theme-dark .splash-gallery-dots span {
    background: rgba(34, 211, 238, 0.20);
    border: 1px solid rgba(34, 211, 238, 0.40);
}
.splash-gallery-dots span.active {
    transform: scale(1.55);
}
body.theme-light .splash-gallery-dots span.active {
    background: #0891b2;
    box-shadow: 0 0 18px rgba(8, 145, 178, 0.85),
                0 0 32px rgba(8, 145, 178, 0.45);
    border-color: #0891b2;
}
body.theme-dark .splash-gallery-dots span.active {
    background: #22d3ee;
    box-shadow: 0 0 22px rgba(34, 211, 238, 0.95),
                0 0 40px rgba(34, 211, 238, 0.55);
    border-color: #22d3ee;
}
.splash-quote-wrap {
    margin-top: 14px;
    margin-bottom: 4px;
    display: flex;
    flex-direction: column;
    align-items: center;
    gap: 10px;
    animation: splashQuoteIn 1.1s ease-out 0.4s both;
}
@keyframes splashQuoteIn {
    0%   { opacity: 0; transform: translateY(8px); letter-spacing: 3px; }
    100% { opacity: 1; transform: translateY(0);   letter-spacing: normal; }
}
.splash-quote-divider {
    width: 140px;
    height: 2px;
    border-radius: 2px;
    opacity: 0.75;
    background: linear-gradient(90deg,
                transparent 0%,
                currentColor 50%,
                transparent 100%);
}
body.theme-light .splash-quote-divider { color: #0891b2; }
body.theme-dark  .splash-quote-divider { color: #22d3ee; }
.splash-quote {
    font-family: 'Playfair Display', 'Georgia', 'Times New Roman', serif;
    font-size: 1.5rem;
    font-weight: 500;
    font-style: italic;
    letter-spacing: 0.4px;
    line-height: 1.4;
    margin: 0;
    padding: 0 16px;
    max-width: 700px;
    text-align: center;
    position: relative;
    transition: color 0.35s ease, text-shadow 0.35s ease;
}
body.theme-light .splash-quote {
    color: #0c4a6e;
    text-shadow: 0 2px 18px rgba(8, 145, 178, 0.20);
}
body.theme-dark .splash-quote {
    color: #e0f2fe;
    text-shadow: 0 2px 22px rgba(34, 211, 238, 0.45),
                 0 0 40px rgba(34, 211, 238, 0.22);
}
.splash-quote::before,
.splash-quote::after {
    font-family: 'Playfair Display', 'Georgia', serif;
    font-size: 2.4rem;
    line-height: 0;
    position: relative;
    top: 0.38em;
    opacity: 0.55;
    font-style: normal;
}
.splash-quote::before { content: '\\201C'; margin-right: 8px; }
.splash-quote::after  { content: '\\201D'; margin-left: 8px; }
.splash-loader {
    width: min(340px, 70vw);
    height: 6px;
    border-radius: 999px;
    overflow: hidden;
    margin-top: 8px;
    position: relative;
}
body.theme-light .splash-loader {
    background: rgba(8, 145, 178, 0.14);
    border: 1px solid rgba(8, 145, 178, 0.18);
}
body.theme-dark .splash-loader {
    background: rgba(34, 211, 238, 0.14);
    border: 1px solid rgba(34, 211, 238, 0.20);
}
.splash-loader-bar {
    height: 100%;
    width: 30%;
    border-radius: 999px;
    background: linear-gradient(90deg, #06b6d4 0%, #22d3ee 50%, #67e8f9 100%);
    box-shadow: 0 0 16px rgba(34, 211, 238, 0.75);
    animation: splashSlide 1.5s cubic-bezier(0.65, 0, 0.35, 1) infinite;
}
@keyframes splashSlide {
    0%   { transform: translateX(-110%); }
    100% { transform: translateX(360%); }
}
.splash-hint {
    font-size: 0.9rem;
    font-weight: 500;
    letter-spacing: 0.6px;
    margin: 4px 0 0 0;
    opacity: 0.85;
    animation: splashHintFade 1.6s ease-in-out infinite;
}
body.theme-light .splash-hint { color: #0e7490; }
body.theme-dark  .splash-hint { color: #67e8f9; }
@keyframes splashHintFade {
    0%, 100% { opacity: 0.45; }
    50%      { opacity: 1; }
}
.splash-dots {
    display: inline-flex;
    gap: 6px;
    margin-top: 6px;
}
.splash-dots span {
    width: 8px; height: 8px;
    border-radius: 50%;
    animation: splashDot 1.4s ease-in-out infinite;
}
body.theme-light .splash-dots span { background: #0891b2; }
body.theme-dark  .splash-dots span { background: #22d3ee; }
.splash-dots span:nth-child(1) { animation-delay: 0s; }
.splash-dots span:nth-child(2) { animation-delay: 0.2s; }
.splash-dots span:nth-child(3) { animation-delay: 0.4s; }
@keyframes splashDot {
    0%, 80%, 100% { transform: scale(0.6); opacity: 0.35; }
    40%           { transform: scale(1);   opacity: 1; }
}
@media (max-width: 720px) {
    .splash-title { font-size: 1.9rem; }
    .splash-subtitle { font-size: 0.9rem; }
    .splash-gallery {
        width: min(440px, 74vmin);
        height: min(440px, 74vmin);
    }
    .splash-gallery-item { border-radius: 32px; }
    .splash-gallery-item::after { border-radius: 28px; }
    .splash-quote { font-size: 1.15rem; padding: 0 10px; }
}
@media (max-width: 480px) {
    .splash-title { font-size: 1.6rem; }
    .splash-gallery {
        width: min(320px, 82vmin);
        height: min(320px, 82vmin);
    }
    .splash-gallery-item { border-radius: 24px; }
    .splash-gallery-item::after { border-radius: 20px; }
    .splash-quote { font-size: 1rem; }
}

/* ============ ABOUT PAGE (Enhanced) ============ */
.about-page-wrap {
    max-width: 1080px;
    margin: 0 auto;
    padding: 8px 4px 40px 4px;
}
.about-panel {
    background: var(--bg-panel-right);
    color: var(--text-main);
    border-radius: 24px;
    padding: 44px 46px 46px 46px;
    box-shadow: 0 24px 70px rgba(15, 23, 42, 0.12),
                0 2px 0 rgba(255, 255, 255, 0.5) inset;
    border: 1px solid var(--border);
    position: relative;
    overflow: hidden;
    transition: background 0.35s ease, border-color 0.35s ease, box-shadow 0.35s ease;
}
body.theme-dark .about-panel {
    background: linear-gradient(145deg, #0b1220 0%, #0f172a 60%, #0c4a6e 140%);
    border-color: rgba(34, 211, 238, 0.20);
    box-shadow: 0 26px 80px rgba(2, 6, 23, 0.65),
                0 0 0 1px rgba(34, 211, 238, 0.06) inset;
}
.about-panel::before {
    content: '';
    position: absolute;
    top: 0; left: 0;
    width: 100%; height: 5px;
    background: linear-gradient(90deg, #0891b2, #06b6d4, #22d3ee, #67e8f9, #fb7185);
    opacity: 0.95;
}
.about-panel::after {
    content: '';
    position: absolute;
    top: -200px; right: -200px;
    width: 480px; height: 480px;
    background: radial-gradient(circle, rgba(6, 182, 212, 0.14), transparent 70%);
    pointer-events: none;
    z-index: 0;
}
body.theme-dark .about-panel::after {
    background: radial-gradient(circle, rgba(34, 211, 238, 0.22), transparent 70%);
}
.about-hero {
    position: relative;
    z-index: 1;
    display: flex;
    align-items: center;
    gap: 22px;
    margin-bottom: 26px;
    padding-bottom: 24px;
    border-bottom: 1px solid var(--border);
    flex-wrap: wrap;
}
body.theme-dark .about-hero { border-bottom-color: rgba(34, 211, 238, 0.18); }
.about-hero-icon {
    width: 76px; height: 76px;
    border-radius: 20px;
    display: flex; align-items: center; justify-content: center;
    flex-shrink: 0;
    background: linear-gradient(135deg, #0e7490 0%, #06b6d4 55%, #22d3ee 100%);
    box-shadow: 0 10px 32px rgba(8, 145, 178, 0.45),
                inset 0 2px 0 rgba(255, 255, 255, 0.25);
    animation: aboutIconFloat 4s ease-in-out infinite;
}
@keyframes aboutIconFloat {
    0%, 100% { transform: translateY(0) rotate(0); }
    50%      { transform: translateY(-5px) rotate(-3deg); }
}
.about-hero-icon svg {
    width: 40px !important; height: 40px !important;
    color: #ffffff;
    filter: drop-shadow(0 2px 6px rgba(0, 0, 0, 0.3));
}
.about-hero-text { display: flex; flex-direction: column; gap: 6px; flex: 1; min-width: 220px; }
.about-hero-text h2 {
    margin: 0;
    font-size: 2.1rem;
    font-weight: 800;
    letter-spacing: 0.3px;
    line-height: 1.15;
    color: var(--brand-700);
    border-bottom: none !important;
    padding-bottom: 0 !important;
}
body.theme-dark .about-hero-text h2 {
    color: #67e8f9;
    text-shadow: 0 2px 22px rgba(34, 211, 238, 0.35);
}
.about-hero-subtitle {
    font-size: 1.02rem;
    font-weight: 500;
    letter-spacing: 0.4px;
    color: var(--text-muted);
    margin: 0;
}
body.theme-dark .about-hero-subtitle { color: #a5f3fc; }
.about-hero-badges {
    display: flex;
    gap: 10px;
    flex-wrap: wrap;
    margin-top: 6px;
}
.about-badge {
    display: inline-flex;
    align-items: center;
    gap: 6px;
    padding: 5px 12px;
    border-radius: 999px;
    font-size: 0.76rem;
    font-weight: 700;
    letter-spacing: 0.4px;
    text-transform: uppercase;
    border: 1px solid;
}
.about-badge.cyan {
    color: #0e7490;
    background: rgba(6, 182, 212, 0.12);
    border-color: rgba(6, 182, 212, 0.35);
}
body.theme-dark .about-badge.cyan {
    color: #67e8f9;
    background: rgba(34, 211, 238, 0.14);
    border-color: rgba(34, 211, 238, 0.40);
}
.about-badge.violet {
    color: #6d28d9;
    background: rgba(139, 92, 246, 0.12);
    border-color: rgba(139, 92, 246, 0.35);
}
body.theme-dark .about-badge.violet {
    color: #c4b5fd;
    background: rgba(139, 92, 246, 0.16);
    border-color: rgba(139, 92, 246, 0.45);
}
.about-badge.rose {
    color: #be123c;
    background: rgba(244, 63, 94, 0.12);
    border-color: rgba(244, 63, 94, 0.35);
}
body.theme-dark .about-badge.rose {
    color: #fda4af;
    background: rgba(244, 63, 94, 0.16);
    border-color: rgba(244, 63, 94, 0.45);
}
.about-description {
    position: relative;
    z-index: 1;
    font-size: 1.03rem;
    line-height: 1.85;
    color: var(--text-main);
    margin: 0 0 28px 0;
    padding: 18px 22px;
    border-radius: 14px;
    background: rgba(6, 182, 212, 0.05);
    border-left: 4px solid var(--brand-600);
}
body.theme-dark .about-description {
    background: rgba(34, 211, 238, 0.06);
    border-left-color: #22d3ee;
    color: #e2e8f0;
}
.about-grid {
    position: relative;
    z-index: 1;
    display: grid;
    grid-template-columns: repeat(2, minmax(0, 1fr));
    gap: 20px;
    margin-bottom: 28px;
}
@media (max-width: 760px) {
    .about-grid { grid-template-columns: 1fr; }
}
.about-card {
    border-radius: 18px;
    padding: 22px 24px;
    border: 1px solid var(--border);
    background: var(--bg-panel-left);
    transition: transform 0.25s ease, box-shadow 0.25s ease, border-color 0.25s ease;
    position: relative;
    overflow: hidden;
}
.about-card:hover {
    transform: translateY(-3px);
    box-shadow: 0 14px 34px rgba(8, 145, 178, 0.16);
    border-color: rgba(6, 182, 212, 0.45);
}
body.theme-dark .about-card {
    background: rgba(2, 6, 23, 0.55);
    border-color: rgba(34, 211, 238, 0.20);
}
body.theme-dark .about-card:hover {
    box-shadow: 0 14px 40px rgba(34, 211, 238, 0.22);
    border-color: rgba(34, 211, 238, 0.55);
}
.about-card-title {
    display: flex;
    align-items: center;
    gap: 10px;
    font-size: 1.02rem;
    font-weight: 700;
    letter-spacing: 0.3px;
    color: var(--brand-700);
    margin: 0 0 16px 0;
    padding-bottom: 12px;
    border-bottom: 1px solid var(--border);
}
body.theme-dark .about-card-title {
    color: #67e8f9;
    border-bottom-color: rgba(34, 211, 238, 0.16);
}
.about-card-title svg {
    color: currentColor;
    flex-shrink: 0;
}
.about-legend-list {
    display: flex;
    flex-direction: column;
    gap: 12px;
}
.about-legend-item {
    display: flex;
    align-items: center;
    gap: 14px;
    padding: 10px 14px;
    border-radius: 11px;
    background: rgba(6, 182, 212, 0.04);
    transition: background 0.2s, transform 0.2s;
}
.about-legend-item:hover {
    background: rgba(6, 182, 212, 0.10);
    transform: translateX(3px);
}
body.theme-dark .about-legend-item { background: rgba(34, 211, 238, 0.05); }
body.theme-dark .about-legend-item:hover { background: rgba(34, 211, 238, 0.12); }
.about-legend-swatch {
    width: 26px; height: 26px;
    border-radius: 8px;
    flex-shrink: 0;
    border: 2px solid rgba(255, 255, 255, 0.6);
    box-shadow: 0 3px 10px rgba(0, 0, 0, 0.20),
                inset 0 1px 0 rgba(255, 255, 255, 0.4);
}
.about-legend-label {
    font-size: 0.92rem;
    font-weight: 600;
    color: var(--text-main);
    line-height: 1.35;
}
.about-features-list {
    list-style: none;
    padding: 0;
    margin: 0;
    display: flex;
    flex-direction: column;
    gap: 12px;
}
.about-features-list li {
    display: flex;
    align-items: flex-start;
    gap: 12px;
    font-size: 0.94rem;
    line-height: 1.55;
    color: var(--text-main);
    padding: 4px 0;
}
.about-feature-check {
    flex-shrink: 0;
    width: 22px; height: 22px;
    border-radius: 50%;
    display: inline-flex;
    align-items: center;
    justify-content: center;
    background: linear-gradient(135deg, #0891b2, #22d3ee);
    color: #ffffff;
    font-size: 0.72rem;
    font-weight: 900;
    box-shadow: 0 3px 10px rgba(8, 145, 178, 0.40);
    margin-top: 1px;
}
body.theme-dark .about-feature-check {
    box-shadow: 0 3px 14px rgba(34, 211, 238, 0.55);
}
.about-dev-card {
    position: relative;
    z-index: 1;
    border-radius: 20px;
    padding: 28px 32px;
    background: linear-gradient(135deg,
                rgba(8, 145, 178, 0.06) 0%,
                rgba(6, 182, 212, 0.10) 50%,
                rgba(251, 113, 133, 0.06) 100%);
    border: 1.5px solid rgba(6, 182, 212, 0.28);
    box-shadow: 0 12px 36px rgba(8, 145, 178, 0.14),
                inset 0 1px 0 rgba(255, 255, 255, 0.4);
    display: flex;
    align-items: center;
    gap: 26px;
    margin-top: 28px;
    flex-wrap: wrap;
    overflow: hidden;
}
.about-dev-card::before {
    content: '';
    position: absolute;
    top: -100px; right: -100px;
    width: 300px; height: 300px;
    background: radial-gradient(circle, rgba(251, 113, 133, 0.14), transparent 70%);
    pointer-events: none;
}
body.theme-dark .about-dev-card {
    background: linear-gradient(135deg,
                rgba(34, 211, 238, 0.08) 0%,
                rgba(8, 145, 178, 0.14) 50%,
                rgba(251, 113, 133, 0.10) 100%);
    border-color: rgba(34, 211, 238, 0.42);
    box-shadow: 0 14px 42px rgba(34, 211, 238, 0.20),
                inset 0 1px 0 rgba(34, 211, 238, 0.15);
}
.about-dev-avatar {
    width: 92px; height: 92px;
    border-radius: 50%;
    flex-shrink: 0;
    display: flex;
    align-items: center;
    justify-content: center;
    font-family: 'Playfair Display', 'Georgia', serif;
    font-size: 2rem;
    font-weight: 800;
    color: #ffffff;
    letter-spacing: 1px;
    background: linear-gradient(135deg, #0e7490 0%, #06b6d4 50%, #fb7185 130%);
    box-shadow: 0 10px 30px rgba(8, 145, 178, 0.50),
                0 0 0 6px rgba(6, 182, 212, 0.15),
                inset 0 3px 0 rgba(255, 255, 255, 0.30);
    position: relative;
    animation: devAvatarGlow 3.6s ease-in-out infinite;
    z-index: 1;
}
@keyframes devAvatarGlow {
    0%, 100% {
        box-shadow: 0 10px 30px rgba(8, 145, 178, 0.50),
                    0 0 0 6px rgba(6, 182, 212, 0.15),
                    inset 0 3px 0 rgba(255, 255, 255, 0.30);
    }
    50% {
        box-shadow: 0 14px 42px rgba(34, 211, 238, 0.70),
                    0 0 0 10px rgba(34, 211, 238, 0.22),
                    inset 0 3px 0 rgba(255, 255, 255, 0.40);
    }
}
.about-dev-avatar::after {
    content: '♥';
    position: absolute;
    bottom: -3px; right: -3px;
    width: 26px; height: 26px;
    border-radius: 50%;
    display: flex;
    align-items: center;
    justify-content: center;
    background: #fb7185;
    color: #ffffff;
    font-size: 0.82rem;
    box-shadow: 0 3px 10px rgba(251, 113, 133, 0.65);
    border: 2px solid var(--bg-panel-right);
}
body.theme-dark .about-dev-avatar::after { border-color: #0b1220; }
.about-dev-info {
    display: flex;
    flex-direction: column;
    gap: 6px;
    flex: 1;
    min-width: 200px;
    position: relative;
    z-index: 1;
}
.about-dev-eyebrow {
    font-size: 0.72rem;
    font-weight: 800;
    letter-spacing: 2.4px;
    text-transform: uppercase;
    color: var(--brand-600);
    opacity: 0.9;
}
body.theme-dark .about-dev-eyebrow { color: #67e8f9; }
.about-dev-name {
    font-family: 'Playfair Display', 'Georgia', serif;
    font-size: 1.85rem;
    font-weight: 700;
    font-style: italic;
    letter-spacing: 0.3px;
    line-height: 1.15;
    margin: 0;
    color: var(--brand-700);
}
body.theme-dark .about-dev-name {
    color: #a5f3fc;
    text-shadow: 0 2px 18px rgba(34, 211, 238, 0.4);
}
.about-dev-role {
    font-size: 0.92rem;
    font-weight: 500;
    letter-spacing: 0.5px;
    color: var(--text-muted);
    margin: 0;
}
body.theme-dark .about-dev-role { color: #94a3b8; }
.about-dev-footer {
    font-size: 0.82rem;
    font-style: italic;
    color: var(--text-muted);
    opacity: 0.85;
    margin: 4px 0 0 0;
    letter-spacing: 0.2px;
}
body.theme-dark .about-dev-footer { color: #94a3b8; }
.about-dev-meta {
    display: flex;
    gap: 18px;
    margin-top: 8px;
    padding-top: 10px;
    border-top: 1px dashed rgba(6, 182, 212, 0.28);
    flex-wrap: wrap;
}
body.theme-dark .about-dev-meta { border-top-color: rgba(34, 211, 238, 0.25); }
.about-dev-meta-item {
    display: inline-flex;
    align-items: baseline;
    gap: 6px;
    font-size: 0.78rem;
}
.about-dev-meta-item .k {
    color: var(--text-muted);
    font-weight: 500;
    text-transform: uppercase;
    letter-spacing: 0.8px;
    font-size: 0.68rem;
}
.about-dev-meta-item .v {
    color: var(--brand-700);
    font-weight: 800;
    font-variant-numeric: tabular-nums;
}
body.theme-dark .about-dev-meta-item .v { color: #67e8f9; }
@media (max-width: 560px) {
    .about-panel { padding: 30px 22px 34px 22px; border-radius: 18px; }
    .about-hero-icon { width: 60px; height: 60px; border-radius: 16px; }
    .about-hero-icon svg { width: 32px !important; height: 32px !important; }
    .about-hero-text h2 { font-size: 1.55rem; }
    .about-dev-card { padding: 22px 20px; gap: 18px; }
    .about-dev-avatar { width: 72px; height: 72px; font-size: 1.55rem; }
    .about-dev-name { font-size: 1.5rem; }
}

/* ============ Global overrides ============ */
h4 { color: var(--brand-700); font-size: 1.35rem; }
body.theme-dark h4 { color: #22d3ee; }

body.theme-dark .form-control,
body.theme-dark input[type="text"] {
    background-color: #0b1220 !important;
    border-color: var(--border) !important;
    color: var(--text-main) !important;
}
body.theme-dark .shiny-input-container label { color: var(--text-main); }
"""


# ================== JS ==================
JS = """
(function() {
    function initLang() {
        if (!document.body.classList.contains('lang-en') &&
            !document.body.classList.contains('lang-fa')) {
            document.body.classList.add('lang-en');
        }
    }
    (function initThemeEarly() {
        try {
            var saved = localStorage.getItem('app-theme') || 'light';
            document.documentElement.classList.add('theme-' + saved);
            if (document.body) document.body.classList.add('theme-' + saved);
        } catch (e) {}
    })();

    // ================= Splash Screen + Carousel =================
    var SPLASH_PER_IMAGE_MS = 2500;
    var SPLASH_TAIL_MS = 500;
    var SPLASH_NO_IMAGE_MS = 3500;

    var _galleryItems = [];
    var _galleryDots  = [];

    function hideSplash() {
        var splash = document.getElementById('splash-screen');
        if (!splash) return;
        splash.classList.add('hide');
        setTimeout(function() {
            if (splash && splash.parentNode) {
                splash.parentNode.removeChild(splash);
            }
        }, 900);
    }

    function showGalleryImage(idx) {
        if (idx >= _galleryItems.length) {
            setTimeout(hideSplash, SPLASH_TAIL_MS);
            return;
        }

        if (idx > 0) {
            _galleryItems[idx - 1].classList.remove('active');
            if (_galleryDots.length) {
                _galleryDots[idx - 1].classList.remove('active');
            }
        }

        _galleryItems[idx].classList.add('active');
        if (_galleryDots.length) {
            _galleryDots[idx].classList.add('active');
        }

        setTimeout(function() {
            showGalleryImage(idx + 1);
        }, SPLASH_PER_IMAGE_MS);
    }

    function scheduleSplashHide() {
        _galleryItems = Array.prototype.slice.call(
            document.querySelectorAll('.splash-gallery-item')
        );
        _galleryDots = Array.prototype.slice.call(
            document.querySelectorAll('.splash-gallery-dots span')
        );

        if (_galleryItems.length === 0) {
            setTimeout(hideSplash, SPLASH_NO_IMAGE_MS);
            return;
        }

        showGalleryImage(0);
    }

    if (document.readyState === 'loading') {
        document.addEventListener('DOMContentLoaded', scheduleSplashHide);
    } else {
        scheduleSplashHide();
    }

    function applyTheme(theme) {
        document.body.classList.remove('theme-light', 'theme-dark');
        document.documentElement.classList.remove('theme-light', 'theme-dark');
        document.body.classList.add('theme-' + theme);
        document.documentElement.classList.add('theme-' + theme);
        try { localStorage.setItem('app-theme', theme); } catch(e) {}
        updateActiveTheme(theme);
    }
    function updateActiveTheme(theme) {
        document.querySelectorAll('.submenu-item[data-theme]').forEach(function(b) {
            b.classList.toggle('active', b.getAttribute('data-theme') === theme);
        });
    }
    function updateActiveLang(lang) {
        document.querySelectorAll('.submenu-item[data-lang]').forEach(function(b) {
            b.classList.toggle('active', b.getAttribute('data-lang') === lang);
        });
    }

    function openSidebar() {
        var sb = document.getElementById('sidebar');
        var ov = document.getElementById('sidebar-overlay');
        if (sb) sb.classList.add('open');
        if (ov) ov.classList.add('open');
        document.body.style.overflow = 'hidden';
    }
    function closeSidebar() {
        var sb = document.getElementById('sidebar');
        var ov = document.getElementById('sidebar-overlay');
        if (sb) sb.classList.remove('open');
        if (ov) ov.classList.remove('open');
        document.body.style.overflow = '';
    }

    document.addEventListener('click', function(e) {
        if (e.target.closest('#hamburger-btn')) { openSidebar(); return; }
        if (e.target.closest('#sidebar-close') ||
            e.target.closest('#sidebar-overlay')) {
            closeSidebar(); return;
        }
        var navBtn = e.target.closest('.sidebar-item[data-page]');
        if (navBtn) {
            var page = navBtn.getAttribute('data-page');
            document.querySelectorAll('.sidebar-item[data-page]').forEach(function(b) {
                b.classList.toggle('active', b === navBtn);
            });
            if (window.Shiny && Shiny.setInputValue) {
                Shiny.setInputValue('active_page', page, {priority: 'event'});
            }
            closeSidebar();
            return;
        }
        var nestedToggle = e.target.closest('.nested-toggle');
        if (nestedToggle) {
            var key = nestedToggle.getAttribute('data-toggle');
            var submenu = document.getElementById('submenu-' + key);
            if (!submenu) return;
            var isOpen = submenu.classList.contains('open');
            document.querySelectorAll('.sidebar-submenu-sub').forEach(function(s) {
                s.classList.remove('open');
            });
            document.querySelectorAll('.nested-toggle').forEach(function(b) {
                b.classList.remove('expanded');
            });
            if (!isOpen) {
                submenu.classList.add('open');
                nestedToggle.classList.add('expanded');
            }
            return;
        }
        var toggleBtn = e.target.closest('.sidebar-item-toggle');
        if (toggleBtn) {
            var key = toggleBtn.getAttribute('data-toggle');
            var submenu = document.getElementById('submenu-' + key);
            if (!submenu) return;
            var isOpen = submenu.classList.contains('open');
            document.querySelectorAll('.sidebar-submenu').forEach(function(s) {
                s.classList.remove('open');
            });
            document.querySelectorAll('.sidebar-item-toggle').forEach(function(b) {
                b.classList.remove('expanded');
            });
            if (!isOpen) {
                submenu.classList.add('open');
                toggleBtn.classList.add('expanded');
            }
            return;
        }
        var langItem = e.target.closest('.submenu-item[data-lang]');
        if (langItem) {
            var lg = langItem.getAttribute('data-lang');
            if (window.Shiny && Shiny.setInputValue) {
                Shiny.setInputValue('selected_lang', lg, {priority: 'event'});
            } else {
                document.body.classList.remove('lang-en', 'lang-fa');
                document.body.classList.add('lang-' + lg);
            }
            updateActiveLang(lg);
            return;
        }
        var themeItem = e.target.closest('.submenu-item[data-theme]');
        if (themeItem) {
            applyTheme(themeItem.getAttribute('data-theme'));
            return;
        }
    });

    document.addEventListener('keydown', function(e) {
        if (e.key === 'Escape') closeSidebar();
    });

    document.addEventListener('click', function(e) {
        var item = e.target.closest('.filmstrip-item');
        if (item && item.dataset.slice) {
            var idx = parseInt(item.dataset.slice, 10);
            if (!isNaN(idx) && window.Shiny && Shiny.setInputValue) {
                Shiny.setInputValue('filmstrip_click',
                    {idx: idx, ts: Date.now()}, {priority: 'event'});
            }
            return;
        }
        var prev = e.target.closest('.filmstrip-arrow.prev');
        if (prev && !prev.disabled && window.Shiny && Shiny.setInputValue) {
            Shiny.setInputValue('filmstrip_nav_prev', Date.now(), {priority: 'event'});
            return;
        }
        var next = e.target.closest('.filmstrip-arrow.next');
        if (next && !next.disabled && window.Shiny && Shiny.setInputValue) {
            Shiny.setInputValue('filmstrip_nav_next', Date.now(), {priority: 'event'});
            return;
        }
    });

    document.addEventListener('keydown', function(e) {
        if (e.key !== 'Enter' || !e.target) return;
        var isJumpInput = (e.target.id === 'jump_input') ||
                          (e.target.classList &&
                           e.target.classList.contains('jump-input'));
        if (!isJumpInput) return;
        e.preventDefault();
        if (window.Shiny && Shiny.setInputValue) {
            Shiny.setInputValue('jump_go', Date.now(), {priority: 'event'});
        } else {
            var goBtn = document.querySelector('.jump-btn');
            if (goBtn) goBtn.click();
        }
    });

    function wireDragDrop() {
        document.querySelectorAll('.file-card').forEach(function(card) {
            ['dragenter', 'dragover'].forEach(function(ev) {
                card.addEventListener(ev, function(e) {
                    e.preventDefault(); e.stopPropagation();
                    card.classList.add('drag-over');
                });
            });
            ['dragleave', 'drop'].forEach(function(ev) {
                card.addEventListener(ev, function(e) {
                    e.preventDefault(); e.stopPropagation();
                    if (ev === 'dragleave' && card.contains(e.relatedTarget)) return;
                    card.classList.remove('drag-over');
                });
            });
        });
    }

    document.addEventListener('DOMContentLoaded', function() {
        initLang();
        var saved = 'light';
        try { saved = localStorage.getItem('app-theme') || 'light'; } catch(e){}
        document.body.classList.remove('theme-light', 'theme-dark');
        document.body.classList.add('theme-' + saved);
        updateActiveTheme(saved);
        updateActiveLang(document.body.classList.contains('lang-fa') ? 'fa' : 'en');
        wireDragDrop();
        var seg = document.querySelector('.sidebar-item[data-page="segmentation"]');
        if (seg) seg.classList.add('active');
    });

    if (window.Shiny && Shiny.addCustomMessageHandler) {
        Shiny.addCustomMessageHandler('set_lang', function(lang) {
            document.body.classList.remove('lang-en', 'lang-fa');
            document.body.classList.add('lang-' + lang);
            updateActiveLang(lang);
        });
        Shiny.addCustomMessageHandler('set_active_page', function(page) {
            document.querySelectorAll('.sidebar-item[data-page]').forEach(function(b) {
                b.classList.toggle('active', b.getAttribute('data-page') === page);
            });
        });
    }

    var mo = new MutationObserver(function() {
        wireDragDrop();
        var theme = document.body.classList.contains('theme-dark') ? 'dark' : 'light';
        updateActiveTheme(theme);
        var lg = document.body.classList.contains('lang-fa') ? 'fa' : 'en';
        updateActiveLang(lg);
    });
    document.addEventListener('DOMContentLoaded', function() {
        mo.observe(document.body, {childList: true, subtree: true});
    });

    function wireResultPanelLoading() {
        document.querySelectorAll('.result-panel').forEach(function(panel) {
            if (panel.dataset.loadingWired === '1') return;
            panel.dataset.loadingWired = '1';
            function update() {
                var loading = panel.querySelector('.recalculating') !== null;
                panel.classList.toggle('is-loading', loading);
            }
            var obs = new MutationObserver(update);
            obs.observe(panel, {
                childList: true,
                subtree: true,
                attributes: true,
                attributeFilter: ['class'],
            });
            update();
        });
    }

    (function () {
        var stateMap = new WeakMap();
        function getState(panel) {
            if (!stateMap.has(panel)) {
                stateMap.set(panel, { scale: 1, tx: 0, ty: 0, dragging: false });
            }
            return stateMap.get(panel);
        }
        function apply(panel) {
            var img = panel.querySelector('.result-panel-image img.slice-img');
            if (!img) return;
            var s = getState(panel);
            img.style.transform =
                'translate(' + s.tx + 'px,' + s.ty + 'px) scale(' + s.scale + ')';
            img.style.cursor = s.scale > 1
                ? (s.dragging ? 'grabbing' : 'grab')
                : 'default';
        }
        function wire(panel) {
            if (panel.dataset.zoomWired === '1') return;
            panel.dataset.zoomWired = '1';
            var wrap = panel.querySelector('.result-panel-image');
            if (!wrap) return;
            var btnIn  = panel.querySelector('.ctrl-zoom-in');
            var btnOut = panel.querySelector('.ctrl-zoom-out');
            var btnFit = panel.querySelector('.ctrl-fit');
            var btnExp = panel.querySelector('.ctrl-expand');
            function step(factor, clientX, clientY) {
                var s = getState(panel);
                var oldScale = s.scale;
                var newScale = Math.max(0.5, Math.min(s.scale * factor, 8));
                if (newScale === oldScale) return;
                var rect = wrap.getBoundingClientRect();
                var cx = (clientX != null ? clientX : rect.left + rect.width / 2)
                       - (rect.left + rect.width / 2);
                var cy = (clientY != null ? clientY : rect.top + rect.height / 2)
                       - (rect.top + rect.height / 2);
                var r = newScale / oldScale;
                s.tx = cx - (cx - s.tx) * r;
                s.ty = cy - (cy - s.ty) * r;
                s.scale = newScale;
                if (s.scale <= 1) { s.tx = 0; s.ty = 0; }
                apply(panel);
            }
            if (btnIn)  btnIn.onclick  = function (e) { e.preventDefault(); e.stopPropagation(); step(1.3); };
            if (btnOut) btnOut.onclick = function (e) { e.preventDefault(); e.stopPropagation(); step(1 / 1.3); };
            if (btnFit) btnFit.onclick = function (e) {
                e.preventDefault(); e.stopPropagation();
                var s = getState(panel);
                s.scale = 1; s.tx = 0; s.ty = 0;
                apply(panel);
            };
            if (btnExp) btnExp.onclick = function (e) {
                e.preventDefault(); e.stopPropagation();
                if (document.fullscreenElement === panel) {
                    document.exitFullscreen();
                } else if (panel.requestFullscreen) {
                    panel.requestFullscreen().catch(function () {});
                }
            };
            wrap.addEventListener('wheel', function (e) {
                var img = wrap.querySelector('img.slice-img');
                if (!img) return;
                e.preventDefault();
                step(e.deltaY < 0 ? 1.15 : 1 / 1.15, e.clientX, e.clientY);
            }, { passive: false });
            var isDown = false, sx = 0, sy = 0, stx = 0, sty = 0;
            wrap.addEventListener('mousedown', function (e) {
                var s = getState(panel);
                if (s.scale <= 1 || e.button !== 0) return;
                isDown = true;
                sx = e.clientX; sy = e.clientY;
                stx = s.tx; sty = s.ty;
                s.dragging = true;
                var img = wrap.querySelector('img.slice-img');
                if (img) img.classList.add('panning');
                e.preventDefault();
            });
            window.addEventListener('mousemove', function (e) {
                if (!isDown) return;
                var s = getState(panel);
                s.tx = stx + (e.clientX - sx);
                s.ty = sty + (e.clientY - sy);
                apply(panel);
            });
            window.addEventListener('mouseup', function () {
                if (!isDown) return;
                isDown = false;
                var s = getState(panel);
                s.dragging = false;
                var img = wrap.querySelector('img.slice-img');
                if (img) img.classList.remove('panning');
                apply(panel);
            });
            wrap.addEventListener('dblclick', function (e) {
                e.preventDefault();
                var s = getState(panel);
                s.scale = 1; s.tx = 0; s.ty = 0;
                apply(panel);
            });
        }
        function initAll() {
            document.querySelectorAll('.result-panel').forEach(function (p) {
                wire(p); apply(p);
            });
        }
        var zoomObserver = new MutationObserver(function () {
            requestAnimationFrame(function () {
                document.querySelectorAll('.result-panel').forEach(apply);
            });
        });
        function boot() {
            initAll();
            wireResultPanelLoading();
            zoomObserver.observe(document.body, {childList: true, subtree: true});
        }
        if (document.readyState === 'loading') {
            document.addEventListener('DOMContentLoaded', boot);
        } else {
            boot();
        }
    })();
})();
"""


# ================== SIDEBAR HTML ==================
SIDEBAR_HTML = f"""
<div class="sidebar-overlay" id="sidebar-overlay"></div>
<aside class="sidebar" id="sidebar">
    <div class="sidebar-header">
        <span class="sidebar-title">
            <span class="en-text">{TEXT['en']['menu']}</span>
            <span class="fa-text">{TEXT['fa']['menu']}</span>
        </span>
        <button type="button" class="sidebar-close" id="sidebar-close" aria-label="Close">
            {CLOSE_ICON_SVG}
        </button>
    </div>
    <nav class="sidebar-nav">
        <button type="button" class="sidebar-item active" data-page="segmentation">
            <span class="item-icon">{SEGMENTATION_NAV_ICON_SVG}</span>
            <span class="item-text">
                <span class="en-text">{TEXT['en']['nav_segmentation']}</span>
                <span class="fa-text">{TEXT['fa']['nav_segmentation']}</span>
            </span>
        </button>
        <button type="button" class="sidebar-item" data-page="view_3d">
            <span class="item-icon">{CUBE_ICON_SVG}</span>
            <span class="item-text">
                <span class="en-text">{TEXT['en']['nav_3d']}</span>
                <span class="fa-text">{TEXT['fa']['nav_3d']}</span>
            </span>
        </button>
        
        <div class="sidebar-group">
            <button type="button" class="sidebar-item sidebar-item-toggle" data-toggle="settings">
                <span class="item-icon">{SETTINGS_NAV_ICON_SVG}</span>
                <span class="item-text">
                    <span class="en-text">{TEXT['en']['nav_settings']}</span>
                    <span class="fa-text">{TEXT['fa']['nav_settings']}</span>
                </span>
                <span class="item-chevron">{CHEVRON_DOWN_SVG}</span>
            </button>
            <div class="sidebar-submenu" id="submenu-settings">
                
                <button type="button" class="submenu-item nested-toggle" data-toggle="theme-sub">
                    <span class="opt-icon">🎨</span>
                    <span class="item-text">
                        <span class="en-text">{TEXT['en']['theme_title']}</span>
                        <span class="fa-text">{TEXT['fa']['theme_title']}</span>
                    </span>
                    <span class="item-chevron">{CHEVRON_DOWN_SVG}</span>
                </button>
                <div class="sidebar-submenu-sub" id="submenu-theme-sub">
                    <button type="button" class="submenu-item" data-theme="light">
                        <span class="opt-icon">☀️</span>
                        <span>
                            <span class="en-text">{TEXT['en']['theme_light']}</span>
                            <span class="fa-text">{TEXT['fa']['theme_light']}</span>
                        </span>
                    </button>
                    <button type="button" class="submenu-item" data-theme="dark">
                        <span class="opt-icon">🌙</span>
                        <span>
                            <span class="en-text">{TEXT['en']['theme_dark']}</span>
                            <span class="fa-text">{TEXT['fa']['theme_dark']}</span>
                        </span>
                    </button>
                </div>

                <button type="button" class="submenu-item nested-toggle" data-toggle="lang-sub">
                    <span class="opt-icon">🌐</span>
                    <span class="item-text">
                        <span class="en-text">{TEXT['en']['lang_title']}</span>
                        <span class="fa-text">{TEXT['fa']['lang_title']}</span>
                    </span>
                    <span class="item-chevron">{CHEVRON_DOWN_SVG}</span>
                </button>
                <div class="sidebar-submenu-sub" id="submenu-lang-sub">
                    <button type="button" class="submenu-item" data-lang="en">
                        <span class="opt-icon">🇬🇧</span>
                        <span>English</span>
                    </button>
                    <button type="button" class="submenu-item" data-lang="fa">
                        <span class="opt-icon">🇮🇷</span>
                        <span>فارسی</span>
                    </button>
                </div>

            </div>
        </div>
        
        <button type="button" class="sidebar-item" data-page="about">
            <span class="item-icon">{INFO_NAV_ICON_SVG}</span>
            <span class="item-text">
                <span class="en-text">{TEXT['en']['nav_about']}</span>
                <span class="fa-text">{TEXT['fa']['nav_about']}</span>
            </span>
        </button>
    </nav>
</aside>
"""


# ================== SPLASH HTML ==================
_gallery_items_html = "".join(
    f'<div class="splash-gallery-item" data-index="{i}">'
    f'<img src="{src}" alt="liver"/>'
    f'</div>'
    for i, src in enumerate(LIVER_GALLERY)
)

_gallery_dots_html = "".join(
    f'<span data-dot="{i}"></span>'
    for i in range(len(LIVER_GALLERY))
)

SPLASH_HTML = f"""
<div class="splash-screen" id="splash-screen" aria-hidden="true">
    <div class="splash-blob splash-blob-1"></div>
    <div class="splash-blob splash-blob-2"></div>
    <div class="splash-content">
        <h1 class="splash-title">
            <span class="en-text">Liver Segmentation</span>
            <span class="fa-text">تقسیم‌بندی کبد</span>
        </h1>
        <p class="splash-subtitle">
            <span class="en-text">{TEXT['en']['splash_welcome']} · {TEXT['en']['splash_subtitle']}</span>
            <span class="fa-text">{TEXT['fa']['splash_welcome']} · {TEXT['fa']['splash_subtitle']}</span>
        </p>

        <div class="splash-gallery">
            {_gallery_items_html}
        </div>

        <div class="splash-gallery-dots">
            {_gallery_dots_html}
        </div>

        <div class="splash-quote-wrap">
            <div class="splash-quote-divider"></div>
            <p class="splash-quote">
                <span class="en-text">{TEXT['en']['splash_quote']}</span>
                <span class="fa-text">{TEXT['fa']['splash_quote']}</span>
            </p>
            <div class="splash-quote-divider"></div>
        </div>

        <div class="splash-loader">
            <div class="splash-loader-bar"></div>
        </div>
        <p class="splash-hint">
            <span class="en-text">{TEXT['en']['splash_preparing']}</span>
            <span class="fa-text">{TEXT['fa']['splash_preparing']}</span>
        </p>
        <div class="splash-dots">
            <span></span><span></span><span></span>
        </div>
    </div>
</div>
"""


# ================== helper: پنل نتیجه ==================
def _result_panel(title_en: str, title_fa: str,
                  slice_lbl_id: str, plot_id: str,
                  legend_id: str = None):
    header_items = [
        ui.div(bilingual(title_en, title_fa), class_="result-panel-title"),
    ]
    if legend_id is not None:
        header_items.append(
            ui.div(ui.output_ui(legend_id), class_="result-panel-legend")
        )
    header_items.append(
        ui.HTML('<button type="button" class="result-panel-close">×</button>')
    )

    return ui.div(
        ui.div(
            ui.HTML('<div class="panel-spinner-lg" aria-hidden="true"></div>'),
            class_="panel-loading-overlay"
        ),
        ui.div(*header_items, class_="result-panel-header"),
        ui.div(
            ui.div(
                ui.output_text(slice_lbl_id, inline=True),
                class_="result-toolbar-left"
            ),
            ui.div(
                ui.HTML(f'<button type="button" class="ctrl-btn ctrl-zoom-in" title="Zoom In">{ZOOM_IN_SVG}</button>'),
                ui.HTML(f'<button type="button" class="ctrl-btn ctrl-zoom-out" title="Zoom Out">{ZOOM_OUT_SVG}</button>'),
                ui.HTML(f'<button type="button" class="ctrl-btn ctrl-fit" title="Fit">{FIT_SVG}</button>'),
                ui.HTML(f'<button type="button" class="ctrl-btn ctrl-expand" title="Expand">{EXPAND_SVG}</button>'),
                class_="result-panel-controls"
            ),
            class_="result-panel-toolbar"
        ),
        ui.div(
            ui.output_ui(plot_id),
            class_="result-panel-image"
        ),
        class_="result-panel"
    )


# ================== helper: کارت آپلود ==================
def _file_card(card_class: str, icon_svg: str,
               title_en: str, title_fa: str,
               hint_en: str, hint_fa: str,
               input_id: str, optional: bool = False):
    title_inner = [bilingual(title_en, title_fa)]
    if optional:
        title_inner.append(ui.HTML(
            f'<span class="badge-optional">'
            f'<span class="en-text">{html_lib.escape(TEXT["en"]["optional"])}</span>'
            f'<span class="fa-text">{html_lib.escape(TEXT["fa"]["optional"])}</span>'
            f'</span>'
        ))

    return ui.div(
        ui.div(
            ui.HTML(f'<div class="file-card-icon-box">{icon_svg}</div>'),
            ui.div(
                ui.div(*title_inner, class_="file-card-title"),
                ui.div(bilingual(hint_en, hint_fa), class_="file-card-hint"),
                class_="file-card-titles"
            ),
            class_="file-card-head"
        ),
        ui.input_file(
            input_id, label="",
            multiple=True, accept=".dcm,.dicom", width="100%"
        ),
        class_=f"file-card {card_class}"
    )


# ================== صفحات ==================
SEGMENTATION_PAGE = ui.div(
    ui.row(
        ui.column(4,
            ui.div(
                ui.div(
                    ui.HTML(f'<span class="panel-title-icon">'
                            f'{UPLOAD_TITLE_ICON_SVG}</span>'),
                    ui.tags.h4(bilingual("Upload Data", "بارگذاری داده")),
                    class_="panel-title-row"
                ),
                _file_card(
                    card_class="dicom-card",
                    icon_svg=DICOM_IMG_ICON_SVG,
                    title_en="DICOM Image Files",
                    title_fa="فایل‌های تصویری DICOM",
                    hint_en="CT scan series — .dcm / .dicom",
                    hint_fa="سری اسکن CT — .dcm / .dicom",
                    input_id="dicom_files",
                    optional=False,
                ),
                _file_card(
                    card_class="mask-card",
                    icon_svg=MASK_ICON_SVG,
                    title_en="Ground Truth Masks",
                    title_fa="ماسک‌های مرجع",
                    hint_en="Reference masks for evaluation",
                    hint_fa="ماسک‌های مرجع برای ارزیابی",
                    input_id="mask_files",
                    optional=True,
                ),
                ui.div(
                    ui.input_action_button(
                        "run",
                        ui.HTML(
                            f'<span style="display:inline-flex;align-items:center;'
                            f'gap:10px;">{PLAY_ICON_SVG}'
                            f'<span class="en-text">Run Segmentation</span>'
                            f'<span class="fa-text">اجرای تقسیم‌بندی</span>'
                            f'</span>'
                        ),
                        class_="btn-run"
                    ),
                    class_="btn-run-wrap"
                ),
                class_="left-panel"
            ),
            ui.div(
                ui.div(
                    ui.HTML(METRICS_ICON_SVG),
                    ui.tags.h4(bilingual("Slice-based Results",
                                         "نتایج برش‌محور")),
                    class_="metrics-header"
                ),
                ui.output_ui("slice_metrics_cards"),
                class_="metrics-panel"
            ),
            ui.div(
                ui.div(
                    ui.HTML(METRICS_ICON_SVG),
                    ui.tags.h4(bilingual("Segmentation Results",
                                         "نتایج تقسیم‌بندی")),
                    class_="metrics-header"
                ),
                ui.output_ui("metrics_cards"),
                class_="metrics-panel"
            ),
        ),
        ui.column(8,
            ui.div(
                ui.h4(bilingual("Results", "نتایج")),
                ui.output_ui("dicom_info_strip"),
                ui.row(
                    ui.column(6, _result_panel(
                        "Original Image", "تصویر اصلی",
                        "slice_lbl_original", "plot_original")),
                    ui.column(6, _result_panel(
                        "Predicted Mask Overlay", "روکش ماسک پیش‌بینی",
                        "slice_lbl_pred", "plot_pred_overlay",
                        "legend_pred")),
                    ui.column(6, _result_panel(
                        "Ground Truth Overlay", "روکش ماسک مرجع",
                        "slice_lbl_gt", "plot_gt_overlay",
                        "legend_gt")),
                    ui.column(6, _result_panel(
                        "Reference vs Prediction", "مرجع در مقابل پیش‌بینی",
                        "slice_lbl_combined", "plot_combined",
                        "legend_combined")),
                    class_="result-grid"
                ),
                ui.div(
                    ui.div(
                        ui.div(
                            ui.HTML(f'<span class="filmstrip-icon">'
                                    f'{SLICE_NAV_ICON_SVG}</span>'),
                            bilingual("Slice Navigation", "پیمایش برش‌ها"),
                            class_="filmstrip-title-left"
                        ),
                        ui.div(
                            ui.input_text(
                                "jump_input", None,
                                placeholder="Slice #", width="96px",
                            ),
                            ui.input_action_button(
                                "jump_go",
                                bilingual("Go", "برو"),
                                class_="jump-btn"
                            ),
                            class_="jump-to-slice"
                        ),
                        class_="filmstrip-title"
                    ),
                    ui.output_ui("slice_filmstrip"),
                    class_="filmstrip-section"
                ),
                class_="right-panel"
            ),
        ),
    )
)

VIEW_3D_PAGE = ui.div(
    ui.div(
        ui.div(
            ui.HTML(f'<div class="view3d-header-icon">{CUBE_ICON_SVG}</div>'),
            ui.div(
                ui.tags.h3(bilingual("3D Liver Reconstruction",
                                     "بازسازی سه‌بعدی کبد")),
                ui.div(bilingual(
                    "Smoothed 3D surface: Ground Truth · Prediction · Difference",
                    "سطح سه‌بعدی هموار: مرجع · پیش‌بینی · تفاوت"),
                    class_="view3d-hint"),
                class_="view3d-header-text"
            ),
            class_="view3d-header"
        ),
        ui.div(
            ui.div(
                ui.div(
                    ui.div(
                        bilingual("Comparison Display", "نمایش مقایسه"),
                        class_="control-label"
                    ),
                    ui.div(
                        ui.input_checkbox(
                            "show_3d_diff",
                            ui.HTML(
                                f'<span class="view3d-color-dot" '
                                f'style="background:#fb7185; color:#fb7185;"></span>'
                                f'<span class="en-text">Show Difference</span>'
                                f'<span class="fa-text">نمایش تفاوت</span>'
                            ),
                            value=True,
                        ),
                        class_="view3d-toggle-row"
                    ),
                    class_="view3d-control-group"
                ),
                
                ui.div(
                    ui.div(
                        bilingual("Evaluation Metrics", "معیارهای ارزیابی"),
                        class_="control-label"
                    ),
                    ui.div(
                        ui.output_ui("metrics_3d_cards"),
                        class_="view3d-metrics-wrap"
                    ),
                    class_="view3d-control-group"
                ),
                
                ui.output_ui("view3d_info"),
                class_="view3d-controls"
            ),
            ui.div(
                ui.output_ui("plot_3d"),
                class_="view3d-canvas-wrap"
            ),
            class_="view3d-body"
        ),
        class_="view3d-wrapper"
    )
)


# ================== صفحه درباره ==================
ABOUT_PAGE = ui.div(
    ui.div(
        # ---------- HERO ----------
        ui.div(
            ui.div(
                ui.HTML("""<svg viewBox="0 0 24 24" fill="none"
                    stroke="currentColor" stroke-width="1.8"
                    stroke-linecap="round" stroke-linejoin="round">
                    <path d="M21 16V8a2 2 0 0 0-1-1.73l-7-4a2 2 0 0 0-2 0l-7 4A2 2 0 0 0 3 8v8a2 2 0 0 0 1 1.73l7 4a2 2 0 0 0 2 0l7-4A2 2 0 0 0 21 16z"></path>
                    <polyline points="3.27 6.96 12 12.01 20.73 6.96"></polyline>
                    <line x1="12" y1="22.08" x2="12" y2="12"></line>
                </svg>"""),
                class_="about-hero-icon"
            ),
            ui.div(
                ui.tags.h2(bilingual(
                    TEXT["en"]["about_title"],
                    TEXT["fa"]["about_title"]
                )),
                ui.p(bilingual(
                    TEXT["en"]["about_subtitle"],
                    TEXT["fa"]["about_subtitle"]
                ), class_="about-hero-subtitle"),
                ui.div(
                    ui.HTML(
                        '<span class="about-badge cyan">'
                        '<span class="en-text">Deep Learning</span>'
                        '<span class="fa-text">یادگیری عمیق</span>'
                        '</span>'
                    ),
                    ui.HTML(
                        '<span class="about-badge violet">'
                        '<span class="en-text">3D Reconstruction</span>'
                        '<span class="fa-text">بازسازی سه‌بعدی</span>'
                        '</span>'
                    ),
                    ui.HTML(
                        '<span class="about-badge rose">'
                        '<span class="en-text">Medical Imaging</span>'
                        '<span class="fa-text">تصویربرداری پزشکی</span>'
                        '</span>'
                    ),
                    class_="about-hero-badges"
                ),
                class_="about-hero-text"
            ),
            class_="about-hero"
        ),

        # ---------- DESCRIPTION ----------
        ui.div(
            bilingual(
                TEXT["en"]["about_description"],
                TEXT["fa"]["about_description"]
            ),
            class_="about-description"
        ),

        # ---------- 2-COLUMN GRID ----------
        ui.div(
            # Colors card
            ui.div(
                ui.div(
                    ui.HTML("""<svg viewBox="0 0 24 24" width="20" height="20"
                        fill="none" stroke="currentColor" stroke-width="2.2"
                        stroke-linecap="round" stroke-linejoin="round">
                        <circle cx="13.5" cy="6.5" r="2.5"></circle>
                        <circle cx="19.5" cy="10.5" r="2.5"></circle>
                        <circle cx="6" cy="12" r="2.5"></circle>
                        <circle cx="10" cy="18" r="2.5"></circle>
                    </svg>"""),
                    bilingual(
                        TEXT["en"]["about_colors_title"],
                        TEXT["fa"]["about_colors_title"]
                    ),
                    class_="about-card-title"
                ),
                ui.div(
                    ui.div(
                        ui.HTML('<span class="about-legend-swatch" '
                                'style="background:linear-gradient(135deg,#facc15,#fde047);"></span>'),
                        ui.div(
                            bilingual(
                                TEXT["en"]["about_color_tp"],
                                TEXT["fa"]["about_color_tp"]
                            ),
                            class_="about-legend-label"
                        ),
                        class_="about-legend-item"
                    ),
                    ui.div(
                        ui.HTML('<span class="about-legend-swatch" '
                                'style="background:linear-gradient(135deg,#06b6d4,#22d3ee);"></span>'),
                        ui.div(
                            bilingual(
                                TEXT["en"]["about_color_fp"],
                                TEXT["fa"]["about_color_fp"]
                            ),
                            class_="about-legend-label"
                        ),
                        class_="about-legend-item"
                    ),
                    ui.div(
                        ui.HTML('<span class="about-legend-swatch" '
                                'style="background:linear-gradient(135deg,#e11d48,#fb7185);"></span>'),
                        ui.div(
                            bilingual(
                                TEXT["en"]["about_color_fn"],
                                TEXT["fa"]["about_color_fn"]
                            ),
                            class_="about-legend-label"
                        ),
                        class_="about-legend-item"
                    ),
                    ui.div(
                        ui.HTML('<span class="about-legend-swatch" '
                                'style="background:linear-gradient(135deg,#475569,#64748b);"></span>'),
                        ui.div(
                            bilingual(
                                TEXT["en"]["about_color_bg"],
                                TEXT["fa"]["about_color_bg"]
                            ),
                            class_="about-legend-label"
                        ),
                        class_="about-legend-item"
                    ),
                    class_="about-legend-list"
                ),
                class_="about-card"
            ),

            # Features card
            ui.div(
                ui.div(
                    ui.HTML("""<svg viewBox="0 0 24 24" width="20" height="20"
                        fill="none" stroke="currentColor" stroke-width="2.2"
                        stroke-linecap="round" stroke-linejoin="round">
                        <polygon points="12 2 15.09 8.26 22 9.27 17 14.14 18.18 21.02 12 17.77 5.82 21.02 7 14.14 2 9.27 8.91 8.26 12 2"></polygon>
                    </svg>"""),
                    bilingual(
                        TEXT["en"]["about_features_title"],
                        TEXT["fa"]["about_features_title"]
                    ),
                    class_="about-card-title"
                ),
                ui.tags.ul(
                    ui.tags.li(
                        ui.HTML('<span class="about-feature-check">✓</span>'),
                        bilingual(
                            TEXT["en"]["about_feat_1"],
                            TEXT["fa"]["about_feat_1"]
                        ),
                    ),
                    ui.tags.li(
                        ui.HTML('<span class="about-feature-check">✓</span>'),
                        bilingual(
                            TEXT["en"]["about_feat_2"],
                            TEXT["fa"]["about_feat_2"]
                        ),
                    ),
                    ui.tags.li(
                        ui.HTML('<span class="about-feature-check">✓</span>'),
                        bilingual(
                            TEXT["en"]["about_feat_3"],
                            TEXT["fa"]["about_feat_3"]
                        ),
                    ),
                    ui.tags.li(
                        ui.HTML('<span class="about-feature-check">✓</span>'),
                        bilingual(
                            TEXT["en"]["about_feat_4"],
                            TEXT["fa"]["about_feat_4"]
                        ),
                    ),
                    class_="about-features-list"
                ),
                class_="about-card"
            ),

            class_="about-grid"
        ),

        # ---------- DEVELOPER CARD ----------
        ui.div(
            ui.div(
                "YG",
                class_="about-dev-avatar"
            ),
            ui.div(
                ui.div(
                    bilingual(
                        TEXT["en"]["about_dev_title"],
                        TEXT["fa"]["about_dev_title"]
                    ),
                    class_="about-dev-eyebrow"
                ),
                ui.h3(
                    bilingual(
                        TEXT["en"]["about_dev_name_en"],
                        TEXT["fa"]["about_dev_name_fa"]
                    ),
                    class_="about-dev-name"
                ),
                ui.p(
                    bilingual(
                        TEXT["en"]["about_dev_role_en"],
                        TEXT["fa"]["about_dev_role_fa"]
                    ),
                    class_="about-dev-role"
                ),
                ui.p(
                    bilingual(
                        TEXT["en"]["about_dev_footer"],
                        TEXT["fa"]["about_dev_footer"]
                    ),
                    class_="about-dev-footer"
                ),
                ui.div(
                    ui.div(
                        ui.HTML(f'<span class="k">{TEXT["en"]["about_version"]}</span>'),
                        ui.HTML(f'<span class="v">{TEXT["en"]["about_version_value"]}</span>'),
                        class_="about-dev-meta-item"
                    ),
                    ui.div(
                        ui.HTML(f'<span class="k">{TEXT["en"]["about_year"]}</span>'),
                        ui.HTML(f'<span class="v">{TEXT["en"]["about_year_value"]}</span>'),
                        class_="about-dev-meta-item"
                    ),
                    class_="about-dev-meta"
                ),
                class_="about-dev-info"
            ),
            class_="about-dev-card"
        ),

        class_="about-panel about-body"
    ),
    class_="about-page-wrap"
)


# ================== UI ==================
app_ui = ui.page_fluid(
    ui.tags.style(CSS),
    ui.tags.script(JS),

    ui.HTML(SIDEBAR_HTML),
    ui.HTML(SPLASH_HTML),

    ui.div(
        ui.HTML(
            f'<button type="button" class="hamburger-btn" id="hamburger-btn" '
            f'aria-label="Menu">{HAMBURGER_ICON_SVG}</button>'
        ),
        ui.div(
            ui.HTML(
                f'<img src="{LIVER_LOGO_SRC}" class="header-liver-icon" alt="Liver"/>'
                if LIVER_LOGO_SRC else ''
            ),
            ui.div(
                ui.h1(bilingual("Liver Segmentation", "تقسیم‌بندی کبد")),
                ui.div(
                    bilingual("AI-Powered 3D Liver Analysis",
                              "تحلیل سه‌بعدی کبد با هوش مصنوعی"),
                    class_="app-subtitle"
                ),
                class_="header-titles"
            ),
            class_="header-content"
        ),
        class_="app-header"
    ),

    ui.navset_hidden(
        ui.nav_panel("Segmentation", SEGMENTATION_PAGE, value="segmentation"),
        ui.nav_panel("3D View", VIEW_3D_PAGE, value="view_3d"),
        ui.nav_panel("About", ABOUT_PAGE, value="about"),
        id="main_nav",
        selected="segmentation"
    ),

    theme=shinyswatch.theme.flatly()
)


# ================== Server ==================
def server(input, output, session):
    lang = reactive.Value("en")
    current_slice = reactive.Value(1)
    n_slices = reactive.Value(1)
    thumbnails = reactive.Value(None)

    def t(key: str) -> str:
        return TEXT.get(lang.get(), TEXT["en"]).get(key, key)

    def _to_unit_range(arr):
        arr = np.asarray(arr, dtype=np.float32)
        amax = float(np.nanmax(arr)) if arr.size else 0.0
        if amax > 1.5:
            arr = arr / 255.0
        if not np.isfinite(arr).all():
            arr = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)
        return arr

    def _auto_window(arr):
        arr = _to_unit_range(arr)
        lo = float(np.percentile(arr, 1.0))
        hi = float(np.percentile(arr, 99.0))
        if hi - lo < 1e-6:
            lo = float(arr.min())
            hi = float(arr.max())
        if hi - lo < 1e-6:
            hi = lo + 1e-6
        return arr, lo, hi

    def _render_slice_b64(base_img, pred=None, gt=None, combined=False):
        arr, lo, hi = _auto_window(base_img)
        arr = np.clip((arr - lo) / (hi - lo), 0.0, 1.0)
        rgb = np.stack([arr, arr, arr], axis=-1).astype(np.float32)

        if combined and pred is not None and gt is not None:
            p = pred > 0
            g = gt > 0
            both      = p & g
            pred_only = p & ~g
            gt_only   = g & ~p
            rgb[gt_only]   = rgb[gt_only]   * 0.25 + np.array([1.00, 0.15, 0.40]) * 0.75
            rgb[pred_only] = rgb[pred_only] * 0.25 + np.array([0.00, 0.85, 1.00]) * 0.75
            rgb[both]      = rgb[both]      * 0.18 + np.array([1.00, 0.95, 0.05]) * 0.82
        else:
            if pred is not None and np.any(pred):
                m = pred > 0
                rgb[m] = rgb[m] * 0.20 + np.array([0.00, 0.85, 1.00]) * 0.80
            if gt is not None and np.any(gt):
                m = gt > 0
                rgb[m] = rgb[m] * 0.20 + np.array([1.00, 0.15, 0.40]) * 0.80

        rgb = np.clip(rgb * 255.0, 0, 255).astype(np.uint8)
        img = Image.fromarray(rgb, mode="RGB")

        buf = io.BytesIO()
        img.save(buf, format="PNG", optimize=True)
        return base64.b64encode(buf.getvalue()).decode("ascii")

    def _slice_img_html(b64):
        return (f'<img src="data:image/png;base64,{b64}" '
                f'class="slice-img" alt="slice"/>')

    def _slice_error_html(msg):
        return f'<div class="slice-error">{html_lib.escape(msg)}</div>'

    def _make_thumbnail_b64(base_img, pred_mask):
        arr, lo, hi = _auto_window(base_img)
        arr = np.clip((arr - lo) / (hi - lo), 0.0, 1.0)
        rgb = np.stack([arr, arr, arr], axis=-1).astype(np.float32)
        if pred_mask is not None and np.any(pred_mask):
            m = pred_mask > 0
            rgb[m] = rgb[m] * 0.35 + np.array([0.00, 0.85, 1.00]) * 0.65
        rgb = np.clip(rgb * 255.0, 0, 255).astype(np.uint8)
        img = Image.fromarray(rgb, mode="RGB").resize((130, 130), Image.BILINEAR)
        buf = io.BytesIO()
        img.save(buf, format="PNG", optimize=True)
        return base64.b64encode(buf.getvalue()).decode("ascii")

    def _legend_html(items):
        parts = []
        for color, text in items:
            r, g, b = (int(c * 255) for c in color)
            parts.append(
                f'<span class="legend-item">'
                f'<span class="legend-dot" style="background:rgb({r},{g},{b})"></span>'
                f'<span>{html_lib.escape(text)}</span>'
                f'</span>'
            )
        return "".join(parts)

    def _metric_card_html(label, value, icon_svg=None, unit="",
                          pct=None, variant="similarity"):
        icon_html = f'<span class="metric-card-icon">{icon_svg}</span>' if icon_svg else ""
        unit_html = f'<span class="metric-card-unit">{html_lib.escape(unit)}</span>' if unit else ""

        bar_html = ""
        if pct is not None:
            try:
                p = max(0.0, min(1.0, float(pct)))
                bar_html = (
                    '<div class="metric-card-bar">'
                    f'<div class="metric-card-bar-fill" style="width:{p*100:.1f}%"></div>'
                    '</div>'
                )
            except Exception:
                bar_html = ""

        return (
            f'<div class="metric-card variant-{variant}">'
            f'<div class="metric-card-top">'
            f'{icon_html}'
            f'<span class="metric-card-label">{html_lib.escape(label)}</span>'
            f'</div>'
            f'<div class="metric-card-value">{html_lib.escape(value)}'
            f'{unit_html}</div>'
            f'{bar_html}'
            '</div>'
        )

    def _metric_grid_html(cards):
        return '<div class="metric-cards-grid">' + "".join(cards) + '</div>'

    def _empty_card_html(message):
        return f'<div class="metric-cards-empty">{html_lib.escape(message)}</div>'

    def _pred_mode_banner(key="metrics_pred_only"):
        en = TEXT["en"].get(key, "")
        fa = TEXT["fa"].get(key, "")
        return (
            '<div class="metric-cards-empty" '
            'style="margin-bottom:14px; font-style:normal; '
            'border-style:solid; background:rgba(6,182,212,0.10); '
            'color:var(--text-main); font-weight:500;">'
            f'<span class="en-text">{html_lib.escape(en)}</span>'
            f'<span class="fa-text">{html_lib.escape(fa)}</span>'
            '</div>'
        )

    def _n_axial(data):
        return int(data["X"].shape[0])

    def _axial_pixel_area_mm2(data):
        sp = data["final_spacing"]
        return float(sp[1]) * float(sp[2])

    def _slice_label_text():
        data = processed_data()
        if "error_key" in data:
            return f"{t('slice')} - / -"
        n = _n_axial(data)
        return f"{t('slice')} {current_slice.get()} / {n}"

    def _safe_idx():
        data = processed_data()
        if "error_key" in data:
            return None, data
        n = _n_axial(data)
        idx = current_slice.get() - 1
        idx = max(0, min(idx, n - 1))
        return idx, data

    @reactive.Effect
    @reactive.event(input.selected_lang)
    async def _change_lang():
        new_lang = input.selected_lang()
        if new_lang and new_lang in ("en", "fa") and new_lang != lang.get():
            lang.set(new_lang)
            await session.send_custom_message("set_lang", new_lang)

    @reactive.Effect
    @reactive.event(input.active_page)
    async def _change_page():
        page = input.active_page()
        if page in ("segmentation", "view_3d", "about"):
            try:
                ui.update_navset("main_nav", selected=page)
            except Exception as e:
                print(f"[nav] update error: {e}")

    @reactive.Calc
    def processed_data():
        if input.run() == 0:
            return {"error_key": "please_upload"}
        if model is None:
            return {"error_key": "model_not_loaded"}

        _t_start = time.perf_counter()

        img_files = input.dicom_files()
        if not img_files:
            return {"error_key": "no_dicom"}

        temp_dir = Path(tempfile.mkdtemp())
        for f in img_files:
            shutil.copy(f["datapath"], temp_dir / f["name"])

        img_paths = sorted(temp_dir.glob("*.dcm"),
                           key=lambda x: prep.extract_number(x.name))
        if not img_paths:
            img_paths = sorted(temp_dir.glob("*.dicom"),
                               key=lambda x: prep.extract_number(x.name))
        if not img_paths:
            return {"error_key": "no_valid_dicom"}

        slices, spacings = [], []
        for p in img_paths:
            img, sx, sy, sz = prep.read_dicom_with_spacing(p)
            slices.append(img)
            spacings.append((sz, sy, sx))
        img_volume = np.stack(slices, axis=0)
        spacing_zyx = np.mean(spacings, axis=0)

        modality = "CT"
        rows_orig = img_volume.shape[1]
        cols_orig = img_volume.shape[2]
        if _HAS_PYDICOM:
            try:
                ds = pydicom.dcmread(str(img_paths[0]), stop_before_pixels=True)
                modality = str(getattr(ds, "Modality", "CT")).strip() or "CT"
                r = int(getattr(ds, "Rows", rows_orig))
                c = int(getattr(ds, "Columns", cols_orig))
                if r > 0: rows_orig = r
                if c > 0: cols_orig = c
            except Exception as e:
                print(f"[dicom metadata] {e}")

        n_slices_orig = int(img_volume.shape[0])

        processed_vol, final_spacing = prep.preprocess_volume(img_volume, spacing_zyx)
        X = processed_vol[..., np.newaxis].astype(np.float32) / 255.0

        probs = model.predict(X, batch_size=8, verbose=0)
        pred_masks = (probs[..., 0] > 0.5).astype(np.uint8)

        gt_masks = None
        mask_files = input.mask_files()
        if mask_files:
            mask_temp_dir = Path(tempfile.mkdtemp())
            for f in mask_files:
                shutil.copy(f["datapath"], mask_temp_dir / f["name"])
            mask_paths = sorted(mask_temp_dir.glob("*.dcm"),
                                key=lambda x: prep.extract_number(x.name))
            if not mask_paths:
                mask_paths = sorted(mask_temp_dir.glob("*.dicom"),
                                    key=lambda x: prep.extract_number(x.name))
            if mask_paths:
                mask_slices = []
                for p in mask_paths:
                    m, _, _, _ = prep.read_dicom_with_spacing(p)
                    mask_slices.append((m > 0).astype(np.uint8) * 255)
                mask_volume = np.stack(mask_slices, axis=0)

                from scipy.ndimage import zoom
                mask_resampled = zoom(
                    mask_volume,
                    (spacing_zyx[0] / 1.0, spacing_zyx[1] / 1.0, spacing_zyx[2] / 1.0),
                    order=0,
                )
                mask_resampled = (mask_resampled > 0.5).astype(np.uint8) * 255
                mask_cropped = mask_resampled[
                    :,
                    prep.CROP_TOP:mask_resampled.shape[1] - prep.CROP_BOTTOM,
                    prep.CROP_LEFT:mask_resampled.shape[2] - prep.CROP_RIGHT,
                ]
                gt_processed = []
                for i in range(mask_cropped.shape[0]):
                    m = prep.resize_or_pad_to_target_mask(
                        mask_cropped[i], prep.FINAL_SIZE)
                    gt_processed.append(m)
                gt_masks = np.stack(gt_processed, axis=0)
                gt_masks = (gt_masks > 127).astype(np.uint8)

        _t_end = time.perf_counter()
        _processing_time_s = _t_end - _t_start

        return {
            "X": X,
            "pred_masks": pred_masks,
            "gt_masks": gt_masks,
            "final_spacing": final_spacing,
            "original_volume": img_volume,
            "processed_volume": processed_vol,
            "n_slices": int(X.shape[0]),
            "n_slices_orig": n_slices_orig,
            "rows_orig": int(rows_orig),
            "cols_orig": int(cols_orig),
            "modality": modality,
            "processing_time_s": _processing_time_s,
        }

    @output
    @render.ui
    def dicom_info_strip():
        data = processed_data()
        if "error_key" in data:
            return ui.HTML(
                '<div class="dicom-info-strip">'
                f'<div class="info-empty">{html_lib.escape(t(data["error_key"]))}</div>'
                '</div>'
            )
        n_sl       = data.get("n_slices", 0)
        n_sl_orig  = data.get("n_slices_orig", 0)
        r          = data.get("rows_orig", 0)
        c          = data.get("cols_orig", 0)
        mod        = data.get("modality", "CT")
        proc_t     = data.get("processing_time_s", 0.0)

        html = (
            '<div class="dicom-info-strip">'
            f'<span class="info-icon">{INFO_ICON_SVG}</span>'
            f'<span class="info-item">'
            f'<span class="info-label">{html_lib.escape(t("info_slices_orig"))}:</span>'
            f'<span class="info-value">{n_sl_orig}</span>'
            f'</span>'
            f'<span class="info-sep">|</span>'
            f'<span class="info-item">'
            f'<span class="info-label">{html_lib.escape(t("info_slices_processed"))}:</span>'
            f'<span class="info-value">{n_sl}</span>'
            f'</span>'
            f'<span class="info-sep">|</span>'
            f'<span class="info-item">'
            f'<span class="info-label">{html_lib.escape(t("info_size"))}:</span>'
            f'<span class="info-value">{r} × {c}</span>'
            f'</span>'
            f'<span class="info-sep">|</span>'
            f'<span class="info-item">'
            f'<span class="info-label">{html_lib.escape(t("info_modality"))}:</span>'
            f'<span class="info-value">{html_lib.escape(mod)}</span>'
            f'</span>'
            f'<span class="info-sep">|</span>'
            f'<span class="info-item">'
            f'<span class="info-label">{html_lib.escape(t("proc_time"))}:</span>'
            f'<span class="info-value info-time">{proc_t:.2f} s</span>'
            f'</span>'
            '</div>'
        )
        return ui.HTML(html)

    @reactive.Effect
    def _build_thumbnails():
        data = processed_data()
        if "error_key" in data:
            thumbnails.set(None)
            n_slices.set(1)
            return
        X = data["X"][..., 0]
        pred = data["pred_masks"]
        n = _n_axial(data)
        thumbs = []
        for i in range(n):
            thumbs.append(_make_thumbnail_b64(X[i], pred[i]))
        thumbnails.set(thumbs)
        n_slices.set(n)
        if current_slice.get() > n:
            current_slice.set(n)

    @reactive.Effect
    @reactive.event(input.filmstrip_click)
    def _filmstrip_click():
        payload = input.filmstrip_click()
        if not payload: return
        try:
            idx = int(payload.get("idx"))
        except (TypeError, ValueError, AttributeError):
            return
        if 1 <= idx <= n_slices.get():
            current_slice.set(idx)

    @reactive.Effect
    @reactive.event(input.filmstrip_nav_prev)
    def _filmstrip_prev():
        cur = current_slice.get()
        if cur > 1: current_slice.set(cur - 1)

    @reactive.Effect
    @reactive.event(input.filmstrip_nav_next)
    def _filmstrip_next():
        cur = current_slice.get()
        if cur < n_slices.get(): current_slice.set(cur + 1)

    @reactive.Effect
    @reactive.event(input.jump_go)
    def _jump_to_slice():
        val = input.jump_input()
        if val is None: return
        s = str(val).strip()
        if s == "": return
        try: target = int(s)
        except ValueError: return
        if 1 <= target <= n_slices.get():
            current_slice.set(target)
            try: ui.update_text("jump_input", value="")
            except Exception: pass

    @output
    @render.ui
    def slice_filmstrip():
        thumbs = thumbnails.get()
        if thumbs is None:
            return ui.HTML('<div class="filmstrip">'
                           '<button class="filmstrip-arrow prev" disabled>‹</button>'
                           '<div class="filmstrip-track">'
                           '<div class="filmstrip-empty">—</div></div>'
                           '<button class="filmstrip-arrow next" disabled>›</button>'
                           '</div>')
        n = len(thumbs)
        cur = current_slice.get()
        window = 6
        half = window // 2
        start = cur - half
        end = cur + half - 1
        if start < 1:
            start = 1; end = min(n, window)
        if end > n:
            end = n; start = max(1, n - window + 1)
        items = []
        for i in range(start, end + 1):
            cls = "filmstrip-item current" if i == cur else "filmstrip-item"
            items.append(
                f'<div class="{cls}" data-slice="{i}">'
                f'<img src="data:image/png;base64,{thumbs[i-1]}"/>'
                f'<div class="filmstrip-num">{i}</div></div>'
            )
        if end < n:
            items.append('<div class="filmstrip-ellipsis">…</div>')
        ld = "disabled" if cur <= 1 else ""
        rd = "disabled" if cur >= n else ""
        html = (
            '<div class="filmstrip">'
            f'<button class="filmstrip-arrow prev" {ld}>‹</button>'
            '<div class="filmstrip-track">' + "".join(items) + '</div>'
            f'<button class="filmstrip-arrow next" {rd}>›</button>'
            '</div>'
        )
        return ui.HTML(html)

    @output
    @render.text
    def slice_lbl_original(): return _slice_label_text()
    @output
    @render.text
    def slice_lbl_pred(): return _slice_label_text()
    @output
    @render.text
    def slice_lbl_gt(): return _slice_label_text()
    @output
    @render.text
    def slice_lbl_combined(): return _slice_label_text()

    @output
    @render.ui
    def plot_original():
        idx, data = _safe_idx()
        if idx is None:
            return ui.HTML(_slice_error_html(t(data["error_key"])))
        base = data["X"][idx, :, :, 0]
        b64 = _render_slice_b64(base)
        return ui.HTML(_slice_img_html(b64))

    @output
    @render.ui
    def plot_pred_overlay():
        idx, data = _safe_idx()
        if idx is None:
            return ui.HTML(_slice_error_html(t(data["error_key"])))
        base = data["X"][idx, :, :, 0]
        pred = data["pred_masks"][idx].astype(np.float32)
        b64 = _render_slice_b64(base, pred=pred)
        return ui.HTML(_slice_img_html(b64))

    @output
    @render.ui
    def plot_gt_overlay():
        idx, data = _safe_idx()
        if idx is None:
            return ui.HTML(_slice_error_html(t(data["error_key"])))
        if data["gt_masks"] is None:
            return ui.HTML(_slice_error_html(t("no_gt_uploaded")))
        base = data["X"][idx, :, :, 0]
        gt   = data["gt_masks"][idx].astype(np.float32)
        b64 = _render_slice_b64(base, gt=gt)
        return ui.HTML(_slice_img_html(b64))

    @output
    @render.ui
    def plot_combined():
        idx, data = _safe_idx()
        if idx is None:
            return ui.HTML(_slice_error_html(t(data["error_key"])))
        if data["gt_masks"] is None:
            return ui.HTML(_slice_error_html(t("no_gt_uploaded")))
        base = data["X"][idx, :, :, 0]
        pred = data["pred_masks"][idx].astype(np.float32)
        gt   = data["gt_masks"][idx].astype(np.float32)
        b64 = _render_slice_b64(base, pred=pred, gt=gt, combined=True)
        return ui.HTML(_slice_img_html(b64))

    @output
    @render.ui
    def legend_pred():
        return ui.HTML(_legend_html([
            ((0.00, 0.85, 1.00), t("legend_liver")),
            ((0.45, 0.45, 0.45), t("legend_background")),
        ]))

    @output
    @render.ui
    def legend_gt():
        return ui.HTML(_legend_html([
            ((1.00, 0.15, 0.40), t("legend_liver")),
            ((0.45, 0.45, 0.45), t("legend_background")),
        ]))

    @output
    @render.ui
    def legend_combined():
        return ui.HTML(_legend_html([
            ((1.00, 0.95, 0.05), t("legend_tp")),
            ((0.00, 0.85, 1.00), t("legend_fp")),
            ((1.00, 0.15, 0.40), t("legend_fn")),
            ((0.45, 0.45, 0.45), t("legend_background")),
        ]))

    # ============ 3D ============
    def _get_mesh(volume, sigma=1.2):
        vol = np.asarray(volume, dtype=np.float32)
        if vol.size == 0:
            return None, None
        if float(vol.max()) < 0.01:
            return None, None
        try:
            vol_smooth = gaussian_filter(vol, sigma=float(sigma))
        except Exception as e:
            print(f"[3D] gaussian_filter error: {e}")
            vol_smooth = vol
        if float(vol_smooth.max()) < 0.01:
            return None, None
        try:
            verts, faces, _, _ = measure.marching_cubes(vol_smooth, level=0.5)
        except (ValueError, RuntimeError) as e:
            print(f"[3D] marching_cubes error: {e}")
            return None, None
        except Exception as e:
            print(f"[3D] unexpected error: {e}")
            return None, None
        if len(verts) == 0 or len(faces) == 0:
            return None, None
        return verts, faces

    def _draw_mesh(ax, verts, faces, color, alpha=0.95):
        if verts is None or faces is None:
            ax.text(0.5, 0.5, 0.5, "Empty",
                    ha="center", va="center",
                    fontsize=12, color="#94a3b8")
            ax.set_axis_off()
            return
        try:
            ax.plot_trisurf(
                verts[:, 0], verts[:, 1], faces, verts[:, 2],
                color=color, alpha=float(alpha),
                linewidth=0, antialiased=True, shade=True,
            )
        except Exception as e:
            print(f"[3D] plot_trisurf error: {e}")
            ax.text(0.5, 0.5, 0.5, "Error",
                    ha="center", va="center",
                    fontsize=12, color="#ef4444")
        ax.set_axis_off()

    @reactive.Calc
    def cached_meshes():
        data = processed_data()
        if "error_key" in data:
            return None

        sigma = 1.2

        gt_vol = data["gt_masks"]
        pr_vol = data["pred_masks"]
        if gt_vol is None:
            return {"gt": None, "pr": None, "df": None}

        gt_v = gt_vol.astype(np.uint8)
        pr_v = pr_vol.astype(np.uint8)
        df_v = np.logical_xor(gt_v > 0, pr_v > 0).astype(np.uint8)

        gt_verts, gt_faces = _get_mesh(gt_v, sigma=sigma)
        pr_verts, pr_faces = _get_mesh(pr_v, sigma=sigma)
        df_verts, df_faces = _get_mesh(df_v, sigma=max(0.5, sigma * 0.7))

        return {
            "gt": (gt_verts, gt_faces),
            "pr": (pr_verts, pr_faces),
            "df": (df_verts, df_faces)
        }

    def _draw_mesh_cached(ax, mesh_data, color, alpha=0.95):
        verts, faces = mesh_data
        if verts is None or faces is None:
            ax.text(0.5, 0.5, 0.5, "Empty", ha="center", va="center", fontsize=12, color="#94a3b8")
            ax.set_axis_off()
            return
        try:
            ax.plot_trisurf(
                verts[:, 0], verts[:, 1], faces, verts[:, 2],
                color=color, alpha=float(alpha),
                linewidth=0, antialiased=True, shade=True,
            )
        except Exception as e:
            print(f"[3D] plot_trisurf error: {e}")
            ax.text(0.5, 0.5, 0.5, "Error", ha="center", va="center", fontsize=12, color="#ef4444")
        ax.set_axis_off()

    @output
    @render.ui
    def plot_3d():
        data = processed_data()
        if "error_key" in data:
            return ui.HTML(
                '<div class="view3d-empty">'
                f'{CUBE_ICON_SVG}'
                f'<div class="msg">{html_lib.escape(t("no_3d_data"))}</div>'
                '</div>'
            )

        meshes = cached_meshes()
        alpha = 0.95

        try:
            show_diff = bool(input.show_3d_diff()) if hasattr(input, "show_3d_diff") else True
        except Exception:
            show_diff = True

        if data["gt_masks"] is None:
            pr_data = meshes["pr"] if meshes else (None, None)
            if pr_data[0] is None:
                return ui.HTML(
                    '<div class="view3d-empty">'
                    f'{CUBE_ICON_SVG}'
                    f'<div class="msg">{html_lib.escape(t("3d_pred_missing"))}</div>'
                    '</div>'
                )

            title = ("Prediction (no reference)" if lang.get() == "en" else "پیش‌بینی (بدون مرجع)")
            fig = Figure(figsize=(8, 6.5), facecolor="#020617")
            FigureCanvasAgg(fig)
            ax = fig.add_subplot(111, projection="3d")
            _draw_mesh_cached(ax, pr_data, color="#fb7185", alpha=alpha)
            ax.set_title(title, fontsize=13, fontweight="bold", color="#e2e8f0", pad=10)
            ax.set_facecolor("#020617")
            try: fig.tight_layout()
            except Exception: pass

            buf = io.BytesIO()
            fig.savefig(buf, format="png", dpi=100, facecolor="#020617", bbox_inches="tight")
            buf.seek(0)
            b64 = base64.b64encode(buf.read()).decode("ascii")
            try: plt.close(fig)
            except Exception: pass

            return ui.HTML(f'<img src="data:image/png;base64,{b64}" class="view3d-image" alt="3d-view"/>')

        gt_data = meshes["gt"] if meshes else (None, None)
        pr_data = meshes["pr"] if meshes else (None, None)
        df_data = meshes["df"] if meshes else (None, None)

        if pr_data[0] is None and gt_data[0] is None:
            return ui.HTML(
                '<div class="view3d-empty">'
                f'{CUBE_ICON_SVG}'
                f'<div class="msg">{html_lib.escape(t("3d_pred_missing"))}</div>'
                '</div>'
            )

        title_gt = "Ground Truth" if lang.get() == "en" else "مرجع"
        title_pr = "Prediction" if lang.get() == "en" else "پیش‌بینی"
        title_df = "Difference" if lang.get() == "en" else "تفاوت"
        main_title = ("3D Qualitative Comparison of Liver Segmentation"
                      if lang.get() == "en"
                      else "مقایسهٔ کیفی سه‌بعدی تقسیم‌بندی کبد")

        n_panels = 3 if show_diff else 2
        fig = Figure(figsize=(6.5 * n_panels, 6.0), facecolor="#020617")
        FigureCanvasAgg(fig)

        ax1 = fig.add_subplot(1, n_panels, 1, projection="3d")
        _draw_mesh_cached(ax1, gt_data, color="#22d3ee", alpha=alpha)
        ax1.set_title(title_gt, fontsize=13, fontweight="bold", color="#e2e8f0", pad=10)
        ax1.set_facecolor("#020617")

        ax2 = fig.add_subplot(1, n_panels, 2, projection="3d")
        _draw_mesh_cached(ax2, pr_data, color="#fb7185", alpha=alpha)
        ax2.set_title(title_pr, fontsize=13, fontweight="bold", color="#e2e8f0", pad=10)
        ax2.set_facecolor("#020617")

        if show_diff:
            ax3 = fig.add_subplot(1, n_panels, 3, projection="3d")
            _draw_mesh_cached(ax3, df_data, color="#fbbf24", alpha=alpha)
            ax3.set_title(title_df, fontsize=13, fontweight="bold", color="#e2e8f0", pad=10)
            ax3.set_facecolor("#020617")

        try: fig.suptitle(main_title, fontsize=15, fontweight="bold", color="#e2e8f0", y=0.98)
        except Exception: pass
        try: fig.tight_layout(rect=[0, 0, 1, 0.96])
        except Exception: pass

        buf = io.BytesIO()
        fig.savefig(buf, format="png", dpi=100, facecolor="#020617", bbox_inches="tight")
        buf.seek(0)
        b64 = base64.b64encode(buf.read()).decode("ascii")
        try: plt.close(fig)
        except Exception: pass

        return ui.HTML(f'<img src="data:image/png;base64,{b64}" class="view3d-image" alt="3d-view"/>')

    @output
    @render.ui
    def view3d_info():
        data = processed_data()
        if "error_key" in data:
            return ui.HTML(
                '<div class="view3d-info">'
                '<div class="view3d-info-title">Info</div>'
                '<div class="view3d-info-row">'
                '<span class="k">Status</span>'
                '<span class="v">—</span>'
                '</div></div>'
            )
        pred_vox = int(np.sum(data["pred_masks"]))

        def _row(k_en, k_fa, v):
            return (
                '<div class="view3d-info-row">'
                f'<span class="k">'
                f'<span class="en-text">{html_lib.escape(k_en)}</span>'
                f'<span class="fa-text">{html_lib.escape(k_fa)}</span>'
                f'</span>'
                f'<span class="v">{html_lib.escape(str(v))}</span>'
                '</div>'
            )

        title_en = TEXT["en"]["3d_info"]
        title_fa = TEXT["fa"]["3d_info"]

        if data["gt_masks"] is None:
            html = (
                '<div class="view3d-info">'
                f'<div class="view3d-info-title">'
                f'<span class="en-text">{html_lib.escape(title_en)}</span>'
                f'<span class="fa-text">{html_lib.escape(title_fa)}</span>'
                f'</div>'
                + _row("Pred Voxels", "وکسل پیش‌بینی", f"{pred_vox:,}")
                + '</div>'
            )
            return ui.HTML(html)

        gt_vox = int(np.sum(data["gt_masks"]))
        diff_vox = int(np.sum(np.logical_xor(
            data["gt_masks"] > 0, data["pred_masks"] > 0
        )))

        html = (
            '<div class="view3d-info">'
            f'<div class="view3d-info-title">'
            f'<span class="en-text">{html_lib.escape(title_en)}</span>'
            f'<span class="fa-text">{html_lib.escape(title_fa)}</span>'
            f'</div>'
            + _row("GT Voxels", "وکسل مرجع", f"{gt_vox:,}")
            + _row("Pred Voxels", "وکسل پیش‌بینی", f"{pred_vox:,}")
            + _row("Diff Voxels", "وکسل تفاوت", f"{diff_vox:,}")
            + '</div>'
        )
        return ui.HTML(html)

    # ============ Metrics ============
    def _build_metrics_html():
        data = processed_data()
        if "error_key" in data:
            return _empty_card_html(t(data["error_key"]))

        if data["gt_masks"] is None:
            pred = data["pred_masks"]
            pred_vox = int(np.sum(pred))
            total_vox = int(pred.size)
            slices_any = pred.reshape(pred.shape[0], -1).any(axis=1)
            slices_with_pred = int(np.sum(slices_any))
            n_sl = int(pred.shape[0])
            coverage = (pred_vox / total_vox * 100.0) if total_vox > 0 else 0.0

            sp = data["final_spacing"]
            try:
                voxel_vol_mm3 = float(sp[0]) * float(sp[1]) * float(sp[2])
            except Exception:
                voxel_vol_mm3 = 1.0
            volume_ml = pred_vox * voxel_vol_mm3 / 1000.0

            cards = [
                _metric_card_html(t("pred_voxels"), f"{pred_vox:,}",
                                  icon_svg=ICON_DICE, variant="similarity"),
                _metric_card_html(t("est_volume"), f"{volume_ml:.1f}",
                                  icon_svg=ICON_IOU, unit="mL",
                                  variant="similarity"),
                _metric_card_html(t("slices_with_liver"),
                                  f"{slices_with_pred} / {n_sl}",
                                  icon_svg=ICON_SLICE, variant="slice"),
                _metric_card_html(t("coverage"), f"{coverage:.2f}",
                                  icon_svg=ICON_PRECISION, unit="%",
                                  variant="detection"),
                _metric_card_html(t("dice"), t("na"),
                                  icon_svg=ICON_DICE, variant="distance"),
                _metric_card_html(t("hd95"), t("na"),
                                  icon_svg=ICON_HD95, variant="distance"),
            ]
            return _pred_mode_banner() + _metric_grid_html(cards)

        dice = mt.dice_3d(data["gt_masks"], data["pred_masks"])
        iou = mt.iou_3d(data["gt_masks"], data["pred_masks"])
        prec = mt.precision_3d(data["gt_masks"], data["pred_masks"])
        rec = mt.recall_3d(data["gt_masks"], data["pred_masks"])
        hd95 = mt.calculate_hd95_physical(
            data["gt_masks"], data["pred_masks"], data["final_spacing"])
        haus = mt.calculate_hausdorff_physical(
            data["gt_masks"], data["pred_masks"], data["final_spacing"])
        return _metric_grid_html([
            _metric_card_html(t("dice"), f"{dice:.3f}",
                              icon_svg=ICON_DICE, pct=dice, variant="similarity"),
            _metric_card_html(t("iou"), f"{iou:.3f}",
                              icon_svg=ICON_IOU, pct=iou, variant="similarity"),
            _metric_card_html(t("precision"), f"{prec:.3f}",
                              icon_svg=ICON_PRECISION, pct=prec, variant="detection"),
            _metric_card_html(t("recall"), f"{rec:.3f}",
                              icon_svg=ICON_RECALL, pct=rec, variant="detection"),
            _metric_card_html(t("hd95"), f"{hd95:.2f}",
                              icon_svg=ICON_HD95, unit="mm", variant="distance"),
            _metric_card_html(t("hausdorff"), f"{haus:.2f}",
                              icon_svg=ICON_AHD, unit="mm", variant="distance"),
        ])

    @output
    @render.ui
    def metrics_cards():
        return ui.HTML(_build_metrics_html())

    @output
    @render.ui
    def metrics_3d_cards():
        return ui.HTML(_build_metrics_html())

    @output
    @render.ui
    def slice_metrics_cards():
        data = processed_data()
        if "error_key" in data:
            return ui.HTML(_empty_card_html(t(data["error_key"])))

        n = _n_axial(data)
        idx = max(0, min(current_slice.get() - 1, n - 1))
        pixel_area_mm2 = _axial_pixel_area_mm2(data)

        pred_s = data["pred_masks"][idx].astype(bool)

        if data["gt_masks"] is None:
            n_pred = int(pred_s.sum())
            total_px = int(pred_s.size)
            coverage = (n_pred / total_px * 100.0) if total_px > 0 else 0.0
            area_mm2 = n_pred * pixel_area_mm2

            cards = [
                _metric_card_html(t("slice_num"), f"{idx + 1} / {n}",
                                  icon_svg=ICON_SLICE, variant="slice"),
                _metric_card_html(t("pred_pixels"), f"{n_pred:,}",
                                  icon_svg=ICON_DICE, variant="similarity"),
                _metric_card_html(t("area"), f"{area_mm2:.1f}",
                                  icon_svg=ICON_IOU, unit="mm²",
                                  variant="similarity"),
                _metric_card_html(t("coverage"), f"{coverage:.2f}",
                                  icon_svg=ICON_PRECISION, unit="%",
                                  variant="detection"),
                _metric_card_html(t("dice"), t("na"),
                                  icon_svg=ICON_DICE, variant="distance"),
                _metric_card_html(t("hausdorff_2d"), t("na"),
                                  icon_svg=ICON_AHD, variant="distance"),
            ]
            return ui.HTML(_pred_mode_banner("slice_pred_only") +
                           _metric_grid_html(cards))

        gt_s = data["gt_masks"][idx].astype(bool)

        tp = int(np.logical_and(gt_s, pred_s).sum())
        fp = int(np.logical_and(~gt_s, pred_s).sum())
        fn = int(np.logical_and(gt_s, ~pred_s).sum())
        dice = (2 * tp / (2 * tp + fp + fn)) if (2 * tp + fp + fn) > 0 else 0.0
        iou  = (tp / (tp + fp + fn)) if (tp + fp + fn) > 0 else 0.0
        prec = (tp / (tp + fp)) if (tp + fp) > 0 else 0.0
        rec  = (tp / (tp + fn)) if (tp + fn) > 0 else 0.0

        if gt_s.any() and pred_s.any():
            from scipy.ndimage import binary_erosion, distance_transform_edt
            struct = np.ones((3, 3), dtype=bool)
            gt_border   = gt_s   ^ binary_erosion(gt_s,   structure=struct)
            pred_border = pred_s ^ binary_erosion(pred_s, structure=struct)
            if gt_border.any() and pred_border.any():
                dt_pred = distance_transform_edt(~pred_border)
                dt_gt   = distance_transform_edt(~gt_border)
                hd_pix  = max(dt_pred[gt_border].max(), dt_gt[pred_border].max())
                hd_mm   = hd_pix * float(np.sqrt(pixel_area_mm2))
            else:
                hd_mm = float("nan")
        else:
            hd_mm = float("nan")

        hd_str = f"{hd_mm:.2f}" if not np.isnan(hd_mm) else t("na")

        return ui.HTML(_metric_grid_html([
            _metric_card_html(t("slice_num"), f"{idx + 1} / {n}",
                              icon_svg=ICON_SLICE, variant="slice"),
            _metric_card_html(t("dice"), f"{dice:.3f}",
                              icon_svg=ICON_DICE, pct=dice, variant="similarity"),
            _metric_card_html(t("iou"), f"{iou:.3f}",
                              icon_svg=ICON_IOU, pct=iou, variant="similarity"),
            _metric_card_html(t("precision"), f"{prec:.3f}",
                              icon_svg=ICON_PRECISION, pct=prec, variant="detection"),
            _metric_card_html(t("recall"), f"{rec:.3f}",
                              icon_svg=ICON_RECALL, pct=rec, variant="detection"),
            _metric_card_html(t("hausdorff_2d"), hd_str,
                              icon_svg=ICON_AHD, unit="mm", variant="distance"),
        ]))


app = App(app_ui, server)

if __name__ == "__main__":
    app.run()
