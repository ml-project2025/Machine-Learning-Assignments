
import cv2
import numpy as np
from scipy.ndimage import center_of_mass

def _binarize(gray: np.ndarray):
    g = cv2.GaussianBlur(gray, (3, 3), 0)
    th = cv2.adaptiveThreshold(g, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                               cv2.THRESH_BINARY, 21, 2)
    if np.mean(th) > 127:
        th = 255 - th
    th = cv2.morphologyEx(th, cv2.MORPH_OPEN, np.ones((3,3), np.uint8), iterations=1)
    return th

def _fit_to_box(th: np.ndarray, size=8) -> np.ndarray:
    ys, xs = np.where(th > 0)
    if len(xs) == 0 or len(ys) == 0:
        return np.zeros((size, size), np.float32)

    x1, x2 = xs.min(), xs.max()
    y1, y2 = ys.min(), ys.max()
    roi = th[y1:y2+1, x1:x2+1]

    h, w = roi.shape
    if h > w:
        new_h = size - 2
        new_w = max(1, int(round(w * new_h / h)))
    else:
        new_w = size - 2
        new_h = max(1, int(round(h * new_w / w)))

    roi_resized = cv2.resize(roi, (new_w, new_h), interpolation=cv2.INTER_AREA)
    canvas = np.zeros((size, size), dtype=np.uint8)
    y_off = (size - new_h) // 2
    x_off = (size - new_w) // 2
    canvas[y_off:y_off+new_h, x_off:x_off+new_w] = roi_resized
    return canvas

def _center_image(img: np.ndarray) -> np.ndarray:
    cy, cx = center_of_mass(img > 0)
    if np.isnan(cx) or np.isnan(cy):
        return img
    cy, cx = int(cy), int(cx)
    shift_y = img.shape[0] // 2 - cy
    shift_x = img.shape[1] // 2 - cx
    M = np.float32([[1, 0, shift_x], [0, 1, shift_y]])
    return cv2.warpAffine(img, M, (img.shape[1], img.shape[0]))

def to_8x8(gray: np.ndarray, invert: bool = False) -> np.ndarray:
    th = _binarize(gray)
    if invert:
        th = 255 - th
    fitted = _fit_to_box(th, size=8)
    centered = _center_image(fitted)
    small = (centered / 255.0) * 16.0
    return small.astype(np.float32)

def load_and_prepare(path: str, invert: bool = False):
    gray = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if gray is None:
        raise FileNotFoundError(f"لم أستطع قراءة الصورة: {path}")
    img8 = to_8x8(gray, invert=invert)
    flat = img8.reshape(1, -1)
    return flat, img8
