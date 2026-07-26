"""
Video preprocessing pipeline for DeepGuard.

Extracts 8 face-cropped frames from a video, computes RGB tensors and DCT
frequency features for the two-stream model. Adapted from the original
Streamlit app.py — Streamlit-specific caching removed.
"""

import cv2
import numpy as np
import torch
from torchvision import transforms
from scipy.fftpack import dct
from ultralytics import YOLO
from django.conf import settings
import threading


# --- Singleton face detector ---
_face_detector = None
_detector_lock = threading.Lock()


def get_face_detector():
    """Load YOLOv8 face detector once (thread-safe)."""
    global _face_detector
    if _face_detector is None:
        with _detector_lock:
            if _face_detector is None:
                _face_detector = YOLO(settings.DEEPGUARD_FACE_DETECTOR_PATH)
    return _face_detector


# ImageNet normalisation (same as original app.py)
_transform = transforms.Normalize([0.45, 0.45, 0.45], [0.225, 0.225, 0.225])


def extract_dct_features(face_rgb):
    """
    Extract DCT high-frequency features from a face crop.

    Args:
        face_rgb: (224, 224, 3) uint8 RGB numpy array

    Returns:
        (1, 224, 224) float32 tensor — normalised DCT high-frequency map
    """
    ycrcb = cv2.cvtColor(face_rgb, cv2.COLOR_RGB2YCrCb)
    y_channel = ycrcb[:, :, 0]
    dct_y = dct(dct(y_channel.T, norm='ortho').T, norm='ortho')
    high_freq = np.abs(dct_y[112:, 112:])
    tensor = torch.from_numpy(cv2.resize(high_freq, (224, 224))).unsqueeze(0).float()
    return tensor / (torch.max(tensor) + 1e-8)


def process_video(video_path):
    """
    Process a video file for DeepGuard inference.

    Extracts 8 evenly-spaced frames, detects faces, crops and resizes them,
    and computes both RGB and DCT tensors.

    Args:
        video_path: Path to the video file (str or Path)

    Returns:
        tuple: (rgb_tensor, dct_tensor, display_crops) or (None, None, None)
            - rgb_tensor: (1, 8, 3, 224, 224) float32
            - dct_tensor: (1, 8, 1, 224, 224) float32
            - display_crops: list of 8 (224, 224, 3) uint8 RGB arrays for UI display
    """
    face_detector = get_face_detector()
    cap = cv2.VideoCapture(str(video_path))

    if not cap.isOpened():
        return None, None, None

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total_frames < 8:
        cap.release()
        return None, None, None

    indices = np.linspace(0, total_frames - 1, 8, dtype=int)
    rgb_tensors, dct_tensors, display_crops = [], [], []
    last_box = None

    for idx in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        if not ret:
            continue

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = face_detector(frame_rgb, verbose=False, device='cpu')

        if len(results[0].boxes) > 0:
            last_box = results[0].boxes[0].xyxy.cpu().numpy()[0]

        if last_box is not None:
            x1, y1, x2, y2 = map(int, last_box)
            pad_w, pad_h = int((x2 - x1) * 0.20), int((y2 - y1) * 0.20)
            y1_pad = max(0, y1 - pad_h)
            y2_pad = min(frame_rgb.shape[0], y2 + pad_h)
            x1_pad = max(0, x1 - pad_w)
            x2_pad = min(frame_rgb.shape[1], x2 + pad_w)

            face_crop = frame_rgb[y1_pad:y2_pad, x1_pad:x2_pad]
            if face_crop.size == 0:
                continue

            face_resized = cv2.resize(face_crop, (224, 224))
            display_crops.append(face_resized)

            tensor_rgb = _transform(
                torch.from_numpy(face_resized).permute(2, 0, 1).float() / 255.0
            )
            rgb_tensors.append(tensor_rgb)
            dct_tensors.append(extract_dct_features(face_resized))

    cap.release()

    if len(rgb_tensors) == 8:
        return (
            torch.stack(rgb_tensors).unsqueeze(0),   # (1, 8, 3, 224, 224)
            torch.stack(dct_tensors).unsqueeze(0),    # (1, 8, 1, 224, 224)
            display_crops
        )
    return None, None, None
