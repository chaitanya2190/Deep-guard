"""
DeepGuard Inference Engine

ONNX-based student model inference + PyTorch autoencoder for anomaly/heatmap generation.
Singleton pattern ensures models load once at Django startup.
"""

import threading
import numpy as np
import torch
import torch.nn as nn
import onnxruntime as ort
from django.conf import settings


class SpatialTemporalAutoencoder(nn.Module):
    """Autoencoder for anomaly detection and heatmap generation."""
    def __init__(self):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv3d(3, 16, kernel_size=3, stride=2, padding=1), nn.ReLU(),
            nn.Conv3d(16, 32, kernel_size=3, stride=2, padding=1), nn.ReLU(),
            nn.Conv3d(32, 64, kernel_size=3, stride=2, padding=1), nn.ReLU()
        )
        self.decoder = nn.Sequential(
            nn.ConvTranspose3d(64, 32, kernel_size=3, stride=2, padding=1, output_padding=1), nn.ReLU(),
            nn.ConvTranspose3d(32, 16, kernel_size=3, stride=2, padding=1, output_padding=1), nn.ReLU(),
            nn.ConvTranspose3d(16, 3, kernel_size=3, stride=2, padding=1, output_padding=1)
        )

    def forward(self, x):
        return self.decoder(self.encoder(x))


class DeepGuardEngine:
    """
    Singleton inference engine.
    - Student model runs via ONNX Runtime (CPU-optimised, 427MB)
    - Autoencoder runs via PyTorch (CPU, 570KB) for anomaly + heatmap
    """
    _instance = None
    _lock = threading.Lock()

    def __new__(cls):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
                    cls._instance._initialized = False
        return cls._instance

    def _load(self):
        if self._initialized:
            return

        # ONNX Runtime session for student model (CPU)
        sess_options = ort.SessionOptions()
        sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        sess_options.intra_op_num_threads = 4
        self.onnx_session = ort.InferenceSession(
            settings.DEEPGUARD_ONNX_MODEL_PATH,
            sess_options,
            providers=['CPUExecutionProvider']
        )
        self.onnx_input_names = [inp.name for inp in self.onnx_session.get_inputs()]
        self.onnx_output_names = [out.name for out in self.onnx_session.get_outputs()]

        # PyTorch autoencoder (tiny, CPU is fine)
        self.device = torch.device('cpu')
        self.autoencoder = SpatialTemporalAutoencoder().to(self.device)
        self.autoencoder.load_state_dict(
            torch.load(settings.DEEPGUARD_AUTOENCODER_PATH, map_location=self.device, weights_only=True)
        )
        self.autoencoder.eval()
        self.mse = nn.MSELoss(reduction='none')

        self._initialized = True

    def predict(self, rgb_tensor, dct_tensor, threshold=0.15):
        """
        Run inference on preprocessed tensors.

        Args:
            rgb_tensor: (1, 8, 3, 224, 224) float32
            dct_tensor: (1, 8, 1, 224, 224) float32
            threshold: MSE threshold for anomaly detection

        Returns:
            dict with deepfake_probability, reconstruction_error, is_anomaly,
            verdict, explanation, and heatmaps (list of per-frame heatmap arrays)
        """
        self._load()

        # --- Student model via ONNX ---
        rgb_np = rgb_tensor.numpy().astype(np.float32)
        dct_np = dct_tensor.numpy().astype(np.float32)

        feed = {}
        if len(self.onnx_input_names) >= 2:
            feed[self.onnx_input_names[0]] = rgb_np
            feed[self.onnx_input_names[1]] = dct_np
        else:
            feed[self.onnx_input_names[0]] = rgb_np

        onnx_output = self.onnx_session.run(self.onnx_output_names, feed)
        logit = onnx_output[0].item() if onnx_output[0].ndim == 0 else onnx_output[0].flatten()[0]
        student_prob = round(float(1 / (1 + np.exp(-logit))) * 100, 2)  # sigmoid

        # --- Autoencoder anomaly + heatmap ---
        with torch.no_grad():
            # rgb_tensor is (1, T, C, H, W), autoencoder expects (1, C, T, H, W)
            rgb_3d = rgb_tensor.permute(0, 2, 1, 3, 4).to(self.device)
            reconstructed = self.autoencoder(rgb_3d)

            # Per-pixel MSE for heatmaps: (1, C, T, H, W) → mean over C → (T, H, W)
            pixel_error = self.mse(reconstructed, rgb_3d)
            pixel_error_per_frame = pixel_error.squeeze(0).mean(dim=0)  # (T, H, W)

            # Scalar reconstruction error
            reconstruction_error = round(pixel_error.mean().item(), 4)
            is_anomaly = reconstruction_error > threshold

            # Generate per-frame heatmaps (normalised 0-255)
            heatmaps = []
            for t in range(pixel_error_per_frame.shape[0]):
                frame_err = pixel_error_per_frame[t].cpu().numpy()  # (H, W)
                # Normalise to 0-1
                err_min, err_max = frame_err.min(), frame_err.max()
                if err_max - err_min > 1e-8:
                    frame_norm = (frame_err - err_min) / (err_max - err_min)
                else:
                    frame_norm = np.zeros_like(frame_err)
                heatmaps.append((frame_norm * 255).astype(np.uint8))

        # --- Build explanation ---
        explanation_parts = []
        if student_prob > 50:
            explanation_parts.append(
                f"The two-stream classifier detected manipulation artifacts with {student_prob}% confidence."
            )
            if student_prob > 80:
                explanation_parts.append(
                    "Strong spatial-frequency inconsistencies suggest face-swap or reenactment techniques."
                )
            else:
                explanation_parts.append(
                    "Moderate frequency-domain anomalies detected in the facial region."
                )
        else:
            explanation_parts.append(
                f"No significant manipulation artifacts detected ({student_prob}% probability)."
            )

        if is_anomaly:
            explanation_parts.append(
                f"⚠ Zero-day anomaly: Reconstruction error ({reconstruction_error}) exceeds "
                f"threshold ({threshold}), indicating unnatural facial structure not seen in training."
            )
        else:
            explanation_parts.append(
                "Facial structure matches the authentic human manifold within normal bounds."
            )

        verdict = "MANIPULATED" if (student_prob > 50 or is_anomaly) else "AUTHENTIC"

        return {
            'deepfake_probability': student_prob,
            'reconstruction_error': reconstruction_error,
            'is_anomaly': is_anomaly,
            'verdict': verdict,
            'explanation': ' '.join(explanation_parts),
            'heatmaps': heatmaps,
        }


# Module-level accessor
def get_engine():
    """Get or create the singleton inference engine."""
    return DeepGuardEngine()
