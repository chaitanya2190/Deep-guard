import os
import cv2
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import streamlit as st
from torchvision import transforms
from ultralytics import YOLO
from scipy.fftpack import dct

# --- Configuration ---
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
STUDENT_WEIGHTS = "./output/student_weights_v2/student_x3d_v2_epoch_10.pth"
AUTOENCODER_WEIGHTS = "./output/autoencoder_weights_v2/autoencoder_v2_epoch_5.pth"
TEMP_VIDEO_PATH = "temp_uploaded_video.mp4"

# --- Architecture Definitions ---
class FrequencyBranch(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv3d(1, 16, 3, padding=1), nn.ReLU(), nn.MaxPool3d((1, 2, 2)),
            nn.Conv3d(16, 32, 3, padding=1), nn.ReLU(), nn.MaxPool3d((1, 2, 2))
        )
        self.fc = nn.Sequential(nn.Linear(32 * 8 * 56 * 56, 128), nn.ReLU(), nn.Dropout(0.5))
    def forward(self, x):
        return self.fc(self.conv(x.permute(0, 2, 1, 3, 4)).flatten(1))

class TwoStreamStudent(nn.Module):
    def __init__(self):
        super().__init__()
        self.spatial = torch.hub.load('facebookresearch/pytorchvideo', 'x3d_s', pretrained=False)
        self.spatial.blocks[5].pool.pool = nn.AvgPool3d(kernel_size=(8, 5, 5), stride=(1, 1, 1), padding=(0, 0, 0))
        self.spatial.blocks[5].proj = nn.Identity()
        self.spatial.blocks[5].activation = nn.Identity()
        self.freq = FrequencyBranch()
        self.classifier = nn.Sequential(
            nn.Linear(2048 + 128, 512), nn.BatchNorm1d(512), nn.ReLU(), 
            nn.Dropout(0.5), nn.Linear(512, 1)
        )
    def forward(self, rgb, dct):
        s_feat = self.spatial(rgb.permute(0, 2, 1, 3, 4)).flatten(1) 
        f_feat = self.freq(dct)
        return self.classifier(torch.cat((s_feat, f_feat), dim=1)).squeeze(1)

class SpatialTemporalAutoencoder(nn.Module):
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

class DeepGuardEnsemble:
    def __init__(self, student_path, autoencoder_path, device):
        self.device = device
        self.student = TwoStreamStudent().to(self.device)
        self.autoencoder = SpatialTemporalAutoencoder().to(self.device)
        
        student_checkpoint = torch.load(student_path, map_location=self.device, weights_only=False)
        if 'model_state_dict' in student_checkpoint:
            self.student.load_state_dict(student_checkpoint['model_state_dict'])
        else:
            self.student.load_state_dict(student_checkpoint)
            
        self.autoencoder.load_state_dict(torch.load(autoencoder_path, map_location=self.device, weights_only=True))
        self.student.eval()
        self.autoencoder.eval()
        self.mse = nn.MSELoss(reduction='none')
        
    def predict(self, rgb_tensor, dct_tensor, threshold):
        with torch.no_grad(), torch.amp.autocast('cuda'):
            student_prob = torch.sigmoid(self.student(rgb_tensor, dct_tensor)).item()
            rgb_3d = rgb_tensor.permute(0, 2, 1, 3, 4)
            reconstruction_error = self.mse(self.autoencoder(rgb_3d), rgb_3d).mean().item()
            is_anomaly = reconstruction_error > threshold
                
        return round(student_prob * 100, 2), round(reconstruction_error, 4), is_anomaly

# --- Preprocessing ---
@st.cache_resource
def load_face_detector():
    return YOLO('yolov8n-face.pt')

face_detector = load_face_detector()
transform = transforms.Normalize([0.45, 0.45, 0.45], [0.225, 0.225, 0.225])

def extract_dct_features(face_rgb):
    ycrcb = cv2.cvtColor(face_rgb, cv2.COLOR_RGB2YCrCb)
    y_channel = ycrcb[:, :, 0]
    dct_y = dct(dct(y_channel.T, norm='ortho').T, norm='ortho')
    high_freq = np.abs(dct_y[112:, 112:])
    tensor = torch.from_numpy(cv2.resize(high_freq, (224, 224))).unsqueeze(0).float()
    return tensor / (torch.max(tensor) + 1e-8)

def process_video_for_inference(video_path):
    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total_frames < 8: return None, None, None
    
    indices = np.linspace(0, total_frames - 1, 8, dtype=int)
    rgb_tensors, dct_tensors, display_crops = [], [], []
    last_box = None

    for idx in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        if not ret: continue

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = face_detector(frame_rgb, verbose=False, device=DEVICE)

        if len(results[0].boxes) > 0:
            last_box = results[0].boxes[0].xyxy.cpu().numpy()[0]

        if last_box is not None:
            x1, y1, x2, y2 = map(int, last_box)
            pad_w, pad_h = int((x2-x1)*0.20), int((y2-y1)*0.20)
            y1_pad, y2_pad = max(0, y1-pad_h), min(frame_rgb.shape[0], y2+pad_h)
            x1_pad, x2_pad = max(0, x1-pad_w), min(frame_rgb.shape[1], x2+pad_w)

            face_crop = frame_rgb[y1_pad:y2_pad, x1_pad:x2_pad]
            if face_crop.size == 0: continue

            face_resized = cv2.resize(face_crop, (224, 224))
            
            # Save raw RGB array for UI display
            display_crops.append(face_resized)

            tensor_rgb = transform(torch.from_numpy(face_resized).permute(2, 0, 1).float() / 255.0)
            rgb_tensors.append(tensor_rgb)
            dct_tensors.append(extract_dct_features(face_resized))

    cap.release()
    if len(rgb_tensors) == 8:
        return torch.stack(rgb_tensors).unsqueeze(0), torch.stack(dct_tensors).unsqueeze(0), display_crops
    return None, None, None

# --- Streamlit UI ---
st.set_page_config(page_title="DeepGuard: Deepfake Detection", layout="wide")
st.title("🛡️ DeepGuard: Two-Stream Deepfake Detection")

@st.cache_resource
def load_ensemble():
    return DeepGuardEnsemble(STUDENT_WEIGHTS, AUTOENCODER_WEIGHTS, DEVICE)

try:
    ensemble = load_ensemble()
except Exception as e:
    st.error(f"Failed to load models. Ensure weights exist at specified paths. Error: {e}")
    st.stop()

st.sidebar.header("Configuration")
anomaly_threshold = st.sidebar.slider("Zero-Day Anomaly Threshold (MSE)", min_value=0.01, max_value=0.50, value=0.15, step=0.01)

uploaded_file = st.file_uploader("Upload a Video (.mp4)", type=["mp4"])

if uploaded_file is not None:
    with open(TEMP_VIDEO_PATH, "wb") as f:
        f.write(uploaded_file.read())
        
    st.video(TEMP_VIDEO_PATH)
    
    if st.button("Analyze Video"):
        with st.spinner("Extracting Spatial, Temporal, and Frequency Features..."):
            rgb_batch, dct_batch, display_crops = process_video_for_inference(TEMP_VIDEO_PATH)
            
        if rgb_batch is not None:
            
            # --- RENDER EXTRACTED FRAMES ---
            st.subheader("Extracted Temporal Sequence (8 Frames)")
            st.caption("These spatial crops are passed directly to the TimeSformer and DCT analyzer.")
            cols = st.columns(8)
            for i, crop in enumerate(display_crops):
                cols[i].image(crop, use_container_width=True, caption=f"Frame {i+1}")
            st.markdown("---")
            
            with st.spinner("Running Inference through DeepGuard Ensemble..."):
                rgb_batch, dct_batch = rgb_batch.to(DEVICE), dct_batch.to(DEVICE)
                prob, mse, is_anomaly = ensemble.predict(rgb_batch, dct_batch, anomaly_threshold)
                
            st.subheader("Analysis Results")
            col1, col2 = st.columns(2)
            
            with col1:
                st.metric(label="Deepfake Probability", value=f"{prob}%")
                if prob > 50:
                    st.error("Verdict: MANIPULATED (Known Artifacts Detected)")
                else:
                    st.success("Verdict: AUTHENTIC (No Known Artifacts)")
                    
            with col2:
                st.metric(label="Reconstruction Error (MSE)", value=mse)
                if is_anomaly:
                    st.warning("⚠️ ZERO-DAY ANOMALY DETECTED: Unnatural facial structure.")
                else:
                    st.info("Structure matches authentic human manifold.")
        else:
            st.error("Failed to extract 8 consecutive face frames. Ensure the video contains a clearly visible face.")
            
    os.remove(TEMP_VIDEO_PATH)