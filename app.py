import streamlit as st
import os
import librosa
import numpy as np
import joblib
import tempfile
import warnings
import soundfile as sf
from scipy.fft import dct
warnings.filterwarnings('ignore')

st.set_page_config(
    page_title="Deepfake Audio Detector",
    page_icon="🎵",
    layout="centered"
)

st.title("🎵 Deepfake Audio Detection Engine")
st.write("Upload a .wav audio file to detect whether it is "
         "Genuine (Human) or AI-Generated (Deepfake).")

# ── Load model ──
MODEL_PATH = os.path.join("models", "deepfake_audio_model.pkl")

@st.cache_resource
def load_model():
    if not os.path.exists(MODEL_PATH):
        st.error("❌ Model not found in 'models/' folder!")
        return None
    return joblib.load(MODEL_PATH)

model = load_model()

# ── Feature extraction — MUST match training exactly ──
def extract_features_from_array(audio, sr):
    """Extracts 225 features — identical to training pipeline."""
    try:
        max_peak = np.max(np.abs(audio))
        if max_peak > 0:
            audio = audio / max_peak

        # MFCCs mean + std (80 values)
        mfccs        = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=40)
        mfccs_mean   = np.mean(mfccs, axis=1)
        mfccs_std    = np.std(mfccs, axis=1)

        # Delta MFCCs (40 values)
        delta_mfccs  = librosa.feature.delta(mfccs)
        delta_mean   = np.mean(delta_mfccs, axis=1)

        # Delta-Delta MFCCs (40 values)
        delta2_mfccs = librosa.feature.delta(mfccs, order=2)
        delta2_mean  = np.mean(delta2_mfccs, axis=1)

        # LFCC (40 values)
        n_fft        = 512
        D            = np.abs(librosa.stft(audio, n_fft=n_fft))
        n_bins       = D.shape[0]
        n_filters    = 40
        freq_points  = np.linspace(0, n_bins-1, n_filters+2, dtype=int)
        filters      = np.zeros((n_filters, n_bins))
        for i in range(n_filters):
            start = freq_points[i]
            mid   = freq_points[i+1]
            end   = freq_points[i+2]
            if mid-start > 0:
                filters[i, start:mid] = np.linspace(0, 1, mid-start)
            if end-mid > 0:
                filters[i, mid:end]   = np.linspace(1, 0, end-mid)
        filter_energies = np.dot(filters, D)
        filter_energies = np.where(filter_energies == 0,
                                   np.finfo(float).eps,
                                   filter_energies)
        lfcc_matrix  = dct(np.log(filter_energies), axis=0, norm='ortho')[:40]
        lfcc_mean    = np.mean(lfcc_matrix, axis=1)

        # Chroma (12 values)
        chroma       = librosa.feature.chroma_stft(y=audio, sr=sr)
        chroma_mean  = np.mean(chroma, axis=1)

        # Spectral Contrast (7 values)
        contrast      = librosa.feature.spectral_contrast(y=audio, sr=sr)
        contrast_mean = np.mean(contrast, axis=1)

        # Scalar features (6 values)
        zcr_mean      = np.mean(librosa.feature.zero_crossing_rate(audio))
        zcr_std       = np.std(librosa.feature.zero_crossing_rate(audio))
        rms_mean      = np.mean(librosa.feature.rms(y=audio))
        rms_std       = np.std(librosa.feature.rms(y=audio))
        rolloff_mean  = np.mean(librosa.feature.spectral_rolloff(y=audio, sr=sr))
        centroid_mean = np.mean(librosa.feature.spectral_centroid(y=audio, sr=sr))

        combined = np.hstack([
            mfccs_mean, mfccs_std, delta_mean, delta2_mean,
            lfcc_mean, chroma_mean, contrast_mean,
            [zcr_mean, zcr_std, rms_mean,
             rms_std, rolloff_mean, centroid_mean]
        ])
        return combined.reshape(1, -1)

    except Exception as e:
        st.error(f"Feature extraction error: {e}")
        return None


# ── UI ──
uploaded_file = st.file_uploader(
    "Choose an audio file...", type=["wav"])

if uploaded_file is not None:
    st.audio(uploaded_file, format="audio/wav")

    if st.button(" Run Analysis"):
        if model is None:
            st.error("Model not loaded.")
        else:
            with st.spinner("Analysing audio..."):
                # Save uploaded file temporarily
                suffix = os.path.splitext(uploaded_file.name)[1]
                with tempfile.NamedTemporaryFile(
                        delete=False, suffix=suffix) as tmp:
                    tmp.write(uploaded_file.getvalue())
                    tmp_path = tmp.name

                try:
                    # Load with soundfile then resample
                    data, native_sr = sf.read(tmp_path)
                    if len(data.shape) > 1:
                        data = np.mean(data, axis=1)
                    if native_sr != 16000:
                        audio = librosa.resample(
                            y=data.astype(np.float32),
                            orig_sr=native_sr,
                            target_sr=16000)
                    else:
                        audio = data.astype(np.float32)

                    features = extract_features_from_array(audio, 16000)

                finally:
                    if os.path.exists(tmp_path):
                        os.remove(tmp_path)

                if features is None:
                    st.error("Could not extract features.")
                else:
                    prediction  = model.predict(features)[0]
                    probs       = model.predict_proba(features)[0]
                    confidence  = probs[prediction] * 100

                    st.write("---")
                    st.subheader("📊 Analysis Results")

                    if prediction == 1:
                        st.error(
                            "🚨 **Classification: AI-Generated Deepfake**")
                    else:
                        st.success(
                            "✅ **Classification: Genuine (Human Voice)**")

                    st.metric(
                        label="Confidence Score",
                        value=f"{confidence:.2f}%")

                    # Show probability breakdown
                    st.write("**Probability Breakdown:**")
                    col1, col2 = st.columns(2)
                    with col1:
                        st.metric("Human (Real)",
                                  f"{probs[0]*100:.2f}%")
                    with col2:
                        st.metric("AI Generated (Fake)",
                                  f"{probs[1]*100:.2f}%")