import sys
import os
import librosa
import numpy as np
import joblib
import warnings
import soundfile as sf
from scipy.fft import dct
warnings.filterwarnings('ignore')

def extract_features(audio, sr):
    """Extracts 225 features — identical to training pipeline."""
    try:
        max_peak = np.max(np.abs(audio))
        if max_peak > 0:
            audio = audio / max_peak

        mfccs        = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=40)
        mfccs_mean   = np.mean(mfccs, axis=1)
        mfccs_std    = np.std(mfccs, axis=1)
        delta_mfccs  = librosa.feature.delta(mfccs)
        delta_mean   = np.mean(delta_mfccs, axis=1)
        delta2_mfccs = librosa.feature.delta(mfccs, order=2)
        delta2_mean  = np.mean(delta2_mfccs, axis=1)

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

        chroma        = librosa.feature.chroma_stft(y=audio, sr=sr)
        chroma_mean   = np.mean(chroma, axis=1)
        contrast      = librosa.feature.spectral_contrast(y=audio, sr=sr)
        contrast_mean = np.mean(contrast, axis=1)

        zcr_mean      = np.mean(librosa.feature.zero_crossing_rate(audio))
        zcr_std       = np.std(librosa.feature.zero_crossing_rate(audio))
        rms_mean      = np.mean(librosa.feature.rms(y=audio))
        rms_std       = np.std(librosa.feature.rms(y=audio))
        rolloff_mean  = np.mean(
            librosa.feature.spectral_rolloff(y=audio, sr=sr))
        centroid_mean = np.mean(
            librosa.feature.spectral_centroid(y=audio, sr=sr))

        return np.hstack([
            mfccs_mean, mfccs_std, delta_mean, delta2_mean,
            lfcc_mean, chroma_mean, contrast_mean,
            [zcr_mean, zcr_std, rms_mean,
             rms_std, rolloff_mean, centroid_mean]
        ]).reshape(1, -1)

    except Exception as e:
        print(f"❌ Feature extraction error: {e}")
        return None


def predict(audio_path, model_path="models/deepfake_audio_model.pkl"):
    # Check files exist
    if not os.path.exists(audio_path):
        print(f"❌ Audio file not found: {audio_path}")
        return
    if not os.path.exists(model_path):
        print(f"❌ Model not found: {model_path}")
        return

    print(f"\n🎵 Analysing: {audio_path}")
    print("-" * 45)

    # Load model
    model = joblib.load(model_path)

    # Load audio
    data, native_sr = sf.read(audio_path)
    if len(data.shape) > 1:
        data = np.mean(data, axis=1)
    if native_sr != 16000:
        audio = librosa.resample(
            y=data.astype(np.float32),
            orig_sr=native_sr,
            target_sr=16000)
    else:
        audio = data.astype(np.float32)

    # Extract features
    features = extract_features(audio, 16000)
    if features is None:
        print("❌ Could not extract features.")
        return

    # Predict
    prediction = model.predict(features)[0]
    probs      = model.predict_proba(features)[0]
    confidence = probs[prediction] * 100

    # Results
    print(f" Result:     "
          f"{' AI-Generated Deepfake' if prediction == 1 else ' Genuine Human Voice'}")
    print(f" Confidence: {confidence:.2f}%")
    print(f" Probability Breakdown:")
    print(f"   Human (Real):       {probs[0]*100:.2f}%")
    print(f"   AI Generated (Fake): {probs[1]*100:.2f}%")
    print("-" * 45)


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python predict.py <path_to_audio.wav>")
        print("Example: python predict.py data/test/real/file1.wav")
    else:
        predict(sys.argv[1])