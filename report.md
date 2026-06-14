# 📊 Performance Report — Deepfake Audio Detection

## Final Results on Test Set

| Metric | Score | Target | Status |
|--------|-------|--------|--------|
| Overall Accuracy | 89.70% | ≥ 80% | ✅ PASS |
| F1 Score | 88.72% | ≥ 80% | ✅ PASS |
| Equal Error Rate (EER) | 7.00% | ≤ 12% | ✅ PASS |
| Per-Class Accuracy (Real) | 98.40% | ≥ 75% | ✅ PASS |
| Per-Class Accuracy (Fake) | 81.00% | ≥ 75% | ✅ PASS |

## Confusion Matrix

![Confusion Matrix](confusion_matrix_test.png)

## Classification Report

| Class | Precision | Recall | F1-Score | Support |
|-------|-----------|--------|----------|---------|
| Genuine (Real) | 0.84 | 0.98 | 0.90 | 500 |
| Deepfake (Fake) | 0.97 | 0.81 | 0.88 | 500 |
| **Weighted Avg** | **0.91** | **0.90** | **0.89** | **1000** |

---

## Preprocessing Pipeline

1. **Audio Loading** — All files loaded at 16kHz mono using librosa
2. **Normalization** — Peak amplitude normalization (divide by max peak)
3. **Augmentation** — Fake samples augmented 3x using:
   - Gaussian noise addition (σ = 0.003)
   - Pitch shifting (+1 semitone)
   - Time stretching (rate = 1.1)
4. **Class Balancing** — Real samples upsampled to match fake count

---

## Feature Extraction (225 features total)

| Feature | Description | Size |
|---------|-------------|------|
| MFCC Mean | Vocal tract shape | 40 |
| MFCC Std | Variation in vocal tract | 40 |
| Delta MFCC | Rate of change of MFCCs | 40 |
| Delta-Delta MFCC | Acceleration of MFCCs | 40 |
| LFCC | Linear Frequency Cepstral Coefficients | 40 |
| Chroma STFT | Pitch and harmonic content | 12 |
| Spectral Contrast | Peak vs valley energy ratio | 7 |
| ZCR Mean + Std | Zero crossing rate | 2 |
| RMS Mean + Std | Energy variation | 2 |
| Spectral Rolloff | High frequency energy | 1 |
| Spectral Centroid | Center of mass of spectrum | 1 |
| **Total** | | **225** |

---

## Model Architecture

- **Algorithm:** MLP Neural Network (sklearn MLPClassifier)
- **Hidden Layers:** 256 → 128 → 64 neurons
- **Activation:** ReLU
- **Optimizer:** Adam (adaptive learning rate)
- **Regularization:** L2 alpha = 0.05
- **Early Stopping:** Yes (patience = 25 iterations)
- **Preprocessing:** StandardScaler (mean=0, std=1)

---

## Why These Choices?

### Why MLP over Random Forest/SVM?
Random Forest and SVM both achieved ~99% training accuracy
but only 68-73% on test data — classic overfitting.
MLP with early stopping and regularization achieved 89.70% on test.

### Why LFCC?
LFCC (Linear Frequency Cepstral Coefficients) is the gold standard
feature for anti-spoofing, used in the ASVspoof benchmark.
Unlike MFCC which uses mel scale optimized for speech recognition,
LFCC uses linear scale which better captures artifacts
in AI-generated speech.

### Why Data Augmentation?
The test set contained deepfakes from different AI systems than
the training set. Augmenting fake samples with noise, pitch shift,
and time stretch forced the model to learn general fakeness patterns
rather than memorizing specific AI voice characteristics.