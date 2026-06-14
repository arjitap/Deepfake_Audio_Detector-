import os
import numpy as np
import soundfile as sf

def create_mock_dataset():
    # This automatically sets up a data folder and a train subfolder
    train_dir = os.path.join("data", "train")
    os.makedirs(train_dir, exist_ok=True)
    
    print(f"📁 Creating sample audio data inside: {train_dir}")
    
    sr = 16000  # 16kHz audio sample rate (standard for speech datasets)
    duration = 2.0  # 2 seconds long
    t = np.linspace(0, duration, int(sr * duration), endpoint=False)
    
    # Generate 10 mock "Genuine" sound waves
    for i in range(10):
        freq = 150 + (i * 5) 
        audio_signal = np.sin(2 * np.pi * freq * t)
        file_path = os.path.join(train_dir, f"real_sample_{i}.wav")
        sf.write(file_path, audio_signal, sr)
        
    # Generate 10 mock "Deepfake" sound waves with simulated high-frequency artifacts
    for i in range(10):
        freq = 150 + (i * 5)
        clean_signal = np.sin(2 * np.pi * freq * t)
        synthetic_noise = 0.1 * np.sin(2 * np.pi * 4000 * t) 
        audio_signal = clean_signal + synthetic_noise
        file_path = os.path.join(train_dir, f"fake_sample_{i}.wav")
        sf.write(file_path, audio_signal, sr)

    print("✅ Done! 10 real files and 10 fake files are sitting in your data folder.")

if __name__ == "__main__":
    create_mock_dataset()