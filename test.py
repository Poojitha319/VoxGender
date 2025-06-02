import numpy as np
import pandas as pd
import joblib
import librosa
from sklearn.metrics import accuracy_score, classification_report

# Load the trained model
model_path = "models\svm_model.pkl" 
try:
    model = joblib.load(model_path)
    print("Model loaded successfully.")
except Exception as e:
    raise FileNotFoundError(f"Failed to load model. Make sure '{model_path}' exists.\n{e}")

# Feature extraction to match original dataset features
def extract_features(file_path):
    signal, sr = librosa.load(file_path)

    centroid = librosa.feature.spectral_centroid(y=signal, sr=sr)[0]
    bandwidth = librosa.feature.spectral_bandwidth(y=signal, sr=sr)[0]
    flatness = librosa.feature.spectral_flatness(y=signal)[0]
    rolloff = librosa.feature.spectral_rolloff(y=signal, sr=sr)[0]
    zcr = librosa.feature.zero_crossing_rate(signal)[0]
    mfccs = librosa.feature.mfcc(y=signal, sr=sr, n_mfcc=20)

    features = {
        'meanfreq': np.mean(centroid),
        'sd': np.std(signal),
        'median': np.median(signal),
        'Q25': np.percentile(signal, 25),
        'Q75': np.percentile(signal, 75),
        'IQR': np.subtract(*np.percentile(signal, [75, 25])),
        'skew': pd.Series(signal).skew(),
        'kurt': pd.Series(signal).kurt(),
        'sp.ent': np.mean(-signal * np.log2(np.abs(signal) + 1e-10)),
        'sfm': np.mean(flatness),
        'mode': float(pd.Series(signal).mode().iloc[0]),
        'centroid': np.mean(centroid),
        'meanfun': np.mean(np.abs(signal)),
        'minfun': np.min(np.abs(signal)),
        'maxfun': np.max(np.abs(signal)),
        'meandom': np.mean(rolloff),
        'mindom': np.min(rolloff),
        'maxdom': np.max(rolloff),
        'dfrange': np.max(rolloff) - np.min(rolloff),
        'modindx': np.mean(np.abs(signal)) / (np.max(np.abs(signal)) + 1e-10),
    }
    return pd.DataFrame([features])

# Example usage with a sample WAV file
voice_file = "sample_voice.wav" 
try:
    X_input = extract_features(voice_file)
    print("Feature extraction successful.")
except Exception as e:
    raise ValueError(f"Failed to extract features from the audio file.\n{e}")

# Predict
prediction = model.predict(X_input)
print("Predicted Gender:", prediction[0])
