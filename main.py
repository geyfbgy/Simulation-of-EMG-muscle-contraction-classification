import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import butter, filtfilt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, accuracy_score, ConfusionMatrixDisplay

# ===============================================================
# 1. EMG SIMULATION
# ===============================================================

def generate_emg_signal(duration=2.0, fs=1000):
    """
    Generate a synthetic EMG signal with simulated muscle contractions.
    """
    t = np.linspace(0, duration, int(fs * duration))
    emg = np.random.normal(0, 0.05, len(t))  # baseline noise

    # Random bursts (contractions)
    for _ in range(np.random.randint(3, 6)):
        max_start = len(t) - int(0.3 * fs)  # ensure we have room for max burst
        start = np.random.randint(0, max_start)
        burst_length = np.random.randint(int(0.1 * fs), int(0.3 * fs))
        end = start + burst_length
        if end > len(t):
            end = len(t)
            burst_length = end - start
        burst = np.random.normal(0, np.random.uniform(0.4, 1.2), burst_length)
        emg[start:end] += burst

    return t, emg, fs

# ===============================================================
# 2. SIGNAL PROCESSING (Filtering + Rectification)
# ===============================================================

def bandpass_filter(signal, fs, lowcut=20, highcut=450, order=4):
    nyquist = 0.5 * fs
    low = lowcut / nyquist
    high = highcut / nyquist
    b, a = butter(order, [low, high], btype='band')
    return filtfilt(b, a, signal)

def preprocess_emg(emg, fs):
    filtered = bandpass_filter(emg, fs)
    rectified = np.abs(filtered)
    return rectified


# ===============================================================
# 3. FEATURE EXTRACTION
# ===============================================================

def extract_features(emg, fs, window_size=0.2):
    step = int(window_size * fs)
    n_windows = len(emg) // step
    features = []

    for i in range(n_windows):
        segment = emg[i * step : (i + 1) * step]
        rms = np.sqrt(np.mean(segment ** 2))
        mav = np.mean(np.abs(segment))
        wl = np.sum(np.abs(np.diff(segment)))
        zc = np.sum(segment[:-1] * segment[1:] < 0)
        features.append([rms, mav, wl, zc])

    return np.array(features)

def label_segments(emg, fs, window_size=0.2):
    """
    Automatically label each segment as weak / medium / strong contraction.
    """
    step = int(window_size * fs)
    n_windows = len(emg) // step
    labels = []

    for i in range(n_windows):
        segment = emg[i * step : (i + 1) * step]
        amplitude = np.mean(np.abs(segment))
        if amplitude < 0.1:
            labels.append("weak")
        elif amplitude < 0.3:
            labels.append("medium")
        else:
            labels.append("strong")

    return np.array(labels)


# ===============================================================
# 5. DATASET GENERATION (multiple trials)
# ===============================================================

def create_dataset(n_samples=50):
    X, y = [], []
    for _ in range(n_samples):
        t, emg, fs = generate_emg_signal()
        processed = preprocess_emg(emg, fs)
        feats = extract_features(processed, fs)
        labels = label_segments(processed, fs)
        X.append(feats)
        y.append(labels)

    X = np.vstack(X)
    y = np.hstack(y)
    return X, y


# ===============================================================
# 6. MACHINE LEARNING MODEL
# ===============================================================

def train_classifier(X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    print(f"\nModel Accuracy: {acc * 100:.2f}%")

    cm = confusion_matrix(y_test, y_pred, labels=["weak", "medium", "strong"])
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["weak", "medium", "strong"])
    disp.plot(cmap="Purples")
    plt.title("Confusion Matrix - EMG Classification")
    plt.show()

    return model


# ===============================================================
# 7. VISUALIZATION OF ONE SAMPLE
# ===============================================================

def visualize_emg_sample():
    t, emg, fs = generate_emg_signal()
    processed = preprocess_emg(emg, fs)

    plt.figure(figsize=(10, 5))
    plt.subplot(2, 1, 1)
    plt.plot(t, emg, color='purple')
    plt.title("Raw Simulated EMG Signal")
    plt.xlabel("Time (s)")
    plt.ylabel("Amplitude (V)")

    plt.subplot(2, 1, 2)
    plt.plot(t, processed, color='red')
    plt.title("Processed EMG (Rectified & Filtered)")
    plt.xlabel("Time (s)")
    plt.ylabel("Amplitude (V)")
    plt.tight_layout()
    plt.show()


# ===============================================================
# 8. MAIN EXECUTION
# ===============================================================

def main():
    print("Generating dataset...")
    X, y = create_dataset(n_samples=80)
    print("Dataset shape:", X.shape, "Labels:", len(y))

    print("Training classifier...")
    model = train_classifier(X, y)

    print("Visualizing EMG sample...")
    visualize_emg_sample()


if __name__ == "__main__":
    main()