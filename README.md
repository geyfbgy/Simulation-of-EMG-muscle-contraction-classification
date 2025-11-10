# 🧠 Simulation of EMG Muscle Contraction Classification

This project demonstrates the **simulation, feature extraction, and classification of Electromyography (EMG) signals** to recognize different muscle contraction levels.  
It showcases how **machine learning techniques** can be applied to **biomedical signal processing** for gesture or contraction classification.

---

## ⚙️ Overview

Electromyography (EMG) measures the electrical activity of muscles during contraction and relaxation.  
In this project, we **simulate EMG signals**, extract **time-domain and frequency-domain features**, and use **machine learning models** to classify muscle contraction states.

### 🔍 Workflow:
1. **Signal Simulation:** Generate synthetic EMG signals representing different contraction levels.  
2. **Preprocessing:** Apply filtering, normalization, and segmentation.  
3. **Feature Extraction:** Compute statistical and morphological features such as:
   - Mean Absolute Value (MAV)
   - Root Mean Square (RMS)
   - Zero Crossing (ZC)
   - Slope Sign Changes (SSC)
   - Waveform Length (WL)
4. **Classification:** Train ML models (SVM, KNN, Random Forest, or Neural Networks).  
5. **Evaluation:** Measure accuracy, F1-score, and visualize confusion matrices.

---

## 🧩 Key Features
- EMG signal generation and visualization  
- Noise filtering and baseline correction  
- Feature extraction (time/frequency domain)  
- Classification of muscle contraction intensity  
- Performance metrics and comparative analysis  

---

## 🧠 Technologies Used
| Category | Tools/Libraries |
|-----------|----------------|
| Programming | Python |
| Signal Processing | NumPy, SciPy |
| Machine Learning | scikit-learn, TensorFlow/PyTorch (optional) |
| Visualization | Matplotlib, Seaborn |
| Environment | VSCode |

---

## 🧪 Installation

```bash
# Clone the repository
git clone https://github.com/<your-username>/emg-muscle-contraction-simulation.git

# Navigate to the project folder
cd emg-muscle-contraction-simulation

# Install dependencies
pip install -r requirements.txt
