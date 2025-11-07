
<div align="center">

# 🩻 Chest X-ray Disease Detection with Liquid Neural Networks (LNNs)
### **An Explainable AI System for Automated Pneumonia & Lung Opacity Detection**

![Python](https://img.shields.io/badge/Python-3.8+-3776AB?style=for-the-badge&logo=python&logoColor=white)
![Flask](https://img.shields.io/badge/Flask-Framework-black?style=for-the-badge&logo=flask)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white)
![License](https://img.shields.io/badge/License-MIT-yellow?style=for-the-badge)
![Status](https://img.shields.io/badge/Status-Development-blue?style=for-the-badge)

</div>

---

## 🎯 Overview
**Chest X-ray Disease Detection (LNN)** is an AI-based web app built using **Flask** and **PyTorch**, designed to detect **Pneumonia** and **Lung Opacity** from X-ray images.  
It provides both **Single Image Analysis** (with heatmap visualization) and **Bulk Folder Processing** for batch results.

This project emphasizes **Explainable AI (XAI)**, using **Occlusion Sensitivity Heatmaps** to highlight infected regions that influenced the model's predictions.

---

## 🧠 Core Features

| Feature | Description |
|:--------|:-------------|
| 🧬 Dual Model Inference | Combines two Liquid Neural Networks (LNNs) |
| 🔥 Occlusion Sensitivity Heatmap | Visual explanation of infected areas |
| 🧍 Single Mode | Real-time heatmap for one image |
| 📁 Bulk Mode | Folder upload (up to 1000 images) + CSV report |
| ⚙️ Risk Stratification | High / Medium / Low tiers based on confidence |
| 🧾 Auto Reports | Generates downloadable CSV files |
| 💾 Secure Handling | Temporary uploads + automatic cleanup |

---

## 🧩 System Architecture (Simplified)

```
User Uploads X-ray(s)
        ↓
Preprocessing & Normalization
        ↓
Model 1: Pneumonia Detection
Model 2: Lung Opacity Detection
        ↓
Confidence Calculation
        ↓
Ensemble Fusion (compare model confidences)
        ↓
Risk Stratification
        ↓
Single Mode → Heatmap Visualization
Bulk Mode → CSV Report Generation
```

---

## 🩻 Explainability (Occlusion Sensitivity)

Since LNNs are fully connected (not CNNs), **Grad-CAM** is not applicable.  
Instead, this system uses **Occlusion Sensitivity**, which:

1. Slides a masking window across the image.  
2. Measures drop in model confidence after each mask.  
3. Highlights regions that strongly affect predictions.  

| Color | Meaning |
|:------|:--------|
| 🔴 Red | High influence (infection region) |
| 🟡 Yellow | Moderate importance |
| 🔵 Blue | Neutral / normal region |

> ⚡ The resulting heatmap clearly shows infected zones on the X-ray image.

---

## ⚙️ Tech Stack

| Layer | Technology |
|:------|:------------|
| Frontend | HTML5, Bootstrap 4, CSS3, JS, Jinja2 |
| Backend | Flask (Python 3.9+) |
| AI Framework | PyTorch (Liquid Neural Networks) |
| Data Handling | PIL, NumPy, TorchVision, CSV |
| Visualization | Occlusion Sensitivity + Jet colormap |

---

## 🧱 Project Structure

```
Chest-Xray-LNN/
├── app.py                  # Flask backend (single + bulk logic)
├── models/
│   ├── liquid_model.pth    # Pneumonia model
│   └── best_model.pth      # Lung Opacity model
├── static/
│   ├── uploads/            # Uploaded images & CSVs
│   └── heatmaps/           # Generated overlays
├── templates/
│   ├── index.html
│   ├── upload_single.html
│   ├── upload_bulk.html
│   ├── result_single.html
│   ├── result_bulk.html
│   └── loading.html
└── requirements.txt
```

---

## 🚀 Setup Instructions

### 1️⃣ Clone Repository
```bash
git clone https://github.com/mXrahul01/Chest-X-Ray-disease-detection-using-Liquid-Neural-Network-LNNs.git
cd Chest-X-Ray-disease-detection-using-Liquid-Neural-Network-LNNs
```

### 2️⃣ Create Virtual Environment
```bash
python -m venv venv
venv\Scripts\activate        # Windows
source venv/bin/activate       # macOS/Linux
```

### 3️⃣ Install Dependencies
```bash
pip install -r requirements.txt
```

### 4️⃣ Add Pre-trained Models
```
models/
 ├── liquid_model.pth
 └── best_model.pth
```

### 5️⃣ Run Application
```bash
python app.py
```
Then open **http://127.0.0.1:5000/** in your browser.

---

## 📈 Outputs & Reports

| File | Description |
|:------|:-------------|
| `result_single.html` | Visual result page (heatmap + predictions) |
| `result_bulk.html` | Table summary + CSV downloads |
| `results.csv` | Combined output of all images |
| `results_disease.csv` | Only infected cases |
| `results_normal.csv` | Only normal images |
| `*_heatmap.jpg` | Infection region overlay |

---

## 🧮 Model Performance

| Model | Task | Accuracy | AUC | Type |
|:------|:------|:----------|:----|:----|
| LNN-1 | Pneumonia Detection | 94% | 0.98 | Liquid Neural Net |
| LNN-2 | Lung Opacity | 91% | 0.96 | Liquid Neural Net |
| Ensemble | Combined Output | **92.5%** | **0.97** | Weighted Fusion |

---

## 🔬 Future Enhancements

- [ ] Integrate Grad-CAM/Integrated Gradients for CNNs  
- [ ] DICOM & 3D CT scan support  
- [ ] FastAPI REST API for remote inference  
- [ ] GPU acceleration with PyTorch Lightning  
- [ ] Auto PDF report generation

---

## 📜 License
Released under the **MIT License** — free for educational and research use.

---

<div align="center">

### 💙 Developed by [**Rahul (mXrahul01)**](https://github.com/mXrahul01)
**Explainable AI for Medical Imaging — bridging accuracy and trust.** 🧠

</div>
