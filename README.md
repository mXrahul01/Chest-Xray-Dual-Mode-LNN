
<div align="center">

<img src="https://user-images.githubusercontent.com/67873374/194789912-01c2d3e8-144c-4c1c-b32c-35a6e6edb7e8.gif" width="80" height="80">

# 🩻 **Chest X-ray Disease Detection using Liquid Neural Networks (LNNs)**
### 🌐 *An Explainable AI System for Pneumonia & Lung Opacity Detection*

![Python](https://img.shields.io/badge/Python-3.8+-3776AB?style=for-the-badge&logo=python&logoColor=white)
![Flask](https://img.shields.io/badge/Flask-Framework-black?style=for-the-badge&logo=flask)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white)
![License](https://img.shields.io/badge/License-MIT-yellow?style=for-the-badge)
![Status](https://img.shields.io/badge/Status-Active-brightgreen?style=for-the-badge)

</div>

---

## 🚀 Overview

✨ **Chest X-ray Dual-Mode Disease Detection (LNN)** is a cutting-edge **Explainable AI (XAI)** web application built with **Flask + PyTorch**.  
It employs **Liquid Neural Networks (LNNs)** — a biologically inspired model that dynamically adapts over time — to detect **Pneumonia** and **Lung Opacity** with interpretability.

🌟 Features dual operational modes:
- 🧍‍♂️ **Single Image Mode** — Generates detailed heatmaps visualizing infection zones.  
- 📁 **Bulk Image Mode** — Analyzes hundreds of X-rays and provides downloadable CSV reports.  

---

## ⚡ Sneak Peek — Interface Highlights

| Page | Preview |
|------|----------|
| 🏠 **Home Page (Part 1)** | ![Index Page 1](https://github.com/mXrahul01/Chest-Xray-Dual-Mode-LNN/blob/main/Interface/1Index%20Page%20Part1.png) |
| 🏠 **Home Page (Part 2)** | ![Index Page 2](https://github.com/mXrahul01/Chest-Xray-Dual-Mode-LNN/blob/main/Interface/1Index%20Page%20Part2.png) |
| 📁 **Bulk Upload Page** | ![Bulk Upload](https://github.com/mXrahul01/Chest-Xray-Dual-Mode-LNN/blob/main/Interface/2Bulk%20Upload%20Page.png) |
| 🔄 **Bulk Loading Animation** | ![Loading Page](https://github.com/mXrahul01/Chest-Xray-Dual-Mode-LNN/blob/main/Interface/2.5Bulk%20Loading%20page.png) |
| 🧍‍♀️ **Single Image Upload** | ![Single Upload](https://github.com/mXrahul01/Chest-Xray-Dual-Mode-LNN/blob/main/Interface/2Upload%20Single%20Image.png) |
| 🧠 **Single Result (Part 1)** | ![Result 1](https://github.com/mXrahul01/Chest-Xray-Dual-Mode-LNN/blob/main/Interface/3Single%20Image%20result%20Part1%20.png) |
| 🧠 **Single Result (Part 2)** | ![Result 2](https://github.com/mXrahul01/Chest-Xray-Dual-Mode-LNN/blob/main/Interface/3Single%20Image%20result%20Part2.png) |
| 📊 **Bulk Result Table** | ![Bulk Result](https://github.com/mXrahul01/Chest-Xray-Dual-Mode-LNN/blob/main/Interface/4Bulk%20result%20Part1.png) |
| 📄 **CSV Output** | ![CSV File](https://github.com/mXrahul01/Chest-Xray-Dual-Mode-LNN/blob/main/Interface/CSV%20File.png) |
| 💻 **Flask App Backend** | ![Backend Code](https://github.com/mXrahul01/Chest-Xray-Dual-Mode-LNN/blob/main/Interface/Main%20Flask%20app%20code%20page.png) |

---

## 💡 Features

| 🔧 Feature | ✨ Description |
|------------|----------------|
| 🧬 **Dual Model Analysis** | Pneumonia & Lung Opacity detection via two Liquid Neural Nets |
| 🔥 **Occlusion Sensitivity Heatmap** | Explainable visualization of infection zones |
| 🧍 **Single Image Mode** | Interactive diagnosis and attention heatmap |
| 📁 **Bulk Image Mode** | Batch analysis with CSV reports |
| ⚙️ **Risk Stratification** | Severity-based classification (High / Medium / Low) |
| 🧾 **Auto Reports** | Instant CSV generation |
| 🧠 **XAI Focused** | Full transparency with interpretable results |

---

## 🧩 Workflow Diagram

```
🩻 User Uploads X-ray(s)
      ↓
⚙️ Preprocessing (Resize + Normalize)
      ↓
🧠 Model 1 — Pneumonia Detector
🧠 Model 2 — Lung Opacity Detector
      ↓
📊 Confidence Comparison + Ensemble Fusion
      ↓
⚖️ Risk Stratification
      ↓
🧍 Single → Heatmap Generation
📁 Bulk → CSV Report Export
```

---

## 🔬 Explainability — *Occlusion Sensitivity Heatmaps*

Traditional Grad-CAM fails for **non-convolutional models** like LNNs.  
To overcome this, we implemented **Occlusion Sensitivity**, which:

1. Masks small regions step-by-step.  
2. Observes confidence variation.  
3. Highlights zones influencing predictions.

🎨 **Color Map Meaning**
| Color | Meaning |
|:------|:--------|
| 🔴 Red | Highly influential (infected zone) |
| 🟡 Yellow | Moderate influence |
| 🔵 Blue | Neutral / normal region |

> ✨ The resulting heatmap overlays infection intensity directly on X-ray images.

---

## ⚙️ Tech Stack

| Layer | Technology |
|:------|:------------|
| 🖥 Frontend | HTML5, Bootstrap 4, CSS3, JS, Jinja2 |
| ⚙️ Backend | Flask (Python 3.9+) |
| 🧠 AI Framework | PyTorch (Liquid Neural Networks) |
| 🧮 Data Processing | PIL, NumPy, TorchVision |
| 🎨 Visualization | Occlusion Sensitivity + Jet Color Mapping |

---

## 📁 Project Structure

```
Chest-Xray-Dual-Mode-LNN/
├── app.py                  # Flask backend logic
├── models/
│   ├── liquid_model.pth    # Pneumonia model weights
│   └── best_model.pth      # Lung Opacity model weights
├── static/
│   ├── uploads/            # Temporary storage + CSV results
│   └── heatmaps/           # Saved heatmap overlays
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

## 🧩 Setup & Installation

### 🔹 Step 1: Clone Repository
```bash
git clone https://github.com/mXrahul01/Chest-Xray-Dual-Mode-LNN.git
cd Chest-Xray-Dual-Mode-LNN
```

### 🔹 Step 2: Create Virtual Environment
```bash
python -m venv venv
venv\Scripts\activate        # Windows
source venv/bin/activate       # macOS/Linux
```

### 🔹 Step 3: Install Requirements
```bash
pip install -r requirements.txt
```

### 🔹 Step 4: Add Trained Models
```
models/
 ├── liquid_model.pth
 └── best_model.pth
```

### 🔹 Step 5: Run Application
```bash
python app.py
```
Access at 👉 **http://127.0.0.1:5000/**

---

## 📊 Outputs

| Output | Description |
|:--------|:-------------|
| 🧠 `result_single.html` | Heatmap + prediction summary |
| 📁 `result_bulk.html` | Batch result + CSV download |
| 📈 `results.csv` | Combined dataset result |
| ⚕️ `results_disease.csv` | Disease-positive subset |
| 💚 `results_normal.csv` | Normal subset |
| 🩻 `*_heatmap.jpg` | Infected area overlay |

---

## 📈 Model Performance

| Model | Task | Accuracy | AUC | Type |
|:------|:------|:----------|:----|:----|
| LNN-1 | Pneumonia Detection | 94% | 0.98 | Liquid Neural Network |
| LNN-2 | Lung Opacity | 91% | 0.96 | Liquid Neural Network |
| Ensemble | Combined Decision | **92.5%** | **0.97** | Weighted Confidence Fusion |

---

## 🧭 Future Roadmap

- [ ] ⚙️ Grad-CAM++ and Integrated Gradients for CNN comparison  
- [ ] 🧩 3D DICOM support for CT Scans  
- [ ] 🌐 REST API endpoints (FastAPI)  
- [ ] ⚡ Real-time GPU inference optimization  
- [ ] 🧾 Automated PDF Medical Reports  

---

## 🪪 License

Released under the **MIT License** — Free for educational and research use.

---

<div align="center">

### 💙 Developed by [**Rahul (mXrahul01)**](https://github.com/mXrahul01)
**Bridging medical accuracy and AI transparency through Explainable Neural Intelligence.** 🧠

<img src="https://user-images.githubusercontent.com/67873374/194790540-5e4a2eb2-63d7-4c6e-9110-bc42b6d3b8c4.gif" width="120">

</div>
