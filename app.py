from flask import Flask, request, redirect, url_for, render_template, jsonify, abort
from werkzeug.utils import secure_filename
import os
import csv
import uuid
import math
from typing import Dict, Any, List, Tuple

import torch
import torch.nn as nn
from PIL import Image, ImageOps
from torchvision import transforms
import numpy as np

# ===================== Flask App & Config ===================== #
BASE_DIR = os.path.abspath(os.path.dirname(__file__))

app = Flask(__name__)

# Folders (relative, deployment-safe)
UPLOAD_DIR = os.path.join(BASE_DIR, "static", "uploads")
HEATMAP_DIR = os.path.join(BASE_DIR, "static", "heatmaps")
REPORT_DIR = os.path.join(BASE_DIR, "static", "reports")
MODEL_DIR = os.path.join(BASE_DIR, "models")

for d in (UPLOAD_DIR, HEATMAP_DIR, REPORT_DIR, MODEL_DIR):
    os.makedirs(d, exist_ok=True)

# Security & upload limits
ALLOWED_EXTENSIONS = {"png", "jpg", "jpeg"}
app.config["UPLOAD_FOLDER"] = UPLOAD_DIR
app.config["MAX_CONTENT_LENGTH"] = 256 * 1024 * 1024  # 256 MB max request

# ===================== Models ===================== #
class LiquidNeuron(nn.Module):
    def __init__(self, input_size, hidden_size):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.W_in = nn.Linear(input_size, hidden_size)
        self.W_rec = nn.Linear(hidden_size, hidden_size)
        self.tau = nn.Parameter(torch.ones(hidden_size))

    def forward(self, x, h_prev):
        # (No temporal unrolling here; behaves like a gated FC)
        h_current = (1 - self.tau) * h_prev + self.tau * torch.tanh(self.W_in(x) + self.W_rec(h_prev))
        return h_current

class LiquidNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super().__init__()
        self.liquid_neuron = LiquidNeuron(input_size, hidden_size)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        b = x.size(0)
        h = torch.zeros(b, self.liquid_neuron.hidden_size, device=x.device)
        x = x.view(b, -1)  # Flatten 224x224
        h = self.liquid_neuron(x, h)
        return self.fc(h)

# Load models relative to project (no hard-coded absolute paths)
DEVICE = torch.device("cpu")  # keep CPU for portability
model1 = LiquidNN(224*224, 128, 2).to(DEVICE)  # "Pneumonia"
model2 = LiquidNN(224*224, 128, 2).to(DEVICE)  # "Lung Opacity"

# Expect these filenames to be present in ./models/
M1_PATH = os.path.join(MODEL_DIR, "liquid_model.pth")
M2_PATH = os.path.join(MODEL_DIR, "best_model.pth")

if not (os.path.exists(M1_PATH) and os.path.exists(M2_PATH)):
    # Give a clear error early if models missing
    raise FileNotFoundError(
        f"Model weights not found. Expected:\n  - {M1_PATH}\n  - {M2_PATH}"
    )

model1.load_state_dict(torch.load(M1_PATH, map_location=DEVICE))
model2.load_state_dict(torch.load(M2_PATH, map_location=DEVICE))
model1.eval()
model2.eval()

# ===================== Transforms & Helpers ===================== #
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.Grayscale(),                # 1-channel
    transforms.ToTensor(),                 # [1,224,224], 0..1
    transforms.Normalize(mean=[0.5], std=[0.5])
])

LABELS_1 = ['Normal', 'Pneumonia']     # model1
LABELS_2 = ['Normal', 'Lung Opacity']  # model2

def allowed_file(filename: str) -> bool:
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def preprocess_pil(path: str) -> Image.Image:
    # ensure grayscale 8-bit, handle corrupted gracefully
    img = Image.open(path).convert("L")
    return ImageOps.exif_transpose(img)

def pil_to_tensor(img: Image.Image) -> torch.Tensor:
    return transform(img).unsqueeze(0).to(DEVICE)

@torch.no_grad()
def predict_both(t_img: torch.Tensor) -> Dict[str, Any]:
    """
    Returns predictions of both models + combined final decision.
    """
    out1 = model1(t_img)
    out2 = model2(t_img)
    p1 = torch.softmax(out1, dim=1)
    p2 = torch.softmax(out2, dim=1)

    conf1, idx1 = torch.max(p1, 1)
    conf2, idx2 = torch.max(p2, 1)
    conf1 = conf1.item()
    conf2 = conf2.item()
    pred1 = LABELS_1[idx1.item()]
    pred2 = LABELS_2[idx2.item()]

    # Final pick = higher confidence
    if conf1 >= conf2:
        final_pred = pred1
        final_conf = conf1
        is_disease = pred1 != "Normal"
        final_model = "model1"
    else:
        final_pred = pred2
        final_conf = conf2
        is_disease = pred2 != "Normal"
        final_model = "model2"

    return {
        "pred1": pred1, "conf1": conf1,
        "pred2": pred2, "conf2": conf2,
        "final_pred": final_pred, "final_conf": final_conf,
        "is_disease": is_disease, "final_model": final_model,
        "p1_all": p1.squeeze(0).cpu().numpy().tolist(),
        "p2_all": p2.squeeze(0).cpu().numpy().tolist(),
    }

def risk_tier(is_disease: bool, final_conf: float) -> str:
    if not is_disease: return "Normal"
    if final_conf >= 0.90: return "High"
    if final_conf >= 0.75: return "Medium"
    if final_conf >= 0.50: return "Low"
    return "Borderline"

# ---------- Explainability: Occlusion Sensitivity (Grad-CAM alternative) ---------- #
# Our models are FC-based (no conv feature maps), so classic Grad-CAM is not meaningful.
# Occlusion Sensitivity: slide a window across the image and measure confidence drop.
def occlusion_heatmap(
        pil_img: Image.Image,
        target_model: nn.Module,
        disease_class_idx: int = 1,
        win: int = 28,
        stride: int = 14
) -> np.ndarray:
    """
    Returns a (H,W) heatmap in [0,1] indicating influential regions for the disease class.
    """
    pil_resized = pil_img.resize((224, 224))
    base_tensor = pil_to_tensor(pil_resized)  # [1,1,224,224]

    with torch.no_grad():
        base_prob = torch.softmax(target_model(base_tensor), dim=1)[0, disease_class_idx].item()

    H, W = 224, 224
    heat = np.zeros((H, W), dtype=np.float32)

    # Work on a copy as numpy for fast masking
    base_np = ((base_tensor[0, 0].cpu().numpy() * 0.5) + 0.5)  # de-normalized to 0..1
    for y in range(0, H - win + 1, stride):
        for x in range(0, W - win + 1, stride):
            occluded = base_np.copy()
            occluded[y:y+win, x:x+win] = 0.5  # neutral occlusion
            # back to tensor with normalization
            occ_t = torch.from_numpy((occluded - 0.5) / 0.5).float()[None, None, :, :].to(DEVICE)
            with torch.no_grad():
                prob = torch.softmax(target_model(occ_t), dim=1)[0, disease_class_idx].item()
            delta = max(0.0, base_prob - prob)  # drop in disease prob
            heat[y:y+win, x:x+win] += delta

    # normalize to [0,1]
    if heat.max() > 0:
        heat = heat / heat.max()
    return heat

def overlay_heatmap_on_pil(pil_img: Image.Image, heatmap: np.ndarray, alpha: float = 0.4) -> Image.Image:
    """
    Colorize heatmap and overlay on original grayscale image.
    """
    import matplotlib.cm as cm
    pil_resized = pil_img.resize((224, 224)).convert("RGB")
    color_map = cm.get_cmap("jet")
    heat_rgba = (color_map(heatmap) * 255).astype(np.uint8)  # RGBA
    heat_rgb = Image.fromarray(heat_rgba[:, :, :3]).resize(pil_resized.size, Image.NEAREST)
    overlay = Image.blend(pil_resized, heat_rgb, alpha=alpha)
    return overlay

# ===================== CSV Helpers ===================== #
def write_csv_all(results: List[Dict[str, Any]], path: str):
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow([
            "Filename",
            "Model1 Prediction", "Model1 Confidence",
            "Model2 Prediction", "Model2 Confidence",
            "Final Prediction", "Final Confidence", "Is Disease", "Risk Tier"
        ])
        for r in results:
            w.writerow([
                r["filename"], r["prediction1"], f"{r['confidence1']:.4f}",
                r["prediction2"], f"{r['confidence2']:.4f}",
                r["final_prediction"], f"{r['final_confidence']:.4f}",
                r["is_disease"], r["risk_tier"]
            ])

def write_csv_subset(results: List[Dict[str, Any]], path: str):
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow([
            "Filename",
            "Model1 Prediction", "Model1 Confidence",
            "Model2 Prediction", "Model2 Confidence",
            "Final Prediction", "Final Confidence", "Risk Tier"
        ])
        for r in results:
            w.writerow([
                r["filename"], r["prediction1"], f"{r['confidence1']:.4f}",
                r["prediction2"], f"{r['confidence2']:.4f}",
                r["final_prediction"], f"{r['final_confidence']:.4f}",
                r["risk_tier"]
            ])

# ===================== Routes ===================== #
@app.route("/")
def index():
    return render_template("index.html")

# ---------- Single Image Mode ---------- #
@app.route("/upload_single", methods=["GET", "POST"])
def upload_single():
    if request.method == "GET":
        return render_template("upload_single.html")

    # POST
    file = request.files.get("image")
    if not file or not allowed_file(file.filename):
        return render_template("upload_single.html", error="Please upload a valid PNG/JPG image.")

    filename = secure_filename(file.filename)
    unique_name = f"{uuid.uuid4().hex}_{filename}"
    file_path = os.path.join(UPLOAD_DIR, unique_name)
    file.save(file_path)

    try:
        pil_img = preprocess_pil(file_path)
        t_img = pil_to_tensor(pil_img)
        pred = predict_both(t_img)

        # Select target model for heatmap (the one that dominated final_pred)
        target_model = model1 if pred["final_model"] == "model1" else model2
        # disease class index is 1 for both models
        heat = occlusion_heatmap(pil_img, target_model, disease_class_idx=1, win=28, stride=14)
        overlay = overlay_heatmap_on_pil(pil_img, heat, alpha=0.45)

        # Save overlay
        os.makedirs(HEATMAP_DIR, exist_ok=True)
        heat_name = unique_name.rsplit(".", 1)[0] + "_heatmap.jpg"
        heat_path = os.path.join(HEATMAP_DIR, heat_name)
        overlay.save(heat_path)

        final_tier = risk_tier(pred["is_disease"], pred["final_conf"])
        # Provide template values
        return render_template(
            "result_single.html",
            image_path=url_for("static", filename=f"uploads/{unique_name}"),
            heatmap_path=url_for("static", filename=f"heatmaps/{heat_name}"),
            pred1=pred["pred1"], conf1=pred["conf1"],
            pred2=pred["pred2"], conf2=pred["conf2"],
            final_label=pred["final_pred"], final_conf=pred["final_conf"],
            risk_tier=final_tier
        )

    except Exception as e:
        print("Single analysis error:", e)
        abort(500, description="Failed to analyze the image.")
    finally:
        # Keep the uploaded image for display; if you want to auto-delete after page render,
        # move deletion into a background cleanup task or delete on a later route.
        pass

# ---------- Bulk Mode ---------- #
@app.route("/upload_bulk", methods=["GET", "POST"])
def upload_bulk():
    if request.method == "GET":
        return render_template("upload_bulk.html")

    # POST
    if "folder" not in request.files:
        return redirect(request.url)

    files = request.files.getlist("folder")
    if not files:
        return render_template("upload_bulk.html", error="Please select at least one image.")

    # Safety cap
    if len(files) > 1000:
        return render_template("upload_bulk.html", error="Limit 1000 images per batch.")

    results: List[Dict[str, Any]] = []

    for file in files:
        if not (file and allowed_file(file.filename)):
            continue

        filename = secure_filename(file.filename)
        unique_name = f"{uuid.uuid4().hex}_{filename}"
        path = os.path.join(UPLOAD_DIR, unique_name)
        file.save(path)
        try:
            img = preprocess_pil(path)
            t_img = pil_to_tensor(img)
            pred = predict_both(t_img)

            final_prediction = pred["final_pred"]
            final_confidence = pred["final_conf"]
            is_disease = pred["is_disease"]
            tier = risk_tier(is_disease, final_confidence)

            results.append({
                "filename": unique_name,
                "prediction1": pred["pred1"], "confidence1": pred["conf1"],
                "prediction2": pred["pred2"], "confidence2": pred["conf2"],
                "final_prediction": final_prediction,
                "final_confidence": final_confidence,
                "is_disease": is_disease,
                "risk_tier": tier
            })
        except Exception as e:
            print(f"Error processing {filename}:", e)
        finally:
            # Bulk mode: delete image to save storage.
            try:
                if os.path.exists(path):
                    os.remove(path)
            except Exception as e:
                print("Cleanup failed:", e)

    # Sort disease cases by max confidence descending
    disease_results = [r for r in results if r["is_disease"]]
    normal_results = [r for r in results if not r["is_disease"]]
    disease_results.sort(key=lambda r: r["final_confidence"], reverse=True)

    # Write CSVs
    all_csv = os.path.join(UPLOAD_DIR, "results.csv")
    disease_csv = os.path.join(UPLOAD_DIR, "results_disease.csv")
    normal_csv = os.path.join(UPLOAD_DIR, "results_normal.csv")
    write_csv_all(results, all_csv)
    write_csv_subset(disease_results, disease_csv)
    write_csv_subset(normal_results, normal_csv)

    # Redirect to a loading screen that will auto-nav to result_bulk
    return redirect(url_for("loading_bulk"))

@app.route("/loading_bulk")
def loading_bulk():
    # Use your refined loading.html (that redirects to result_bulk)
    return render_template("loading.html")

@app.route("/result_bulk")
def result_bulk():
    disease_csv = os.path.join(UPLOAD_DIR, "results_disease.csv")
    results = []
    if os.path.exists(disease_csv):
        with open(disease_csv, "r") as f:
            reader = csv.reader(f)
            next(reader, None)
            for row in reader:
                results.append({
                    "filename": row[0],
                    "prediction1": row[1], "confidence1": float(row[2]),
                    "prediction2": row[3], "confidence2": float(row[4]),
                    "final_prediction": row[5],
                    "final_confidence": float(row[6]) if len(row) > 6 else max(float(row[2]), float(row[4]))
                })
    # Sort top 15 by max confidence (as before) agr tum chahte ho ki result sirf kuch ka ho to "top_disease_results = results[:10] 15,20 kuch bhi use kar sakte ho"
    results.sort(key=lambda x: max(x["confidence1"], x["confidence2"]), reverse=True)
    top_disease_results = results
    return render_template("result_bulk.html", results=top_disease_results)

# Optional: readiness endpoint if you want polling instead of fixed timeout
@app.route("/result_bulk_ready")
def result_bulk_ready():
    ready = os.path.exists(os.path.join(UPLOAD_DIR, "results.csv"))
    return jsonify({"ready": ready})

# Static file passthrough (kept for compatibility)
@app.route("/uploads/<filename>")
def uploaded_file(filename):
    return redirect(url_for("static", filename=f"uploads/{filename}"), code=301)

# ===================== Run ===================== #
if __name__ == "__main__":
    # For prod: use gunicorn/uvicorn; debug only in dev
    app.run(debug=True)
