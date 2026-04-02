# Night Vision Object Detection — Project Requirements
## Course Project Context

This is a college course project implementing object detection for night vision / thermal imaging using deep learning (YOLO + transfer learning). The goal is a working, presentable system — not a research-grade production system.

---

## 🎯 Project Goal

Build a deep learning pipeline that:
1. Takes thermal infrared images as input
2. Detects and classifies objects (Person, Car, Bicycle, Dog) using YOLOv8
3. Outputs images/video with bounding boxes drawn
4. Provides a simple demo UI for presentation

---

## 🗂️ Project Structure

```
night-vision-detection/
│
├── data/
│   ├── raw/                    # Original FLIR dataset images
│   ├── processed/              # Resized/normalized images
│   └── dataset.yaml            # YOLOv8 dataset config file
│
├── models/
│   ├── yolov8n.pt              # Pretrained YOLOv8 nano weights (downloaded)
│   └── best.pt                 # Fine-tuned model weights (output of training)
│
├── runs/                       # YOLOv8 auto-saves training runs here
│
├── src/
│   ├── prepare_data.py         # Script to organize dataset into YOLOv8 format
│   ├── train.py                # Fine-tuning / transfer learning script
│   ├── evaluate.py             # Run evaluation metrics (mAP, precision, recall)
│   ├── predict.py              # Run inference on single image or folder
│   └── utils.py                # Helper functions (draw boxes, etc.)
│
├── app/
│   └── demo_app.py             # Streamlit or Gradio demo UI
│
├── notebooks/
│   └── exploration.ipynb       # Optional: data exploration / quick tests
│
├── requirements.txt            # Python dependencies
├── dataset.yaml                # YOLO dataset config (classes, paths)
├── PROJECT_REQUIREMENTS.md     # This file
└── README.md
```

---

## 📦 Python Dependencies

```
# requirements.txt
ultralytics>=8.0.0         # YOLOv8 - main model framework
torch>=2.0.0               # PyTorch backend
torchvision>=0.15.0        # Image transforms
opencv-python>=4.7.0       # Image reading and drawing bounding boxes
Pillow>=9.0.0              # Image handling
numpy>=1.24.0              # Array operations
matplotlib>=3.7.0          # Plotting results
streamlit>=1.28.0          # Demo UI (easy web app)
gradio>=4.0.0              # Alternative demo UI
pyyaml>=6.0                # For reading .yaml config files
scikit-learn>=1.2.0        # For evaluation metrics
tqdm>=4.65.0               # Progress bars
roboflow                   # Optional: to download dataset from Roboflow
```

---

## 📊 Dataset

**Name:** FLIR Thermal Dataset (Free ADAS version)  
**Source Option 1:** https://www.flir.in/oem/adas/adas-dataset-form/  
**Source Option 2:** Roboflow — search "FLIR thermal yolov8" at https://roboflow.com/  
**Recommended subset:** 800–1500 images for quick training  
**Format needed:** YOLOv8 format (images + labels in .txt files)

**Classes:**
```
0: person
1: car
2: bicycle
3: dog
```

**dataset.yaml format:**
```yaml
path: ./data
train: train/images
val: val/images
test: test/images

nc: 4
names: ['person', 'car', 'bicycle', 'dog']
```

---

## 🧠 Model Details

**Base Model:** YOLOv8 Nano (`yolov8n.pt`)  
- Pretrained on COCO dataset (80 classes, RGB images)
- We fine-tune it on thermal grayscale images

**Why YOLOv8n (nano)?**
- Fastest to train
- Smallest model size
- Good enough accuracy for a course project demo

**Transfer Learning Strategy:**
- Load `yolov8n.pt` pretrained weights
- Replace/retrain the detection head for 4 classes
- Freeze backbone layers for first few epochs (optional)
- Fine-tune for 20–30 epochs

---

## 🔧 Training Configuration

```python
# train.py target configuration
MODEL = "yolov8n.pt"        # pretrained base
DATA = "dataset.yaml"       # dataset config
EPOCHS = 25                 # enough for convergence on small dataset
IMG_SIZE = 640              # standard YOLO input size
BATCH_SIZE = 16             # reduce to 8 if GPU memory is low
DEVICE = "0"                # "0" for GPU, "cpu" for CPU-only
PROJECT = "runs/train"
NAME = "night_vision_v1"
```

---

## 📈 Evaluation Metrics Required

The project report needs these metrics:
- **mAP@0.5** (mean Average Precision at IoU 0.5) — primary metric
- **mAP@0.5:0.95** — stricter metric
- **Precision**
- **Recall**
- **F1-Score**
- **Confusion Matrix**
- **Training loss curves** (box loss, cls loss, dfl loss)

YOLOv8 automatically saves all of these in the `runs/train/` folder after training.

---

## 🖥️ Demo App Requirements

**Framework:** Streamlit (preferred — simpler)  
**File:** `app/demo_app.py`

**Features needed:**
1. Upload a thermal image (JPG/PNG)
2. Run inference using fine-tuned `best.pt`
3. Display output image with bounding boxes and labels
4. Show confidence scores per detection
5. Show count of each class detected (e.g., "2 persons, 1 car")

**Optional nice-to-have:**
- Side-by-side: original image vs detected image
- Upload video and process frame by frame
- Confidence threshold slider

---

## 🔄 Data Preprocessing Steps

1. Resize all images to 640x640
2. Thermal images are grayscale — convert to 3-channel (repeat grayscale across RGB channels) OR keep as-is (YOLOv8 handles both)
3. Normalize pixel values to [0, 1]
4. Data augmentation (YOLOv8 does this automatically during training):
   - Horizontal flip
   - Random rotation (±10°)
   - Mosaic augmentation (built-in)
   - Random brightness/contrast

---

## ⚡ Quick Start Commands

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Prepare dataset (after downloading FLIR data)
python src/prepare_data.py

# 3. Train model
python src/train.py

# 4. Evaluate
python src/evaluate.py

# 5. Run inference on an image
python src/predict.py --image path/to/image.jpg

# 6. Launch demo app
streamlit run app/demo_app.py
```

---

## 📝 Scripts to Build (Tell AI IDE to generate these)

### `src/prepare_data.py`
- Takes raw FLIR dataset folder as input
- Splits into train/val/test (70/20/10)
- Copies images and labels into correct YOLOv8 folder structure
- Generates `dataset.yaml`

### `src/train.py`
- Loads YOLOv8n pretrained model
- Runs `model.train()` with config from above
- Saves best weights to `models/best.pt`

### `src/evaluate.py`
- Loads `models/best.pt`
- Runs `model.val()` on test set
- Prints mAP, precision, recall, F1
- Saves confusion matrix plot

### `src/predict.py`
- Loads `models/best.pt`
- Accepts `--image` or `--folder` argument
- Runs inference and saves output images with boxes drawn

### `app/demo_app.py`
- Streamlit app
- Image upload widget
- Inference + visualization
- Class count display

---

## 🎤 Presentation Talking Points

1. **Problem:** Cameras fail at night; thermal imaging captures heat
2. **Solution:** YOLOv8 fine-tuned on thermal images via transfer learning
3. **Dataset:** FLIR thermal dataset (~26k images, 4 classes)
4. **Result:** mAP of X%, real-time inference at Y FPS
5. **Demo:** Live demo with uploaded images showing bounding boxes

---

## ⚠️ Important Notes for AI IDE

- Use `ultralytics` library (YOLOv8) — NOT older darknet/YOLOv5
- Thermal images may be grayscale — handle both grayscale and RGB inputs
- If no GPU available, use `device='cpu'` — training will be slow but works
- Keep dataset small (500–1000 images) for quick training on limited hardware
- Do NOT implement YOLO from scratch — use the `ultralytics` package
- All file paths should use `os.path` or `pathlib.Path` for cross-platform compatibility
