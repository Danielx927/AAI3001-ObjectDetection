# AAI3001 Fruit Detection Project

An automated fruit detection and quality assessment system using deep learning models (YOLO, Faster R-CNN) with a web interface for real-time inference.

---

## Overview

### Problem Statement
Manual fruit inspection is time-consuming, labour-intensive, and susceptible to inconsistency and human error in quality assessment and detecting defective fruits. Traditional methods lack the scalability and precision required for modern food processing and retail operations.

### Motivation
Automated fruit classification systems can enhance efficiency, consistency, and scalability across retail, logistics, and food processing operations. By leveraging deep learning models for object detection and classification, this system provides:
- **Fast and accurate** fruit detection and localization
- **Consistent quality assessment** eliminating human subjectivity
- **Scalable solution** for high-volume processing environments
- **Real-time inference** through an intuitive web interface

### Solution
This project implements a three-stage pipeline combining state-of-the-art object detection models (YOLO, Faster R-CNN) with ResNet-based classifiers to detect fruits, identify their type, and assess their quality automatically.

---

## Object Detection Pipeline

The system uses a **three-stage pipeline** to detect fruits and classify their type and quality:

### Pipeline Architecture

```
Input Image
    ↓
┌─────────────────────────────────────┐
│  Stage 1: Object Detection Model   │
│  (YOLOv8/YOLOv11/Faster R-CNN)     │
│  → Predicts bounding boxes          │
└─────────────────────────────────────┘
    ↓
┌─────────────────────────────────────┐
│  Stage 2: Image Cropping            │
│  → Extract regions based on boxes   │
└─────────────────────────────────────┘
    ↓
┌─────────────────────────────────────┐
│  Stage 3: Classification Models     │
│  ├─ Fruit Type (ResNet18)           │
│  │  → Classifies fruit category     │
│  └─ Quality Assessment (ResNet18)   │
│     → Determines fruit quality      │
└─────────────────────────────────────┘
    ↓
Final Output: Bounding boxes + Type + Quality
```

### Stage Details

1. **Object Detection (Bounding Box Prediction)**
   - Model options: YOLOv8n, YOLOv8m, YOLOv11l, or Faster R-CNN
   - Solely responsible for localizing fruits in the image
   - Outputs: Bounding box coordinates for each detected fruit

2. **Image Cropping**
   - Each predicted bounding box is used to crop the corresponding region from the original image
   - Cropped images are preprocessed for classification

3. **Dual Classification**
   - **Fruit Type Classifier** (`fruit_type_resnet18.pth`): Identifies the specific fruit category
   - **Quality Classifier** (`quality_resnet18.pth`): Assesses the quality/freshness of the fruit
   - Both models run in parallel on each cropped region

### Output

The final detection results include:
- Bounding box coordinates
- Fruit type label
- Quality assessment
- Confidence scores for each prediction

---

## Dataset

### Overview
- **Total Images:** 1,504
- **Total Classes:** 10 fruit types (Apple, Banana, Orange, Watermelon, Grapes, Strawberry, Mango, Pineapple, Kiwi, Pear)
- **Format:** YOLO (bounding box annotations in `.txt` files)
- **Split Ratio:** 75% Train / 15% Validation / 10% Test

### Data Collection & Annotation

The dataset was collected through a combination of self-annotation and external sources to ensure class balance and diversity.

#### Self-Annotated Images (~1,054 images)

A total of **1,504 images** were annotated manually using **LabelImg**, with approximately **100 images per class** filtered from our previously collected dataset used in the first half of the project on fruit classification.

#### External Kaggle Sources (450 images)

To increase variability and representation, **450 additional images** were sourced from multiple Kaggle datasets. Only images that matched the 10 fruit classes were included.

All imported images were:
- Renamed and reorganized into the project structure
- Re-labelled to ensure consistent class naming
- Checked for duplicates before inclusion
- Converted to YOLO format labels

**Kaggle Sources:**

1. **Fruit Images for Object Detection**  
   <https://www.kaggle.com/datasets/mbkinaci/fruit-images-for-object-detection>  
   *300 images of apple, banana and orange*

2. **Fruit Detection Dataset**  
   <https://www.kaggle.com/datasets/lakshaytyagi01/fruit-detection>  
   *150 images of watermelon*

### Data Splitting & Class Balancing

To ensure robust model evaluation and fair representation across classes, the dataset underwent **stratified splitting** and **class balancing**:

#### Train/Validation/Test Split

- The dataset was split using **multi-label iterative stratification** to preserve each fruit class's distribution between splits, even for images containing multiple fruit types
- **Split ratio:** 75% Train / 15% Validation / 10% Test
- All images and their corresponding YOLO label files were kept together within each split

#### Class Balancing

After splitting:
- **Class balancing was applied to the training set only** (validation and test sets were left untouched for realistic evaluation)
- Target: **300 labels per class** in the training set
- Methods used:
    - **Undersampling:** If a class had more than 300 bounding boxes, images and labels were randomly selected until the class count reached the target
    - **Oversampling:** If a class was underrepresented, images containing that class were duplicated until reaching the target count
- This process ensures no model bias toward majority classes and corrects for natural imbalances

> **Final training data features a balanced number of labeled objects per class (300 each), maximizing fairness and reliability in model learning.**

---

## Project Structure

```
AAI3001-Project2/
├── app.py                              # Flask web application for fruit detection
├── helper.py                           # Utility functions for dataset  processing (used in data_collection.ipynb)
├── model_evaluation.py                 # Model evaluation and metrics calculation (used in bbox_accuracy_comparison.ipynb)
├── requirements.txt                    # Python dependencies
├── data.yaml                           # YOLO dataset configuration
├── metrics.yaml                        # Model performance metrics
│
├── dataset/                            # Dataset directory
│   ├── fruit_images (apple, banana, orange)/  # Kaggle dataset (300 images)
│   ├── input                           # Consolidated set of images and corresponding labels
│       ├── images/                     
│       ├── labels/                     
│   └── split/                          # Train/val/test splits
│       ├── train/
│       │   ├── images/                 # Training images
│       │   └── labels/                 # YOLO format labels
│       ├── val/
│       │   ├── images/
│       │   └── labels/
│       └── test/
│           ├── images/
│           └── labels/
│
├── models/                             # Trained model weights
│   ├── yolov8n.pth                     # YOLOv8 Nano
│   ├── yolov8m.pth                     # YOLOv8 Medium
│   ├── yolov11l.pth                    # YOLOv11 Large
│   ├── faster_rcnn_fruits.pth          # Faster R-CNN
│   ├── fruit_type_resnet18.pth         # Fruit classification model
│   └── quality_resnet18.pth            # Quality assessment model
│
├── yolo/                               # YOLO training and inference
│   ├── train.ipynb                     # YOLO training notebook
│   ├── model_inference.ipynb           # YOLO inference notebook
│   ├── yolov8n.pt                      # Ultralytics format weights
│   ├── yolov8m.pt
│   ├── yolo11l.pt
│   └── runs/                           # Training run outputs
│
├── rcnn_family/                        # Faster R-CNN implementation
│   ├── train_faster_rcnn.py            # Training script
│   ├── rcnn_model.py                   # Model architecture
│   ├── rcnn_dataset.py                 # Dataset loader
│   └── rcnn_experiments.ipynb          # Experimentation notebook
│
├── runs/                               # YOLO training runs
│   ├── fruits_yolov8n/
│   ├── fruits_yolov8m/
│   └── fruits_yolov11l/
│
├── evaluation_results/                 # Model evaluation outputs
│   ├── yolov8n/
│   ├── yolov8m/
│   ├── yolov11l/
│   └── faster_rcnn/
│
├── test_results/                      # Inference results
│   ├── yolo/
│   └── faster_rcnn/
│
├── templates/                         # Flask HTML templates
│   └── index.html
│
├── data_collection.ipynb              # Dataset preparation notebook
└── bbox_accuracy_comparison.ipynb     # Model comparison analysis
```

### Key Directories

- **`dataset/split/`**: Contains all training/validation/test data in YOLO format
- **`models/`**: Stores trained model weights for inference
- **`yolo/`**: YOLO-specific training scripts and notebooks
- **`rcnn_family/`**: Faster R-CNN implementation and experiments
- **`evaluation_results/`**: Performance metrics and evaluation outputs on test set
- **`templates/`**: Web application frontend files

---

## Running the Application

Follow these steps to run the web application successfully. The app will NOT run if any step is skipped or done out of order.

### 1. Create a New Python Virtual Environment (REQUIRED)

It is highly recommended to isolate dependencies.

**Mac / Linux**
```bash
python3 -m venv venv
source venv/bin/activate
```

**Windows**
```bash
python -m venv venv
venv\Scripts\activate
```

You must see the `(venv)` prefix in your terminal before continuing.

### 2. Install All Required Dependencies

You MUST install using the provided `requirements.txt`:

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

This installs:
- PyTorch + Torchvision  
- Ultralytics YOLO  
- Flask  
- OpenCV  
- Pillow  
- Gradio (if used for deployment)  
- All helper libraries

If installation fails, ensure you have Python **3.9–3.11**.

### 3. Run the Application

Start the Flask app using:

```bash
python app.py
```

You should see output similar to:

```
* Running on http://127.0.0.1:5000 (Press CTRL+C to quit)
```

### 4. Open the Web Interface

Once the server is running:

1. Open your browser
2. Go to: `http://127.0.0.1:5000`

You should now see the full Fruit Detection interface, including:
- Upload button
- Model selection
- Object detection results
- Bounding box highlighting
- Live Video Object Detection
- Performance Metrics

### 5. Stopping the App

To safely stop the server:

- Press **CTRL + C** in the terminal

---

## Model Deployment

All trained models are deployed to Hugging Face Hub for easy access and sharing:

**Repository:** [Danielx927/fruit-detection-models](https://huggingface.co/Danielx927/fruit-detection-models)

**Available Models:**
- `yolov8n.pt` - YOLOv8 Nano
- `yolov8m.pt` - YOLOv8 Medium
- `yolo11l.pt` - YOLO11 Large
- `faster_rcnn.pth` - Faster R-CNN

**Usage:**

```python
from huggingface_hub import hf_hub_download
from ultralytics import YOLO

# Download and load a YOLO model
model_path = hf_hub_download(repo_id="Danielx927/fruit-detection-models", filename="yolov8n.pt")
model = YOLO(model_path)
results = model.predict('image.jpg')
```

---
