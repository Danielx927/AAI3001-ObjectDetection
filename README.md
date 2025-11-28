# AAI3001 Fruit Detection Project

This repository contains a comprehensive fruit detection system using YOLO and Faster R-CNN models, with a web interface for real-time inference.

---

## Project Structure

```
AAI3001-Project2/
├── app.py                              # Flask web application for fruit detection
├── helper.py                           # Utility functions for dataset processing
├── model_evaluation.py                 # Model evaluation and metrics calculation
├── requirements.txt                    # Python dependencies
├── data.yaml                           # YOLO dataset configuration
├── metrics.yaml                        # Model performance metrics
│
├── dataset/                            # Dataset directory
│   ├── fruit_images (apple, banana, orange)/  # Kaggle dataset (300 images)
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

- **`dataset/split`**: Contains all training/validation/test data in YOLO format
- **`models/`**: Stores trained model weights for inference
- **`yolo/`**: YOLO-specific training scripts and notebooks
- **`rcnn_family/`**: Faster R-CNN implementation and experiments
- **`evaluation_results/`**: Performance metrics and evaluation outputs of models on test set
- **`templates/`**: Web application frontend files

---

## Running the Application (app.py)

Follow these steps to run the application successfully. The app will NOT run if any step is skipped or done out of order.

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
- All helper libraries used by `app.py` and `app2.py`

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

### 5. Open the Web Interface

Once the server is running:

1. Open your browser
2. Go to:

```
http://127.0.0.1:5000
```

You should now see the full Fruit Detection interface, including:
- Upload button
- Model selection
- Object detection results
- Bounding box highlighting
- Live Video Object Detection
- Performance Metrics

### 6. Optional: Running the Gradio Deployment (`app.py` for Hugging Face)

If you are running the Gradio version instead of Flask, use:

```bash
python app.py
```

You will see a Gradio URL such as:

```
Running on http://127.0.0.1:7860
```

Open that link to use the hosted interface.

### 7. Stopping the App

To safely stop the server:

- Press **CTRL + C** in the terminal.




## Data Collection & Annotation

The dataset consists of **1,504 fruit images** across **10 fruit classes**. Data was collected through a combination of self-annotation and external sources to ensure class balance and diversity.

---

### 1. Self-Annotated Images (1054 images)

A total of **1504 images** were annotated manually using **LabelImg**, of which contains
**around 100 images per class** that were filtered from our previously collected dataset used in the first half of the project on fruit classification. 

---

### 2. External Kaggle Sources (450 images)

To increase variability and representation, **450 additional images** were sourced from **multiple Kaggle datasets**.  
Only images that matched the 10 fruit classes were included.

All imported images were:
- Renamed and reorganized into the project structure
- Re-labelled to ensure consistent class naming
- Checked for duplicates before inclusion
- Converted any non-yolo format labels into yolo format

#### Kaggle Sources Used

Below are the Kaggle datasets referenced (insert actual links in the placeholders):

1. **Dataset Source 1**  
   *Fruit Images for Object Detection:* <https://www.kaggle.com/datasets/mbkinaci/fruit-images-for-object-detection>  
   *Notes:* 300 images of apple, banana and orange were taken from this dataset

2. **Dataset Source 2**  
   *Fruit Detection Dataset:* <https://www.kaggle.com/datasets/lakshaytyagi01/fruit-detection>  
   *Notes:* 150 images of watermelon were taken from this dataset


## Data Splitting & Class Balancing

To ensure robust model evaluation and fair representation across classes, the dataset underwent **stratified train/validation splitting** and **class balancing**:

### 1. Train/Validation Split

- The dataset was split into **training** and **validation** sets using *multi-label iterative stratification*.
  This method preserves each fruit class's distribution between splits, even for images containing multiple fruit types.
- **Validation size:** 20% of the dataset
- All images and their corresponding YOLO label files were kept together within each split.

### 2. Balancing Class Distribution (Undersampling & Oversampling)

After splitting:
- **Class balancing was applied to the training set only** (the validation set was left untouched for realistic evaluation).
- For each fruit class, we targeted equal numbers of labeled objects (bounding boxes):
    - **Undersampling:** If a class had more than the target number of bounding boxes, images and labels were randomly selected so no class exceeded the target.
    - **Oversampling:** If a class was underrepresented, images containing that class were duplicated until the class reached the target count.
- This process ensures no model bias toward majority classes and corrects for natural imbalances after splitting.

> **Final training data features a balanced number of labeled objects per class, maximizing fairness and reliability in model learning.**

