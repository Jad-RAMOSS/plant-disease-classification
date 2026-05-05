# Plant Disease Image Classifier

**Egyptian Chinese University — Data Mining Project**

A deep-learning pipeline for automatic plant disease detection from leaf photographs. Built on **EfficientNetB3** fine-tuned end-to-end on the PlantVillage dataset, achieving **88 % test accuracy** across 38 disease classes spanning 14 plant species. The repository ships two ready-to-use inference interfaces: a **PyQt6 desktop application** and a **Next.js web application**.

---

## Table of Contents

1. [Results](#results)
2. [Dataset](#dataset)
3. [Model Architecture](#model-architecture)
4. [Training Procedure](#training-procedure)
5. [Inference](#inference)
6. [Project Structure](#project-structure)
7. [Tech Stack](#tech-stack)
8. [Installation](#installation)
9. [Usage — Training](#usage--training)
10. [Usage — Desktop GUI](#usage--desktop-gui)
11. [Usage — Web Application](#usage--web-application)
12. [Supported Classes](#supported-classes)

---

## Results

| Split          | Loss   | Accuracy  |
|----------------|--------|-----------|
| Train          | 0.1881 | ~90 %     |
| Validation     | 0.1993 | ~88 %     |
| **Test**       | **0.2004** | **88 %** |

A full per-class precision / recall / F1 classification report is printed automatically at the end of each training run via `scikit-learn.classification_report`.

> The saved model file is named after its test accuracy: `efficientnetb3-Plant Village Disease-88.47.h5`. Both inference interfaces auto-discover the model at startup by scanning `outputs/` for the first `.h5` file whose name does not contain `weights` — replacing the file automatically routes inference to any newly trained model without code changes.

---

## Dataset

**PlantVillage** — a publicly available benchmark of 54,305 healthy and diseased crop leaf images collected under controlled, uniform conditions.

| Property              | Value                              |
|-----------------------|------------------------------------|
| Source                | PlantVillage (color variant only)  |
| Total images          | ~54,305                            |
| Classes               | 38                                 |
| Plant species         | 14                                 |
| Image format          | RGB JPEG                           |
| Input resolution      | 224 × 224 px                       |
| Train / Val / Test    | 80 % / 10 % / 10 % (stratified)   |

Expected directory layout (excluded from version control via `.gitignore`):

```
plantvillage dataset/
├── color/          ← used for training
├── grayscale/      ← unused
└── segmented/      ← unused
```

### Data Pipeline (`data.py`)

1. **Discovery** — `define_paths(data_dir)` walks every class sub-folder and collects absolute file paths and labels into two parallel lists.
2. **DataFrame** — `define_df(files, classes)` combines them into a Pandas DataFrame with columns `filepaths` and `labels`.
3. **Splitting** — `split_data(data_dir)` applies two consecutive `sklearn.train_test_split` calls with `stratify=labels` and `random_state=123`, producing an 80 / 10 / 10 split that is reproducible and class-balanced.
4. **Generators** — `create_gens(...)` builds three Keras `ImageDataGenerator.flow_from_dataframe` pipelines:
   - **Train**: horizontal flip enabled; preprocessing function is the identity (no pixel scaling — EfficientNet's internal rescaling layer handles normalisation).
   - **Validation / Test**: no augmentation, same identity preprocessing.
   - **Test batch size**: computed as the largest divisor of `len(test_df)` that keeps batch size ≤ 80, ensuring every test sample is evaluated exactly once with no remainder dropped.

---

## Model Architecture

```
Input: 224 × 224 × 3 (RGB, raw uint8-range pixel values)
│
├─ EfficientNetB3 (ImageNet weights, trainable=True, pooling='max')
│    Compound-scaled CNN: depth × 1.6, width × 1.8, resolution × 1.2
│    Output: (None, 1536)
│
├─ BatchNormalization (axis=-1, momentum=0.99, epsilon=0.001)
│
├─ Dense(256, activation='relu')
│    kernel_regularizer : L2(λ=0.016)
│    activity_regularizer: L1(λ=0.006)
│    bias_regularizer    : L1(λ=0.006)
│
├─ Dropout(rate=0.45, seed=123)
│
└─ Dense(38, activation='softmax')

Optimizer : Adam(lr=0.001)   [legacy Keras implementation for TF 2.x]
Loss      : categorical_crossentropy
Metrics   : accuracy
```

**Base model** — `tf.keras.applications.efficientnet.EfficientNetB3` pre-trained on ImageNet-1K (1.28 M images, 1000 classes). The top classification layers are replaced; global max pooling is applied at the base.

**Fine-tuning strategy** — `base_model.trainable = True` (full end-to-end fine-tuning by default). A `--freeze` CLI flag sets `trainable = False` to train only the classification head, useful for rapid prototyping on limited compute.

**Regularisation** — Three complementary mechanisms are stacked to prevent overfitting on the relatively small PlantVillage training set:
- L2 weight decay penalises large kernel weights.
- L1 activity regularisation encourages sparse feature activations.
- 45 % Dropout randomly zeroes activations during training.

---

## Training Procedure

### Custom Adaptive Callback (`callbacks.py`)

All training dynamics are governed by `MyCallback`, a bespoke `keras.callbacks.Callback` implementing a **two-phase adaptive learning-rate schedule** with interactive human-in-the-loop control.

#### Phase 1 — Warm-up (`train_accuracy < threshold`)
- Monitors **training accuracy** (`highest_tracc`).
- If no new high is reached within `patience` epochs, the learning rate is multiplied by `factor` (default ×0.5) and `stop_count` is incremented.
- Best weights are saved whenever a new `highest_tracc` is recorded.

#### Phase 2 — Fine-tuning (`train_accuracy ≥ threshold`)
- Switches monitoring to **validation loss** (`lowest_vloss`).
- Same LR-reduction and weight-saving logic, now driven by validation loss improvement.

#### Early Stopping
- If `stop_count > stop_patience` (i.e. the LR has been reduced `stop_patience` times in a row with no improvement), training halts immediately and `model.set_weights(best_weights)` restores the best checkpoint.

#### Interactive Mode
- At every `ask_epoch` interval the callback prints a prompt. The user types:
  - `H` or `h` — halt training immediately.
  - An integer `n` — extend training by `n` more epochs before the next prompt.
- At `on_train_begin`, the user is first asked whether they want interactive prompts at all (`y/n`).

#### Per-epoch Console Output
```
Epoch  Loss      Accuracy  V_loss    V_acc     LR        Next LR   Monitor    % Improv   Duration
  1/40  0.321     91.234    0.29872   92.103    0.00100   0.00100   val_loss    0.00       42.31
```

Per-batch progress (`accuracy` and `loss`) is written on a single overwriting line (`\r`) to avoid log flooding.

### Default Hyperparameters (`config.py`)

| Parameter      | Default | Description                                         |
|----------------|---------|-----------------------------------------------------|
| `img_size`     | (224, 224) | Input resolution (H × W)                        |
| `batch_size`   | 40      | Mini-batch size                                     |
| `learning_rate`| 0.001   | Initial Adam LR                                     |
| `epochs`       | 2       | Maximum epochs (override via `--epochs`)            |
| `patience`     | 1       | Epochs without improvement before LR reduction      |
| `stop_patience`| 3       | LR reductions without improvement before stopping   |
| `threshold`    | 0.9     | Train accuracy required to enter Phase 2            |
| `factor`       | 0.5     | LR reduction multiplier                             |
| `ask_epoch`    | 5       | Epoch interval for interactive continuation prompt  |
| `freeze_base`  | False   | Whether to freeze the EfficientNet backbone         |

### Artefact Saving (`utils.py`)

Two files are written to `outputs/` after training:

| File | Contents |
|------|----------|
| `efficientnetb3-{subject}-{test_acc:.2f}.h5` | Full model (architecture + weights), loadable via `tf.keras.models.load_model` |
| `efficientnetb3-{subject}-weights.h5` | Weights only; requires architecture to be rebuilt before loading |

A **class dictionary CSV** is also saved: `{subject}-class_dict.csv` with columns `class_index`, `class`, `height`, `width`. This CSV is the single source of truth for inference — both the desktop GUI and web app read class labels and image dimensions exclusively from it, making inference fully portable across retraining runs.

---

## Inference

### `predict.py` — Shared inference utilities

Three functions used by both the desktop GUI and the web server:

```python
find_model(outputs_dir='outputs') -> (model_path, class_dict_path)
```
Scans `outputs_dir` for the first `.h5` file not containing `weights` in its name, and the first `.csv` file. Raises `FileNotFoundError` if either is absent.

```python
load_artifacts(model_path, class_dict_path) -> (model, class_dict, img_size)
```
Loads the Keras model with `tf.keras.models.load_model`, reads the CSV into a `{int_index: class_name}` dict, and extracts `img_size` as `(width, height)` for OpenCV.

```python
predict_image(model, class_dict, img_size, image_path) -> List[Dict]
```
Loads the image with OpenCV, converts BGR→RGB, resizes to `img_size`, expands dims to batch of 1, runs `model.predict`, sorts by descending probability, and returns the top-5 predictions as `[{"class": str, "confidence": float (0–100)}, ...]`.

---

## Project Structure

```
project/
│
├── main.py                                 # CLI training entry point
├── gui.py                                  # PyQt6 desktop inference app
├── requirements.txt                        # Python deps (training + GUI)
├── plant-village-disease-classification-acc-99-6.ipynb  # Original Kaggle notebook
├── ECU_Plant_Disease_Proposal-2.pdf        # Project proposal document
│
├── outputs/                                # Generated artefacts
│   ├── *.h5                                # Model weights — gitignored
│   └── *-class_dict.csv                    # Class index mapping — committed
│
├── plant_disease/                          # Core Python package
│   ├── __init__.py
│   ├── config.py                           # Hyperparameter dataclass
│   ├── data.py                             # Dataset discovery, split, Keras generators
│   ├── model.py                            # EfficientNetB3 architecture builder
│   ├── train.py                            # model.fit() wrapper
│   ├── callbacks.py                        # Adaptive two-phase training callback
│   ├── evaluate.py                         # evaluate() + classification report
│   ├── predict.py                          # find_model(), load_artifacts(), predict_image()
│   ├── utils.py                            # save_model(), save_class_dict()
│   └── visualize.py                        # Sample image grid, loss/accuracy curves
│
└── webapp/                                 # Next.js web application
    ├── predict_server.py                   # Flask inference API (port 8001)
    ├── start.sh                            # One-command launcher (Flask + Next.js)
    ├── requirements.txt                    # Flask
    ├── package.json
    ├── next.config.js                      # Proxy rewrite: /api/python/* → :8001
    ├── tailwind.config.js                  # ECU brand colour tokens
    ├── postcss.config.js
    ├── public/
    │   └── ecu-logo.png
    ├── pages/
    │   ├── _app.js
    │   └── index.js                        # Drag-and-drop classifier UI
    └── styles/
        └── globals.css
```

---

## Tech Stack

### Python / Training & GUI

| Library              | Version   | Role                                               |
|----------------------|-----------|----------------------------------------------------|
| TensorFlow / Keras   | 2.15.0    | Model definition, training, serialisation          |
| NumPy                | 1.26.4    | Array operations, prediction post-processing       |
| Pandas               | 2.2.0     | DataFrame data pipeline, CSV I/O                   |
| scikit-learn         | 1.4.0     | Stratified splitting, classification report        |
| OpenCV (`cv2`)       | 4.9.0.80  | Image I/O, BGR→RGB conversion, resizing            |
| Matplotlib           | 3.8.2     | Training curves, sample image grids                |
| Seaborn              | 0.13.1    | Plot styling                                       |
| PyQt6                | 6.11.0    | Desktop GUI (drag-and-drop, threaded inference)    |
| Flask                | 3.x       | Lightweight HTTP inference server for the web app  |

### Web Application

| Technology      | Version | Role                                              |
|-----------------|---------|---------------------------------------------------|
| Next.js         | 16.x    | React framework, dev server, proxy rewrites       |
| React           | 18.x    | Component model, state management                 |
| Tailwind CSS    | 3.x     | Utility-first styling, ECU brand colour tokens    |
| react-dropzone  | 14.x    | Drag-and-drop file upload                         |
| Node.js         | 18+     | Next.js runtime                                   |

### Web Inference Architecture

```
Browser
  │  POST /api/python/predict  (multipart FormData)
  ▼
Next.js dev server  :3000
  │  rewrite rule in next.config.js  →  no CORS needed
  ▼
Flask server  :8001  (predict_server.py)
  │  save to tempfile → predict_image() → delete tempfile
  ▼
TensorFlow model  (loaded once at startup, ~30 s cold start)
  │  returns top-5  [{class, confidence}]
  ▼
Browser renders confidence bars + class dropdown
```

The Flask server exposes four endpoints:

| Endpoint       | Method | Response payload                                              |
|----------------|--------|---------------------------------------------------------------|
| `/health`      | GET    | `{"status": "ok"}`                                            |
| `/model-info`  | GET    | `{"model": "...", "classes": 38, "img_size": [224, 224]}`     |
| `/classes`     | GET    | `{"classes": [{"index": 0, "name": "Apple___Apple_scab"}, …]}`|
| `/predict`     | POST   | `{"predictions": [{"class": "…", "confidence": 97.4}, …]}`   |

---

## Installation

### Prerequisites

- **Python 3.9 – 3.11** (TensorFlow 2.15 does not support Python 3.12+)
- **Node.js 18+**
- **Conda** (recommended)

### 1. Python Environment

```bash
conda create -n plant-disease python=3.11 -y
conda activate plant-disease

# Training + GUI dependencies
pip install -r requirements.txt

# Web server dependency
pip install flask
```

### 2. Node.js Dependencies

```bash
cd webapp
npm install
```

---

## Usage — Training

```bash
conda activate plant-disease

# Default run (dataset at 'plantvillage dataset/color', outputs to 'outputs/')
python main.py

# Custom paths and extended training
python main.py \
  --data-dir "/path/to/plantvillage/color" \
  --save-path outputs \
  --epochs 30 \
  --batch-size 40

# Frozen backbone — train head only (faster, requires less VRAM)
python main.py --freeze --epochs 15
```

After completion, `outputs/` contains:
```
outputs/
├── efficientnetb3-Plant Village Disease-88.47.h5   # full model (loadable directly)
├── efficientnetb3-Plant Village Disease-weights.h5 # weights only
└── Plant Village Disease-class_dict.csv            # class index ↔ name map
```

---

## Usage — Desktop GUI

```bash
conda activate plant-disease
python gui.py
```

The app (`PlantVisionApp`) loads the model in a background `QThread` on startup. A header status label transitions from yellow "Initialising…" to green "Ready" when loading is complete. Drag a leaf image onto the drop zone or click "Browse…" — inference runs in a second background thread and results appear immediately.

**UI layout:**
- **Left panel** — drag-and-drop zone with image preview
- **Right panel** — predicted plant name, disease / condition label, colour-coded confidence percentage + progress bar (green = healthy, red = diseased), top-5 ranked predictions with individual confidences
- **Footer** — active model filename + dataset summary

---

## Usage — Web Application

### One-command launch (recommended)

```bash
cd webapp
./start.sh
```

`start.sh` starts the Flask prediction server, polls `/health` until the model finishes loading (typically 20–40 s), then starts the Next.js dev server. Both processes shut down together on `Ctrl-C`.

### Manual launch (two terminals)

```bash
# Terminal 1 — Python prediction server
conda activate plant-disease
python webapp/predict_server.py
# Prints: "Ready — model: efficientnetb3-Plant Village Disease-88.47.h5"

# Terminal 2 — Next.js frontend
cd webapp && npm run dev
```

Open **http://localhost:3000**.

### Web UI features

| Feature | Description |
|---------|-------------|
| Drag-and-drop upload | JPG, PNG, WebP, BMP — or click to browse |
| Server status badge | Live green / yellow / red indicator in the header |
| Active model name | Fetched from `/model-info`, updates when model is swapped |
| Top prediction banner | Green for healthy, red for diseased; shows confidence % |
| Top-5 confidence bars | Bars proportional to the top prediction's confidence |
| Classes dropdown | Collapsible panel, all 38 classes grouped by plant, live search filter, colour-coded pills |
| About panel | Architecture, dataset, class count, species count |

### Switching models

1. Replace (or add) a `.h5` file in `outputs/`.
2. Restart `predict_server.py`.
3. The UI automatically shows the new model name and class list — no frontend changes needed.

---

## Supported Classes

38 classes across 14 plant species. Raw label format: `PlantName___Condition`.

| Index | Class label | Display name |
|-------|-------------|--------------|
| 0  | `Apple___Apple_scab` | Apple — Apple Scab |
| 1  | `Apple___Black_rot` | Apple — Black Rot |
| 2  | `Apple___Cedar_apple_rust` | Apple — Cedar Apple Rust |
| 3  | `Apple___healthy` | Apple — Healthy |
| 4  | `Blueberry___healthy` | Blueberry — Healthy |
| 5  | `Cherry_(including_sour)___Powdery_mildew` | Cherry — Powdery Mildew |
| 6  | `Cherry_(including_sour)___healthy` | Cherry — Healthy |
| 7  | `Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot` | Corn — Cercospora / Gray Leaf Spot |
| 8  | `Corn_(maize)___Common_rust_` | Corn — Common Rust |
| 9  | `Corn_(maize)___Northern_Leaf_Blight` | Corn — Northern Leaf Blight |
| 10 | `Corn_(maize)___healthy` | Corn — Healthy |
| 11 | `Grape___Black_rot` | Grape — Black Rot |
| 12 | `Grape___Esca_(Black_Measles)` | Grape — Esca (Black Measles) |
| 13 | `Grape___Leaf_blight_(Isariopsis_Leaf_Spot)` | Grape — Leaf Blight |
| 14 | `Grape___healthy` | Grape — Healthy |
| 15 | `Orange___Haunglongbing_(Citrus_greening)` | Orange — Huanglongbing (Citrus Greening) |
| 16 | `Peach___Bacterial_spot` | Peach — Bacterial Spot |
| 17 | `Peach___healthy` | Peach — Healthy |
| 18 | `Pepper,_bell___Bacterial_spot` | Pepper (Bell) — Bacterial Spot |
| 19 | `Pepper,_bell___healthy` | Pepper (Bell) — Healthy |
| 20 | `Potato___Early_blight` | Potato — Early Blight |
| 21 | `Potato___Late_blight` | Potato — Late Blight |
| 22 | `Potato___healthy` | Potato — Healthy |
| 23 | `Raspberry___healthy` | Raspberry — Healthy |
| 24 | `Soybean___healthy` | Soybean — Healthy |
| 25 | `Squash___Powdery_mildew` | Squash — Powdery Mildew |
| 26 | `Strawberry___Leaf_scorch` | Strawberry — Leaf Scorch |
| 27 | `Strawberry___healthy` | Strawberry — Healthy |
| 28 | `Tomato___Bacterial_spot` | Tomato — Bacterial Spot |
| 29 | `Tomato___Early_blight` | Tomato — Early Blight |
| 30 | `Tomato___Late_blight` | Tomato — Late Blight |
| 31 | `Tomato___Leaf_Mold` | Tomato — Leaf Mold |
| 32 | `Tomato___Septoria_leaf_spot` | Tomato — Septoria Leaf Spot |
| 33 | `Tomato___Spider_mites Two-spotted_spider_mite` | Tomato — Spider Mites |
| 34 | `Tomato___Target_Spot` | Tomato — Target Spot |
| 35 | `Tomato___Tomato_Yellow_Leaf_Curl_Virus` | Tomato — Yellow Leaf Curl Virus |
| 36 | `Tomato___Tomato_mosaic_virus` | Tomato — Mosaic Virus |
| 37 | `Tomato___healthy` | Tomato — Healthy |

---

*Egyptian Chinese University · Data Mining Project · EfficientNetB3 · PlantVillage*
