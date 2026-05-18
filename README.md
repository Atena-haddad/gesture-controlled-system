# Gesture-Controlled System using Computer Vision and Machine Learning

## Overview

This project is a real-time hand gesture recognition system developed for an Intelligent Consumer Technologies course.

The system allows users to control applications, such as presentation slides, using webcam-based hand gestures. It uses MediaPipe to detect hand landmarks, extracts normalized landmark features, and classifies gestures using both a rule-based approach and a machine learning model.

The main goal of the project is to create a simple, touchless, and intuitive interaction system that can recognize hand gestures in real time and convert them into computer commands.

---

## Supported Gestures

The system currently supports five hand gestures:

- `OPEN_PALM`
- `FIST`
- `POINT_LEFT`
- `POINT_RIGHT`
- `THUMBS_UP`

These gestures can be mapped to different actions, such as moving forward or backward in a presentation.

---

## System Architecture

The system follows this pipeline:

```text
Camera → MediaPipe Hand Detection → Feature Extraction → Gesture Classification → Application Control
```

The workflow is:

1. A webcam captures the user's hand in real time.
2. MediaPipe detects 21 hand landmarks.
3. Each landmark is represented by x, y, and z coordinates.
4. The landmarks are converted into a 63-dimensional feature vector.
5. The gesture is classified using either:
   - a rule-based classifier
   - a k-Nearest Neighbors machine learning classifier
6. The predicted gesture is mapped to an application control action.

---

## Technologies Used

- Python
- OpenCV
- MediaPipe
- NumPy
- k-Nearest Neighbors
- PyAutoGUI

---

## Feature Extraction

MediaPipe provides 21 landmarks for each detected hand.

Each landmark has three coordinates:

```text
x, y, z
```

Therefore, each hand pose is represented as:

```text
21 landmarks × 3 coordinates = 63 features
```

To make the features more stable, the coordinates are normalized using:

- wrist-centered transformation
- scale normalization based on the wrist-to-middle-MCP distance

This helps reduce the effect of hand position and hand size in the camera frame.

---

## Dataset

The dataset was manually collected using webcam input.

Dataset characteristics:

- 304 gesture samples
- Around 60 samples per gesture
- 5 gesture classes
- 63 features per sample
- Features extracted from MediaPipe hand landmarks

The gesture classes are:

```text
FIST
OPEN_PALM
POINT_LEFT
POINT_RIGHT
THUMBS_UP
```

---

## Model

The machine learning model uses k-Nearest Neighbors.

Model configuration:

```text
Algorithm: k-Nearest Neighbors
k = 7
Train/test split = 80/20
```

Two gesture classification methods were implemented and compared:

1. Rule-based classification using geometric thresholds
2. Machine learning classification using k-NN

---

## Results

| Method | Live Accuracy |
|---|---:|
| Rule-based classifier | 75% |
| k-NN classifier | 95% |

The machine learning approach showed better real-time robustness than the rule-based system.

The rule-based classifier is easier to understand and does not require training, but it is more sensitive to camera angle, hand size, and threshold values.

The k-NN classifier performs better because it learns from collected gesture examples instead of relying only on manually defined rules.

---

## Project Structure

```text
gesture-controlled-system/
│
├── README.md
├── requirements.txt
├── collect_dataset.py
├── train_knn.py
├── gesture_controller.py
├── gesture_controller_hybrid.py
├── live_eval.py
│
├── data/
│   └── gestures_dataset.csv
│
├── models/
│   └── knn_model.npz
│
├── docs/
│   └── presentation.pdf
│
└── media/
    └── demo.mp4
```

---

## File Descriptions

| File | Description |
|---|---|
| `collect_dataset.py` | Collects hand gesture samples using the webcam |
| `train_knn.py` | Trains the k-NN gesture classifier |
| `gesture_controller.py` | Runs the rule-based gesture controller |
| `gesture_controller_hybrid.py` | Runs the hybrid controller with RULE and ML modes |
| `live_eval.py` | Tests live prediction accuracy |
| `requirements.txt` | Lists the required Python libraries |
| `data/gestures_dataset.csv` | Collected gesture dataset |
| `models/knn_model.npz` | Saved k-NN model |
| `docs/` | Project presentation and documentation |
| `media/` | Demo videos or GIFs |

---

## Installation

### 1. Clone the repository

```bash
git clone https://github.com/YOUR_USERNAME/gesture-controlled-system.git
cd gesture-controlled-system
```

Replace `YOUR_USERNAME` with your GitHub username.

### 2. Create a virtual environment

Optional but recommended:

```bash
python -m venv venv
```

Activate it:

On macOS/Linux:

```bash
source venv/bin/activate
```

On Windows:

```bash
venv\Scripts\activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

---


##Recommended Python version: Python 3.11

This project was tested in a Conda environment with Python 3.11.


## How to Run

### 1. Collect gesture data

```bash
python collect_dataset.py
```

Use the keyboard labels shown on the screen to record gestures.

The default controls are:

```text
1 = OPEN_PALM
2 = FIST
3 = POINT_LEFT
4 = POINT_RIGHT
5 = THUMBS_UP
s = save dataset
ESC = quit
```

---

### 2. Train the k-NN model

```bash
python train_knn.py
```

This trains the k-NN classifier and saves the model.

---

### 3. Run the rule-based gesture controller

```bash
python gesture_controller.py
```

This version uses manually defined geometric rules to recognize gestures.

---

### 4. Run the hybrid controller

```bash
python gesture_controller_hybrid.py
```

In the hybrid controller:

```text
m = switch between RULE mode and ML mode
ESC = quit
```

This allows comparison between the rule-based method and the machine learning method.

---

### 5. Run live evaluation

```bash
python live_eval.py
```

This script allows live testing of the predicted gesture against manually entered ground-truth labels.

Controls:

```text
m = switch between RULE and ML mode
r = reset counters
1 = OPEN_PALM
2 = FIST
3 = POINT_LEFT
4 = POINT_RIGHT
5 = THUMBS_UP
ESC = quit
```

---

## Example Use Case

One example use case is presentation control.

For example:

| Gesture | Action |
|---|---|
| `OPEN_PALM` | Next slide |
| `FIST` | Previous slide |
| `THUMBS_UP` | Play/Pause or confirmation action |

This makes it possible to control slides without touching the keyboard or mouse.

---

## Limitations

- The dataset was collected from a single user.
- Performance may change with different lighting conditions.
- Performance may change with different camera positions.
- Rule-based classification is sensitive to threshold values.
- The model has limited cross-user validation.
- The system currently recognizes only a small number of gestures.

---

## Future Improvements

Possible future improvements include:

- Collecting a larger dataset from multiple users
- Adding more gesture classes
- Testing the system under different lighting conditions
- Improving robustness for different hand sizes and camera angles
- Adding a graphical user interface
- Using temporal models for gesture sequences
- Exploring deep learning models for gesture classification
- Testing the system in real-world scenarios

---

## Course Context

This project was developed as part of an Intelligent Consumer Technologies course.

It demonstrates the use of computer vision and machine learning for real-time human-computer interaction.

---

## Author

Developed by Atena Haddad Mirlangerood.

---

## License

This project is for educational and portfolio purposes.
