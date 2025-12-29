# EE 4065 - Embedded Digital Image Processing
## Homework 6: Handwritten Digit Recognition with CNN Models

**Student:** Kaan Atalay  
**ID:** 150720057  
**Due Date:** January 15, 2026

---

## Project Description

This project implements handwritten digit recognition using three different CNN architectures:
- **SqueezeNet** - Compact model with fire modules
- **EfficientNet-B0** - Compound scaling with MBConv blocks
- **MobileNet** - Depthwise separable convolutions

Based on Section 13.7 from:
> C. Ünsalan, B. Höke, and E. Atmaca, "Embedded Machine Learning with Microcontrollers: Applications on STM32 Boards", Springer Nature, ISBN: 978-3031709111, 2025

---

## Project Structure

```
embedded-hw6/
├── src/
│   ├── __init__.py
│   ├── data_loader.py      # MNIST data loading and preprocessing
│   ├── train.py            # Training and evaluation script
│   └── models/
│       ├── __init__.py
│       ├── squeezenet.py   # SqueezeNet implementation
│       ├── efficientnet.py # EfficientNet-B0 implementation
│       └── mobilenet.py    # MobileNet implementation
├── report/
│   ├── main.tex            # LaTeX report source
│   └── main.pdf            # Compiled PDF report
├── results/                # Training results and figures
├── models/                 # Saved trained models
├── requirements.txt        # Python dependencies
└── README.md
```

---

## Installation

1. Create a virtual environment (recommended):
```bash
python -m venv venv
venv\Scripts\activate  # Windows
source venv/bin/activate  # Linux/Mac
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

---

## Usage

### Train all models:
```bash
cd src
python train.py
```

### Test individual models:
```bash
python -c "from models.squeezenet import get_squeezenet_summary; get_squeezenet_summary()"
```

---

## Results

After training, the following files will be generated:

- `results/sample_images.png` - Sample MNIST images
- `results/*_training_history.png` - Training curves for each model
- `results/*_confusion_matrix.png` - Confusion matrices
- `results/model_comparison.png` - Model comparison chart
- `results/results_summary.json` - Numerical results
- `models/*_best.keras` - Best model weights

---

## Report

The LaTeX report is located in `report/main.tex`. To compile:

```bash
cd report
pdflatex main.tex
```

---

## License

This project is for educational purposes only.

