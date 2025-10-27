# Brain-Tumor-Segmentation-using-UNETR-in-TensorFlow
This repository demonstrates the utilization of UNETR (Vision Transformer) for brain tumor segmentation using TensorFlow.

## Setup Instructions

### 1. Clone the Repository
```bash
git clone https://github.com/yourusername/Brain-Tumor-Segmentation-using-UNETR-in-TensorFlow.git
cd Brain-Tumor-Segmentation-using-UNETR-in-TensorFlow
```

### 2. Create Virtual Environment
```bash
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Dataset
The dataset contains 3064 pairs of MRI brain images and their respective binary mask indicating tumor.
The dataset is already included in the `BraTS_data/` directory.

## Usage

### Training the Model
```bash
python train.py
```
- The model will be trained for 500 epochs
- Checkpoints will be saved in the `files/` directory
- Training uses Dice loss and SGD optimizer
- **GPU Support**: Automatically uses Apple Silicon GPU (Metal) if available
- **TensorBoard logs** are saved in `logs/` directory

### Monitoring Training

#### 1. Check GPU Availability
```bash
python check_gpu.py
```

#### 2. Monitor with TensorBoard (Recommended)
**Terminal 1 - Start TensorBoard:**
```bash
python start_tensorboard.py
# Or manually: tensorboard --logdir=logs
```
Then open http://localhost:6006 in your browser to view:
- **Graphs**: Model architecture visualization
- **Scalars**: Loss and metrics over epochs
- **Histograms**: Weight distributions
- **Images**: Sample predictions (if enabled)

**Terminal 2 - Start Training:**
```bash
source brainenv/bin/activate
python train.py
```

#### 3. Resource Monitoring
```bash
# Simple terminal monitor
python monitor_resources.py

# Advanced monitoring with logging
python monitor_training.py --pid <PID>

# Built-in macOS monitoring
top -o cpu -s 2
```

### Testing the Model
```bash
python test.py
```
- Loads the trained model and generates predictions
- Results are saved in the `results/` directory
- Each result shows: Input Image → Ground Truth → Prediction

### Explainability with Grad-CAM

The repository now includes **Grad-CAM (Gradient-weighted Class Activation Mapping)** for model explainability, allowing you to visualize which regions of the input image contribute most to the tumor segmentation decisions.

#### Basic Usage with Grad-CAM
```bash
# Generate Grad-CAM for 10 test samples
python test.py --gradcam --gradcam-samples 10

# Use specific target layers
python test.py --gradcam --gradcam-layer conv_block_7 conv_block_6

# Customize visualization
python test.py --gradcam --gradcam-alpha 0.5 --gradcam-colormap hot

# Process all test samples with Grad-CAM
python test.py --gradcam --gradcam-samples 0
```


#### Grad-CAM Features
- **Automatic Layer Detection**: Automatically identifies optimal convolutional layers for Grad-CAM
- **Multi-layer Aggregation**: Combines features from multiple decoder layers for comprehensive visualization
- **Flexible Visualization**: Customizable colormaps, transparency, and output formats
- **Comprehensive Reports**: HTML reports with visual summaries
- **Metric Calculation**: Optional computation of evaluation metrics

#### Grad-CAM Output Structure
```
results/gradcam/
├── case_001_original.png          # Original input image
├── case_001_heatmap.png           # Grad-CAM heatmap
├── case_001_overlay.png           # Heatmap overlay on image
├── case_001_prediction.png        # Model prediction
├── case_001_ground_truth.png      # Ground truth mask
├── case_001_comparison.png        # Side-by-side comparison
├── case_001_metadata.json         # Processing metadata and metrics
└── gradcam_report.html            # HTML report summary
```

#### Recommended Target Layers
The implementation automatically detects suitable layers, but you can specify custom targets:
- **`conv_block_7`**: Final decoder layer (highest spatial resolution)
- **`conv_block_6`**: Mid-level decoder features
- **`conv_block_5`**: Higher-level semantic features

#### Troubleshooting Grad-CAM
- **No layers found**: Ensure your model has convolutional layers with appropriate names
- **Blank heatmaps**: Try different target layers or check model training quality
- **Memory issues**: Reduce batch size or number of parallel workers
- **Slow processing**: Use fewer target layers or reduce number of samples

## Architecture

| ![The block diagram of the Original UNETR model.](figures/unetr_architecture.png) |
| :--: |
| *The block diagram of the Original UNETR model.* |

## Dataset
The dataset contains 3064 pairs of MRI brain images and their respective binary mask indicating tumor.
<br/> <br/>
Download the dataset: [Brain Tumor Segmentation](https://www.kaggle.com/datasets/nikhilroxtomar/brain-tumor-segmentation)
<br/>
Original Dataset: [Brain Tumor Segmentation](https://figshare.com/articles/dataset/brain_tumor_dataset/1512427)

## Results
The sequence in the images below is `Input Image`, `Ground Truth` and `Prediction`. <br/> <br/>
| ![](results/2.png) |
| :--: |
| ![](results/6.png) |
| ![](results/19.png) |
| ![](results/21.png) |
| ![](results/68.png) |

## How to improve
- Train on more epochs.
- Increase the input image resolution.
- Apply data augmentation.


## Contact
For more follow me on:

- <a href="https://www.youtube.com/idiotdeveloper"> YouTube </a>
- <a href="https://facebook.com/idiotdeveloper"> Facebook </a>
- <a href="https://twitter.com/nikhilroxtomar"> Twitter </a>
- <a href="https://www.instagram.com/nikhilroxtomar"> Instagram </a>
- <a href="https://t.me/idiotdeveloper"> Telegram </a>
