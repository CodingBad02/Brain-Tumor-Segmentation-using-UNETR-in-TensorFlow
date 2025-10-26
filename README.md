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

### Testing the Model
```bash
python test.py
```
- Loads the trained model and generates predictions
- Results are saved in the `results/` directory
- Each result shows: Input Image → Ground Truth → Prediction

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
