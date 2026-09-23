# Cat-Dog-Panda Classifier

A PyTorch image classification project using Transfer Learning with a pretrained ResNet18 model to classify images into three classes: Cat, Dog, and Panda.

## Features

* Transfer Learning using ResNet18
* PyTorch `ImageFolder`
* Image size: 224 × 224
* ImageNet normalization
* Data augmentation using flip, rotation, and crop
* Frozen ResNet18 backbone
* Custom classifier: `512 → 256 → 3`
* ReLU and Dropout (`p=0.5`)
* CrossEntropyLoss
* Adam optimizer
* Learning rate: `0.001`
* CUDA/GPU support
* Best validation checkpoint
* Confusion matrix and example predictions
* Streamlit interface for new image classification

## Dataset

The dataset contains 3,000 images.

| Class  | Train | Validation | Test |
| ------ | ----: | ---------: | ---: |
| Cats   |   700 |        150 |  150 |
| Dogs   |   700 |        150 |  150 |
| Pandas |   700 |        150 |  150 |
| Total  |  2100 |        450 |  450 |

Kaggle Dataset:
https://www.kaggle.com/datasets/gpiosenka/cats-dogs-pandas-images

## Model Architecture

```text
Pretrained ResNet18
        |
Frozen Backbone
        |
Linear(512 → 256)
        |
ReLU
        |
Dropout(0.5)
        |
Linear(256 → 3)
```

## Environment

* OS: Windows
* IDE: Visual Studio Code
* Conda Environment: `yolo-coco`
* PyTorch: `2.14.0+cu126`
* CUDA: `12.6`
* GPU: NVIDIA GeForce MX550

## Setup

```powershell
conda activate yolo-coco
pip install -r requirements.txt
```

## Training

```powershell
python train.py
```

## Prediction

```powershell
python predict.py
```

## Streamlit Application

```powershell
streamlit run app.py
```

## Output

<img width="1245" height="885" alt="image" src="https://github.com/user-attachments/assets/a363023d-f7da-4675-a7fd-652e5069ad80" />


<img width="1235" height="617" alt="image" src="https://github.com/user-attachments/assets/632011b6-d8c9-4d3e-81e2-f464e7c69b5b" />
## Project Structure

```text
Cat-dog-panda/
│
├── notebooks/
│   └── cat_dog_panda_transfer_learning.ipynb
├── models/
│   └── best_resnet18_cat_dog_panda.pth
├── screenshots/
│   └── 05_streamlit.png
├── app.py
├── train.py
├── predict.py
├── prepare_dataset.py
├── requirements.txt
└── README.md
```

## Results

Test Loss: To be updated after final training

Test Accuracy: To be updated after final training

All results and screenshots are generated from the actual local execution.

**Sajith Ahamed F**
B.Tech Artificial Intelligence & Machine Learning
Saveetha Engineering College, Chennai
