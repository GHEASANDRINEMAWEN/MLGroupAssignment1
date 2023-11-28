# Steel Defect Detection Using Deep Learning

## Overview
This project utilizes deep learning techniques to detect and classify defects in steel manufacturing. We employ a pre-trained ResNet50 model for feature extraction and fine-tune it to identify various types of defects in steel images.

## Dataset
- Training Images: [train_images](train_images/)
- Test Images: [test_images](test_images/)
- Annotations: [train.csv](train.csv)

## Model Architecture
- Base Model: ResNet50
- Custom Dense Layers for Classification

## Dependencies
- Python 3.x
- TensorFlow 2.x
- Keras
- Pandas
- NumPy
- OpenCV
- Matplotlib
- Seaborn
- Tqdm

## Installation
1. Clone the repository: git clone https://github.com/your-username/steel-defect-detection.git
cd steel-defect-detection
2. Install dependencies: pip install -r requirements.txt

  ## Usage
1. Run the Jupyter Notebook for model training and evaluation:
2. Follow the instructions in the notebook to preprocess data, train the model, and evaluate its performance.

## Training and Evaluation
- Preprocess images (resize and normalize).
- Train the model.
- Fine-tune the model by unfreezing top layers.
- Evaluate the model on the test dataset.

## Results
- View training and validation accuracy/loss curves in the notebook.
- Make predictions using the trained model.

## Contributing
Contributions, issues, and feature requests are welcome. Please check the for open tasks or create a new one.

##Link To Deployed Platform :  https://steeldetection-nbxhn2x64emnegsvuhontq.streamlit.app/
##Link to data used: https://drive.google.com/drive/folders/1Jdi_vA03N5mWVNExHBG4iE2pjHhnZ7Ji?usp=sharing

