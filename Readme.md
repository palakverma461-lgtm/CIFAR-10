# CIFAR-10 Custom Image Classification & Streamlit Web App

A complete end-to-end Deep Learning project that extracts CIFAR-10 dataset into organized local folders, trains a custom Deep Convolutional Neural Network (CNN), and deploys it via an interactive Streamlit web application with confidence scoring.

---

## 📁 Project Structure

```text
CIFAR - 10/
│
├── dataset/               # Automatically generated folders for all 10 classes
│   ├── airplane/
│   ├── automobile/
│   ├── bird/
│   ├── cat/
│   ├── deer/
│   ├── dog/
│   ├── frog/
│   ├── horse/
│   ├── ship/
│   └── truck/
│
├── create_dataset.py      # Extracts images from CIFAR-10 into local class folders
├── train_folder.py        # Trains the CNN model using local image directories
├── app.py                 # Streamlit Web UI for drag-and-drop inference
├── requirements.txt       # Project dependencies
└── README.md              # Project documentation