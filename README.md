# 🖼️ Image Captioning with ResNet50 + Transformer Decoder

## 📌 Project Overview

This project implements an **Image Captioning system** using Deep Learning. The goal is to generate natural language descriptions for input images by combining:

- A **Convolutional Neural Network (CNN)** for image feature extraction (ResNet50)
- A **Transformer-based Decoder** for text generation

The model learns the relationship between visual features and textual descriptions using the Flickr8k dataset.

---

## 📊 Dataset

The project uses the **Flickr8k dataset**, which contains:
- ~8,000 images
- 5 captions per image

Each caption is preprocessed and wrapped with special tokens:
- `<start>` → beginning of sentence
- `<end>` → end of sentence

---

## ⚙️ Preprocessing Steps

### 📝 Text Processing
- Lowercasing all captions
- Removing special characters and numbers
- Adding `<start>` and `<end>` tokens
- Tokenization using Keras `Tokenizer`
- Padding sequences to fixed length (max length = 40)

### 🖼️ Image Processing
- Resizing images to `224 × 224`
- Normalization using ResNet50 preprocessing
- Feature extraction using pretrained ResNet50 (ImageNet weights)
- Extracted features stored as 2048-dimensional vectors

---

## 🧠 Model Architecture

The model consists of two main components:

### 1. Image Encoder
- Pretrained **ResNet50 (without top layer)**
- Output: 2048-dimensional feature vector
- Features are stored for faster training (`features.pkl`)

### 2. Text Decoder (Transformer-based)
- Token Embedding layer
- Positional Encoding
- 3 stacked Transformer Decoder blocks:
  - Self-Attention (causal mask)
  - Cross-Attention (image + text)
  - Feed Forward Network
- Final Dense layer with Softmax activation

---

## 🔁 Training Pipeline

- Loss Function: `Sparse Categorical Crossentropy`
- Optimizer: `Adam (lr = 3e-4)`
- Batch Size: 32
- Training-Validation split: 80/20
- Data Generator used for memory efficiency

### 🛑 Regularization Techniques:
- Dropout
- EarlyStopping
- ModelCheckpoint

---

## 🔍 Caption Generation

For inference, **Beam Search (beam size = 3)** is used instead of greedy decoding to improve sentence quality.

---

## 📏 Evaluation

Model performance is evaluated using:

### BLEU Score
- Corpus-level BLEU score
- Measures similarity between predicted and ground truth captions
- Smoothing function used to handle short sentences

### Qualitative Evaluation
- Random image predictions after each epoch
- Visual comparison between:
  - Ground truth captions
  - Predicted captions

---

## 📁 Saved Files

- `features.pkl` → extracted image features
- `tokenizer.pkl` → fitted tokenizer
- `model.keras` → trained captioning model

---

## 🚀 How to Run

### 1. Install dependencies
pip install tensorflow numpy pandas nltk tqdm
# Run 
jupyter notebook main.ipynb
