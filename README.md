# Binary Digit Classification with TensorFlow

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue.svg)](https://www.python.org/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange.svg)](https://www.tensorflow.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

A clean, educational implementation of a **binary neural network classifier** for handwritten digits **0** and **1** using TensorFlow/Keras. The model achieves perfect accuracy on a filtered subset of the classic handwritten digits dataset.

![Sample model predictions on test digits](sample_predictions.png)

*Figure: Model predictions on randomly sampled test images. Each 20×20 grayscale thumbnail shows the true label, predicted label, and confidence score.*

---

## ✨ Features

- **End-to-end ML pipeline**: Data loading → Preprocessing → Modeling → Evaluation → Visualization
- **Binary classification** with sigmoid output and binary cross-entropy
- **High performance**: 100% accuracy, precision, and recall on both train and test sets
- **Comprehensive visualizations**: Training curves, confusion matrix, sample predictions
- **Educational structure**: Well-documented Jupyter notebook with step-by-step explanations
- **Extensible design**: Ready for multiclass expansion, CNNs, and deployment

---

## 📊 Dataset

### Overview
- **Source**: Filtered handwritten digits (originally from a larger dataset like MNIST-style)
- **Classes**: Only digits **0** and **1**
- **Samples**: 1,000 total
- **Features**: 400 (20×20 grayscale images flattened)
- **Labels**: Binary (0 or 1)

### Preprocessing
- Filtered dataset to binary classes
- Flattened images into feature vectors
- 80/20 train-test split for generalization evaluation
- Data quality verified via random sample visualization

**Data Shapes**:
- `X.npy`: `(1000, 400)`
- `y.npy`: `(1000, 1)`

---

## 🧠 Model Architecture

```python
model = Sequential([
    Dense(10, activation='relu', input_shape=(400,)),
    Dense(10, activation='relu'),
    Dense(1, activation='sigmoid')
])
```

| Layer            | Units | Activation | Role                          |
|------------------|-------|------------|-------------------------------|
| Input            | 400   | -          | Flattened pixel values        |
| Hidden Layer 1   | 10    | ReLU       | Feature extraction            |
| Hidden Layer 2   | 10    | ReLU       | Deeper pattern learning       |
| Output           | 1     | Sigmoid    | Binary probability (0-1)      |

**Design Choices**:
- **ReLU**: Efficient non-linearity, avoids vanishing gradients
- **Sigmoid**: Natural probability output for binary tasks
- **Adam optimizer** with learning rate 0.01 for fast convergence

---

## 🚀 Training & Evaluation

### Hyperparameters
- **Loss**: Binary Cross-Entropy
- **Optimizer**: Adam (lr=0.01)
- **Epochs**: 20
- **Metrics**: Accuracy, Precision, Recall

### Results

**Training Performance**
| Metric    | Score |
|-----------|-------|
| Accuracy  | 100%  |
| Precision | 100%  |
| Recall    | 100%  |

**Test Performance** (200 samples)
| Metric    | Score |
|-----------|-------|
| Accuracy  | 1.00  |
| Precision | 1.00  |
| Recall    | 1.00  |

**Confusion Matrix** (Test Set)

| Actual \ Predicted | 0   | 1   |
|--------------------|-----|-----|
| **0**              | 96  | 0   |
| **1**              | 0   | 104 |

- **Zero misclassifications**
- Rapid convergence with smooth loss/accuracy curves
- No signs of overfitting

---

## 📈 Visualizations

The notebook includes rich Matplotlib visualizations:
- Training & validation loss curves
- Accuracy, Precision, and Recall progression over epochs
- Sample predictions grid with confidence scores
- Confusion matrix heatmap

These plots demonstrate stable learning and perfect generalization.

---

## 🛠️ Technologies

- **Python** 3
- **TensorFlow / Keras** (Deep Learning)
- **NumPy** (Data handling)
- **Matplotlib** (Visualization)
- **scikit-learn** (Metrics & splitting)

---

## 📥 Installation & Usage

### Prerequisites
```bash
pip install tensorflow numpy matplotlib scikit-learn
```

### Steps
1. Clone the repository
2. Place `X.npy` and `y.npy` in the project root
3. Open `handwritten_digit_classification.ipynb`
4. Run all cells sequentially

**Expected Output**: Trained model, performance metrics, visualizations, and sample predictions.

---

## 📁 Project Structure
```
├── handwritten_digit_classification.ipynb  # Main notebook
├── X.npy                                   # Features
├── y.npy                                   # Labels
├── images/
│   └── sample_predictions.png              # Featured image
├── README.md
└── requirements.txt (optional)
```

---

## 🎯 Key Takeaways

- Simple dense networks can achieve **perfect performance** on linearly separable binary image tasks
- Importance of proper data preprocessing and visualization
- Strong baseline for more complex digit recognition projects
- Excellent starting point for understanding Keras workflows

---

## 🔮 Future Enhancements

1. **Multiclass Extension** – Classify all 0-9 digits
2. **CNN Architecture** – Leverage convolutional layers for better spatial understanding
3. **Data Augmentation** – Rotation, scaling, noise for robustness
4. **Hyperparameter Tuning** – Grid search / Optuna
5. **Regularization** – Dropout, L2, early stopping
6. **Deployment** – Flask API, Streamlit app, or TensorFlow Serving
7. **Explainability** – LIME, SHAP, or Grad-CAM visualizations
8. **Interactive Demo** – Gradio or Streamlit drawing canvas

---

## 📄 License

This project is open-sourced under the **MIT License** – feel free to use, modify, and learn from it.

## 👤 Author

Created with ❤️ for educational purposes.

---

**Happy Coding!** Feel free to ⭐ the repo if you found this helpful. Pull requests and issues are welcome!
**BVenkatManoj**

---

*Last Updated: 2026*
