# Binary Digit Classification Using TensorFlow Neural Network

## Overview

This project implements a binary classification neural network using TensorFlow/Keras to recognize handwritten digits **0** and **1**. The model is trained on preprocessed image data and demonstrates excellent performance, achieving perfect classification accuracy on both training and test datasets.

The project covers the complete machine learning workflow, including data preprocessing, neural network design, model training, evaluation, visualization, and prediction.

## Dataset

The dataset consists of:

* **X.npy** – Flattened handwritten digit images.
* **y.npy** – Corresponding digit labels.

### Data Preprocessing

* Filtered the dataset to include only digits **0** and **1**.
* Generated a binary classification problem.
* Input shape: **(1000, 400)**

  * 1000 samples
  * 400 features per sample (20×20 grayscale image flattened into a vector)
* Label shape: **(1000, 1)**

A random sample of images was visualized to verify data quality and label correctness.

---

## Neural Network Architecture

The model is built using TensorFlow's Sequential API.

| Layer         | Units        | Activation |
| ------------- | ------------ | ---------- |
| Input Layer   | 400 Features | -          |
| Dense Layer 1 | 10           | ReLU       |
| Dense Layer 2 | 10           | Sigmoid    |
| Output Layer  | 1            | Sigmoid    |

### Architecture Summary

* **ReLU Activation**

  * Introduces non-linearity.
  * Helps prevent vanishing gradients.

* **Sigmoid Activation**

  * Produces outputs between 0 and 1.
  * Suitable for binary classification.

---

## Training Configuration

### Loss Function

Binary Cross-Entropy

```python
tf.keras.losses.BinaryCrossentropy()
```

### Optimizer

Adam Optimizer

```python
tf.keras.optimizers.Adam(0.01)
```

### Evaluation Metrics

* Accuracy
* Precision
* Recall

### Training

* Epochs: **20**
* Training performed using the entire filtered dataset.
* Training history recorded for visualization and analysis.

---

## Results

### Training Performance

The model rapidly converged during training and achieved:

* Accuracy: **100%**
* Precision: **100%**
* Recall: **100%**

The loss decreased to an extremely small value, indicating successful learning.

### Test Performance

An 80-20 train-test split was used to evaluate generalization.

| Metric    | Score |
| --------- | ----- |
| Accuracy  | 1.00  |
| Precision | 1.00  |
| Recall    | 1.00  |

The identical performance on both training and test sets suggests excellent generalization with no observable overfitting.

---

## Confusion Matrix

The confusion matrix produced the following results:

| Actual / Predicted | 0  | 1   |
| ------------------ | -- | --- |
| 0                  | 96 | 0   |
| 1                  | 0  | 104 |

Results indicate:

* 96 True Negatives
* 104 True Positives
* 0 False Positives
* 0 False Negatives

This confirms perfect classification performance on the test set.

---

## Visualizations

The project includes plots for:

* Training Loss
* Accuracy
* Precision
* Recall

These visualizations demonstrate:

* Fast convergence
* Stable learning
* Consistent high performance throughout training

---

## Predictions

The trained model was tested on individual samples and successfully predicted both digit classes:

* Digit **0** → Correctly classified
* Digit **1** → Correctly classified

---

## Technologies Used

* Python
* TensorFlow / Keras
* NumPy
* Matplotlib
* Scikit-learn

---

## Key Takeaways

✅ Binary classification using neural networks

✅ TensorFlow/Keras model development

✅ Data preprocessing and visualization

✅ Performance evaluation with multiple metrics

✅ Confusion matrix analysis

✅ Overfitting assessment using train-test split

---

## Future Work

While the current model achieves perfect performance for binary classification of digits **0** and **1**, several enhancements can be explored to make the project more robust and applicable to real-world handwritten digit recognition tasks.

### 1. Extend to Multiclass Classification

Expand the model to classify all digits from **0–9** instead of only two classes. This would transform the problem into a multiclass classification task and provide a more comprehensive digit recognition system.

### 2. Experiment with Deeper Neural Networks

Investigate the impact of additional hidden layers and neurons on model performance. Deeper architectures may learn more complex patterns and feature representations.

### 3. Implement Convolutional Neural Networks (CNNs)

Replace the fully connected network with a CNN architecture. CNNs are specifically designed for image data and typically achieve superior performance in image classification tasks.

### 4. Hyperparameter Optimization

Perform systematic tuning of:

* Learning rate
* Number of hidden units
* Number of layers
* Batch size
* Activation functions

to identify the optimal model configuration.

### 5. Apply Regularization Techniques

Introduce techniques such as:

* Dropout
* L1/L2 Regularization
* Early Stopping

to improve generalization when working with larger and more challenging datasets.

### 6. Data Augmentation

Generate additional training samples through image transformations such as:

* Rotation
* Scaling
* Translation
* Noise injection

to improve model robustness against variations in handwriting styles.

### 7. Real-Time Digit Recognition

Develop a user-friendly application that allows users to draw digits and receive real-time predictions from the trained model.

### 8. Model Deployment

Deploy the trained model using:

* TensorFlow Serving
* Flask/FastAPI
* Streamlit
* Cloud platforms

to make the digit classifier accessible through a web interface or API.

### 9. Performance Comparison with Classical Machine Learning Models

Compare the neural network's performance with traditional machine learning algorithms such as:

* Logistic Regression
* Support Vector Machines (SVM)
* Random Forests

to evaluate the advantages and limitations of each approach.

### 10. Explainable AI (XAI)

Integrate interpretability techniques to visualize which pixels contribute most to the model's predictions, improving transparency and understanding of neural network decisions.


## Conclusion

This project demonstrates the successful implementation of a neural network for handwritten digit classification. By focusing on digits **0** and **1**, the model achieved perfect accuracy, precision, and recall on both training and test datasets. The results highlight the effectiveness of neural networks in solving simple image classification problems and provide a strong foundation for extending the approach to multiclass digit recognition tasks.
