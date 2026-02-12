```python
Final Exam - AI ML
```


```python
Mikheil Davidovi
```


```python
Task 1: CNN
```


```python
# Convolutional Neural Networks (CNNs)
```


```python
## Overview
```


```python
Convolutional Neural Networks (CNNs) are a specialized type of deep learning architecture designed primarily for processing
grid-like data structures, such as images. CNNs have revolutionized computer vision and have found extensive applications in 
cybersecurity, particularly in malware detection, network intrusion detection, and visual content analysis.
```


```python
## Architecture Components
```


```python
### 1. Convolutional Layers
```


```python
The convolutional layer is the core building block of a CNN. It applies a set of learnable filters (kernels) to the input 
data. Each filter slides across the input, performing element-wise multiplication and summation to produce feature maps. 
These filters automatically learn to detect patterns such as edges, textures, and more complex features in deeper layers.
```


```python
### 2. Activation Functions
```


```python
After convolution, an activation function (typically ReLU - Rectified Linear Unit) is applied to introduce non-linearity
into the model. ReLU replaces all negative values with zero, enabling the network to learn complex patterns.
```


```python
### 3. Pooling Layers
```


```python
Pooling layers reduce the spatial dimensions of feature maps, decreasing computational complexity and helping to prevent 
overfitting. Max pooling selects the maximum value from each patch, while average pooling computes the average.
```


```python
### 4. Fully Connected Layers
```


```python
After several convolutional and pooling layers, the feature maps are flattened and passed through fully connected layers, 
which perform the final classification or regression task.
```


```python
### 5. Output Layer
```


```python
The final layer produces the network's predictions, using softmax activation for multi-class classification or sigmoid for 
binary classification.
```


```python
## CNN Architecture Visualization
```


```python
Input Image (28x28x1)
       ↓
Conv Layer 1 (32 filters, 3x3) → ReLU → MaxPool (2x2)
       ↓
Conv Layer 2 (64 filters, 3x3) → ReLU → MaxPool (2x2)
       ↓
Flatten
       ↓
Fully Connected (128 neurons) → ReLU → Dropout
       ↓
Output Layer (Softmax)
```


```python
## Application in Cybersecurity: Malware Detection
```


```python
CNNs can be used to detect malware by converting executable files into grayscale images and classifying them as malicious or 
benign. This approach leverages the visual patterns in binary data.
```


```python
### Python Implementation
```


```python
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.datasets import mnist
from tensorflow.keras.utils import to_categorical

# Load and preprocess data
(X_train, y_train), (X_test, y_test) = mnist.load_data()

X_train = X_train.reshape(-1, 28, 28, 1).astype("float32") / 255.0
X_test = X_test.reshape(-1, 28, 28, 1).astype("float32") / 255.0

y_train = to_categorical(y_train, 10)
y_test = to_categorical(y_test, 10)

# Build CNN model
model = Sequential([
    Conv2D(32, (3, 3), activation="relu", input_shape=(28, 28, 1)),
    MaxPooling2D((2, 2)),

    Conv2D(64, (3, 3), activation="relu"),
    MaxPooling2D((2, 2)),

    Flatten(),
    Dense(128, activation="relu"),
    Dropout(0.5),
    Dense(10, activation="softmax")
])

model.compile(
    optimizer="adam",
    loss="categorical_crossentropy",
    metrics=["accuracy"]
)

model.summary()

# Train model
history = model.fit(
    X_train,
    y_train,
    batch_size=128,
    epochs=10,
    validation_split=0.2,
    verbose=1
)

# Evaluate model
test_loss, test_accuracy = model.evaluate(X_test, y_test, verbose=0)
print(f"\nTest Accuracy: {test_accuracy:.4f}")
print(f"Test Loss: {test_loss:.4f}")

# Plot training history
plt.figure(figsize=(12, 4))

plt.subplot(1, 2, 1)
plt.plot(history.history["accuracy"], label="Training Accuracy")
plt.plot(history.history["val_accuracy"], label="Validation Accuracy")
plt.title("Model Accuracy")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.legend()
plt.grid(True)

plt.subplot(1, 2, 2)
plt.plot(history.history["loss"], label="Training Loss")
plt.plot(history.history["val_loss"], label="Validation Loss")
plt.title("Model Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.savefig("cnn_training_history.png", dpi=300, bbox_inches="tight")
plt.show()

# Feature map visualization
def visualize_feature_maps(model, image, layer_idx=0):
    # Force model build
    _ = model.predict(image[np.newaxis, ...])

    activation_model = Model(
        inputs=model.inputs,
        outputs=model.layers[layer_idx].output
    )

    activations = activation_model.predict(image[np.newaxis, ...])

    n_features = activations.shape[-1]
    n_cols = 8
    n_rows = n_features // n_cols

    plt.figure(figsize=(n_cols * 1.5, n_rows * 1.5))

    for i in range(n_features):
        plt.subplot(n_rows, n_cols, i + 1)
        plt.imshow(activations[0, :, :, i], cmap="viridis")
        plt.axis("off")

    plt.suptitle(f"Feature Maps – Layer {layer_idx}")
    plt.tight_layout()
    plt.savefig("feature_maps.png", dpi=300, bbox_inches="tight")
    plt.show()

# Grad-CAM implementation
def grad_cam(model, image, conv_layer_name):
    # Force model build
    _ = model.predict(image[np.newaxis, ...])

    grad_model = Model(
        inputs=model.inputs,
        outputs=[
            model.get_layer(conv_layer_name).output,
            model.outputs[0]
        ]
    )

    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(image[np.newaxis, ...])
        class_idx = tf.argmax(predictions[0])
        loss = predictions[:, class_idx]

    grads = tape.gradient(loss, conv_outputs)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

    conv_outputs = conv_outputs[0]
    heatmap = tf.reduce_sum(conv_outputs * pooled_grads, axis=-1)

    heatmap = tf.maximum(heatmap, 0)
    heatmap /= tf.reduce_max(heatmap) + 1e-10

    return heatmap.numpy()

def overlay_gradcam(image, heatmap, alpha=0.4):
    image = image.squeeze()

    heatmap = tf.image.resize(
        heatmap[..., np.newaxis],
        (image.shape[0], image.shape[1])
    ).numpy().squeeze()

    plt.imshow(image, cmap="gray")
    plt.imshow(heatmap, cmap="jet", alpha=alpha)
    plt.axis("off")
    plt.title("Grad-CAM Visualization")
    plt.show()

# Run visualizations
sample_image = X_test[0]

# Feature maps (first Conv layer)
visualize_feature_maps(model, sample_image, layer_idx=0)

# Grad-CAM (second Conv layer)
heatmap = grad_cam(model, sample_image, conv_layer_name="conv2d_1")
overlay_gradcam(sample_image, heatmap)

# Cybersecurity-specific use case
def exe_to_image(exe_path, width=256):
    """Convert executable file to grayscale image"""
    with open(exe_path, "rb") as f:
        exe_bytes = np.frombuffer(f.read(), dtype=np.uint8)

    size = width * width
    if len(exe_bytes) < size:
        exe_bytes = np.pad(exe_bytes, (0, size - len(exe_bytes)))
    else:
        exe_bytes = exe_bytes[:size]

    return exe_bytes.reshape(width, width)
```


```python
## Advantages in Cybersecurity
```


```python
1. Automatic Feature Extraction: CNNs learn relevant features automatically, reducing manual feature engineering
2. Spatial Hierarchy: Captures patterns at multiple scales
3. Resilience to Variations: Effective even with obfuscated or polymorphic malware
4. Visualization Capability: Feature maps help understand what the model has learned
```


```python
## Conclusion
```


```python
CNNs have proven highly effective in cybersecurity applications, particularly for image-based malware detection, network 
traffic visualization analysis, and anomaly detection in visual security data. Their ability to automatically learn 
hierarchical features makes them powerful tools for identifying complex attack patterns.
```
