# Face Mask Detection using Deep and Transfer Learning

This repository contains a set of deep learning models to detect whether a person is wearing a face mask or not. The models are built using TensorFlow and Keras, creating a custom **CNN** model and leveraging various pre-trained architectures such as [**MobileNet**](https://keras.io/api/applications/mobilenet/), [**VGG16**](https://keras.io/api/applications/vgg/), and [**ResNet50**](https://keras.io/api/applications/resnet/). Each model has been customized and fine-tuned for binary classification **(Mask/No Mask)**.


## Models Implemented

### 1. Custom CNN Model

 - **Architecture**: A custom Convolutional Neural Network designed from scratch.

 - **Modifications**:
   - Four convolutional blocks, each with Conv2D, BatchNormalization, and MaxPooling2D layers.
   - Fully connected layers with ReLU activation, batch normalization, and dropout.
   - Output layer with softmax activation for binary classification.

- **Input Shape**: (128, 128, 3)

- **Custom Layers**:
   - Conv2D with filters (32, 64, 128, 256) for increasing complexity.
   - Dropout for regularization.

- **Accuracy**: 96.96%
<hr>

### 2. MobileNet-Based Model

- **Architecture**: MobileNet pre-trained on ImageNet.
  
- **Modifications**:
  
  - Added global average pooling.
  - Dense layers with dropout and batch normalization.
  - Output layer with sigmoid activation for binary classification.
    
- **Input Shape**: (128, 128, 3)

- **Accuracy**: 99.19%
<hr>

### 3. VGG16-Based Model

- **Architecture**: VGG16 pre-trained on ImageNet.
  
- **Modifications**:
  
  - Added global average pooling.
  - Multiple dense layers with dropout and ReLU activation.
  - Output layer with sigmoid activation for binary classification.
    
- **Input Shape**: (224, 224, 3).

- **Accuracy**: 97.15%
<hr>
  
### 4. ResNet50-Based Model

- **Architecture**: ResNet50 pre-trained on ImageNet.
  
- **Modifications**:
  
  - Added global average pooling.
  - Dense layers with dropout and ReLU activation.
  - Output layer with sigmoid activation for binary classification.
    
- **Input Shape**: (128, 128, 3).

- **Accuracy**: 75.64%
<hr>

## Dataset Structure
Ensure your dataset is organized as follows:
```
DATASET/
  Train/
    with_mask/
    without_mask/
  Val/
    with_mask/
    without_mask/
  Test/
    with_mask/
    without_mask/
```
- Download the dataset from [here](https://www.kaggle.com/datasets/omkargurav/face-mask-dataset)
- Run the ``Create_Dataset.py`` file to create a organized dataset for model creation.

<hr>

## Preprocessing Steps
- Images are rescaled to the range [0, 1].
- Data augmentation is applied to the training set:
  - Random rotations, zooming, shearing, and horizontal flipping.
<hr>

## Dependencies

- OpenCV (for some preprocessing steps)
- NumPy
- Sklearn
- TensorFlow
- Keras
- Facenet_Pytorch
  
Install the required packages using:
```bash
pip install -r requirements.txt
```
<hr>

## Training
To train any of the models, you can use the corresponding code provided in the notebook or scripts. Each model uses an early stopping mechanism to avoid overfitting.

### Example Command for Training
```python
history = model.fit(
    train_generator,
    steps_per_epoch=train_generator.samples // train_generator.batch_size,
    validation_data=val_generator,
    validation_steps=val_generator.samples // val_generator.batch_size,
    epochs=20,
    callbacks=[early_stopping]
)
```
<hr>

## Evaluation
Evaluate the models on the test set to measure their performance:
```python
loss, accuracy = model.evaluate(test_generator)
print(f"Test Accuracy: {accuracy * 100:.2f}%")
```
## Trained Weights
- You can download the trained weights from [here](https://www.dropbox.com/scl/fi/uwbqr8hsvuo0mzc2id4zy/models.zip?rlkey=2ak4xae8xhvgvgj6jf4t1ncp2&e=2&st=8nb9xidj&dl=0)

## For Face Mask Detection 
- Refer [this](Detection.md)
## Results
Each model is evaluated based on its accuracy and loss on the test set. Results may vary depending on the dataset size and quality.

## Contributions
Feel free to contribute to this project by adding new models or improving existing ones. Create a pull request with your proposed changes.

## License
This project is licensed under the [MIT License](LICENSE). See the LICENSE file for details.
