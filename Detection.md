# Face Mask Detection Using OpenCV

This project uses a webcam to detect if a person is wearing a face mask or not, leveraging a pre-trained deep learning model and the [MTCNN (Multi-task Cascaded Convolutional Networks)](https://arxiv.org/pdf/1604.02878) for face detection.

## How It Works
1. The webcam captures real-time video frames.
2. The MTCNN model detects faces in each frame.
3. Detected faces are cropped and resized to match the input requirements of a custom-trained deep learning model for mask detection.
4. The custom model predicts whether the detected face is wearing a mask or not, and the result is displayed on the video feed.

## Prerequisites
- Python 3.8+
- Installed libraries:
  - OpenCV
  - facenet-pytorch
  - Keras
  - NumPy
  - PIL (Pillow)

Install the required packages using:
```bash
pip install -r requirements.txt
```

## Directory Structure
Ensure the custom model file is placed in the appropriate directory. Update the path to the model in the script if necessary.
```
Face-Mask-Detection/
  |-- Face_Mask_Detection.py
  |-- custom_models/
      |-- custom_face_mask_model.h5
      |-- mobilenet_face_mask_model.h5
      |-- resnet50_face_mask_model.h5
      |-- vgg16_face_mask_model.h5
```

## Usage

1. Place your trained model `.h5` file in the `custom_models` directory.
2. Update the path to the model in the script:
   ```python
   model = load_model(r"custom_models/your_trained_model.h5")
   ```
3. Run the script to start the webcam:
   ```bash
   python Face_Mask_Detection.py
   ```
4. The video feed will show detected faces with bounding boxes and a label indicating if a mask is found.
   - **"Mask Found"**: Indicates a mask is detected.
   - **"No Mask"**: Indicates no mask is detected.

## Key Code Components
- **Face Detection**: The MTCNN model detects faces in the video frame.
- **Preprocessing**: Detected faces are resized to `(128, 128)` and normalized.
- **Prediction**: The custom model predicts the mask status with a softmax activation output.
- **Visualization**: Bounding boxes and labels are drawn on the video feed using OpenCV.

## Example Output
- Bounding boxes around faces in the webcam feed.
- Real-time text label (e.g., "Mask Found" or "No Mask").

## Quit the Application
Press `q` to exit the webcam feed.

## Notes
- Ensure the MTCNN and Keras models are compatible with the input frame dimensions and preprocessing steps.
- Modify the threshold for mask detection if needed:
  ```python
  label = "Mask Found" if prediction[0][0] > 0.5 else "No Mask"
  ```
  Adjust the `0.5` value to suit your model's output distribution.

## License
This project is licensed under the [MIT License](LICENSE). See the LICENSE file for details.
