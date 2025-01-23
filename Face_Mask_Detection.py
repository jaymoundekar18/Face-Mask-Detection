# Importing the required libraries

import numpy as np
import cv2
from keras.models import load_model
from PIL import Image
from facenet_pytorch import MTCNN

detector = MTCNN()

cam = cv2.VideoCapture(0)

model  = load_model("custom_face_mask_model.h5")   # Load the pretrained face mask detection model 

while True:
  ret, frame = cam.read()

  if ret:

    img = cv2.flip(frame,1)

    boxes, face_confidence = detector.detect(img)

    if boxes is not None:
      for i, box in enumerate(boxes):
        x1, y1, x2, y2 = [int(val) for val in box]

        cropped_face = img[y1:y2, x1:x2]

      face_resizes = cv2.resize(cropped_face, (128,128))
      face_normalized = face_resizes / 255.0
      face_input = np.expand_dims(face_normalized, axis=0)

      prediction = model.predict(face_input)
      label = "Mask Found" if prediction[0][0] > 0.5 else "No Mask "
      color = (0,0,255) if label == "Mask" else (0,255,0)

      cv2.rectangle(img, (x1,y1), (x2,y2), (0,255,0), 1)
      cv2.putText(img,label, (50, 50), cv2.FONT_HERSHEY_COMPLEX, 1, color, 2)
      cv2.imshow("Face Mask Detection",img)
      
  if cv2.waitKey(1) & 0xFF == ord('q'):
    break

cam.release()
cv2.destroyAllWindows()
  
      
