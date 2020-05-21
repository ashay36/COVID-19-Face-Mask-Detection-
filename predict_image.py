from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.models import load_model
import numpy as np
import cv2
import os

CONFIG = {"face" : "/face detector",
          "model" : "mask_detector_model",
      	  "image" : "/images",
	"confidence" : 0.5}

prototxtPath = os.path.sep.join([CONFIG["face"], "deploy.prototxt"])
weightsPath = os.path.sep.join([CONFIG["face"], "res10_300x300_ssd_iter_140000.caffemodel"])
net = cv2.dnn.readNet(prototxtPath, weightsPath)

model = load_model(CONFIG["model"])

imgPath = os.path.sep.join([CONFIG["image"], "input.jfif"])
img = cv2.imread(imgPath)
orig = img.copy()
(h, w) = img.shape[:2]

blob = cv2.dnn.blobFromImage(img, 1.0, (300, 300), (104.0, 177.0, 123.0))
net.setInput(blob)
detections = net.forward()

for i in range(0, detections.shape[2]):
	confidence = detections[0, 0, i, 2]
	if confidence > CONFIG["confidence"]:
		box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
		(x1, y1, x2, y2) = box.astype("int")
		
		(x1, x2) = (max(0, x1), max(0, x2))
		(y1, y2) = (min(w - 1, y1), min(h - 1, y2))
		
		face = img[y1 : y2, x1 : x2]
		face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
		
		face = cv2.resize(face, (224, 224))
		face = img_to_array(face)
		face = preprocess_input(face)
		face = np.expand_dims(face, axis = 0)
		
		pred = model.predict(face)[0]
		
		label = "Mask On" if pred[0] > pred[1] else "Without Mask"
		color = (0, 255, 0) if label == "Mask On" else (0, 0, 0)
		
		label = "{} : {:.2f}%".format(label, max(pred) * 100)
		
		cv2.putText(img, label, (x1 - 10, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
		cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)

outPath = os.path.sep.join([CONFIG["image"], "output.jpg"])
cv2.imwrite(outPath, img)

cv2.imshow("Output", img)

cv2.waitKey(0)	
cv2.destroyAllWindows()

 




