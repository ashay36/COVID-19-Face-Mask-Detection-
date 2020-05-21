# COVID-19-Face-Mask-Detection-
A deep learning based model that detects whether a person has a face mask on or not with nearly 100% accuracy.

```
Author : Ashay Ajbani

Pretrained model used : MobileNetV2

Trainable parameters : 3M

Number of classes : 2

Train Accuracy : 99.29%

Validation Accuracy : 100%
```

# Dependencies
<ul>
  <li> <a href="https://www.tensorflow.org/">Tensorflow</a> </li>
  <li> <a href="https://www.keras.io/">Keras</a> </li>
  <li> <a href="https://www.opencv.org/">OpenCV</a> </li>
  <li> <a href="https://www.numpy.org/">NumPy</a> </li>
</ul> 

# Overview
<ul>
  <li>For training refer <i><b>mask_detection.ipynb</b></i></li>
  <li>For testing on your own image run <i><b>detect_facemask.py</b></i></li>
</ul> 

<b> Techniques used to reduce overfitting </b>
<ul>
  <li> Data Augmentation </li>
  <li> BatchNormalization </li>
  <li> Dropout </li>
</ul> 
