# Deep-Fake-Detection-using-InceptionResNetV2
It is a Deep-Fake identification system utilizing a pre-existing InceptionResNetV2 framework, incorporating extensive fine-tuning and image augmentation technique. The training phase involves a dataset comprising authentic and manipulated video frames, and the model's efficacy is assessed through accuracy metrics. 

There have recently been significant worries regarding the manipulation and possible misuse of visual content due to the rise in deepfake technology. The potential for highly convincing fake videos to be produced using sophisticated AI algorithms presents serious risks, such as the spread of false information and the fabrication of evidence for nefarious ends. This study proposes a model based on the InceptionResNetV2 architecture, which is well-known for its superior performance in image recognition tasks, in an effort to address the critical need for efficient deep-fake detection.
The suggested remedy entails a multi-step procedure that begins with careful video frame pre-processing. The frontal face detector in the dlib library is used to extract, crop, and resize facial regions into a standard 128x128-pixel format. Next, the InceptionResNetV2 architecture is used, and the pre-trained model is adjusted for the complex task of deepfake detection. Binary classification is made possible by a custom classifier that can discern between real and fake video frames. 
In this research endeavor, we employed transfer learning with the InceptionResNetV2 architecture to address a binary image classification problem distinguishing between real and fake images. The InceptionResNetV2 model was pre-trained on ImageNet, and its convolutional layers were fine-tuned for our specific task. This transfer learning approach allows the model to leverage the rich hierarchical features learned from a diverse dataset.

<img width="260" alt="image" src="https://github.com/ArishiGupta18/DeepFake-Detection-Using-InceptionResNetV2/assets/85198302/63bddc8d-e739-4d80-af2b-43516cafef49">

Fig1: Model Summary

Our model architecture included a Global Average Pooling layer and a Dense layer with two units and softmax activation for binary classification. The model was trained over ten epochs using the Adam optimizer with a learning rate of 1e-5. The training process was monitored using a validation set to assess the model's generalization performance.
The model is rigorously trained using a labeled dataset that includes both real and fake video frames. An Adam optimizer with a carefully selected learning rate is used to optimize parameters over multiple epochs.

![image](https://github.com/ArishiGupta18/DeepFake-Detection-Using-InceptionResNetV2/assets/85198302/154c59ce-2015-4854-91b2-bcbb702d73ec)
![image](https://github.com/ArishiGupta18/DeepFake-Detection-Using-InceptionResNetV2/assets/85198302/d88b0ae4-b62f-4b6d-a84a-e3e44b819db6)


Fig2: Accuracy vs epoch and Loss vs epoch graph with explicit validation data

![image](https://github.com/ArishiGupta18/DeepFake-Detection-Using-InceptionResNetV2/assets/85198302/530d9831-7a1c-45d6-9e0a-6a02d13a8b7e) ![image](https://github.com/ArishiGupta18/DeepFake-Detection-Using-InceptionResNetV2/assets/85198302/74827e33-eb6d-4f70-84f4-394a64cf2147)


Fig3: Accuracy vs epoch and Loss vs epoch graph with 15% validation data and early stopping


The evaluation of our model involved a comprehensive analysis. The training history, including accuracy and loss, was visualized over epochs to identify trends and potential issues like overfitting or underfitting. Additionally, a confusion matrix was employed to examine the model's performance in terms of true positives, false positives, false negatives, and true negatives. Classification metrics such as precision, recall, and F1-score were computed to provide a nuanced understanding of the model's behavior.


Fig4: Confusion Matrix

In interpreting the results, attention was given to the implications of false positives and false negatives in the context of our specific problem. The findings from this study contribute valuable insights into the application of transfer learning, specifically with the InceptionResNetV2 architecture, for binary image classification tasks. The nuances of model performance and behavior shed light on the effectiveness and limitations of the proposed approach, paving the way for further refinement and exploration in future research endeavors.
The model's performance evaluation, which includes metrics like accuracy and a thorough confusion matrix, is the central focus of the study. In addition to offering a reliable method for identifying altered visual content and averting potentially harmful applications, this analysis contributes to ongoing efforts to reduce the risks associated with deepfake technology by providing insightful information about the model's capacity to distinguish between real and fake content.
In Fig 2, validation data is provided explicitly using the validation_data parameter in the model.fit method. The verbose parameter in model.fit is not explicitly set, so it uses the default value (usually 1, which shows progress bars during training). In the second code snippet, verbose is explicitly set to 1 in model.fit, meaning it will show progress bars during training.
In Fig 3, a validation split is used instead of separate validation data. The validation_split parameter in the model.fit method is set to 0.15, meaning that 15% of the training data will be used for validation. And it also  includes early stopping using the EarlyStopping callback. This callback monitors the validation loss and stops training if the loss does not improve after a certain number of epochs (patience parameter). It also  uses the EarlyStopping callback, which is not present in Fig:2.

# How to run?
The dataset was downloaded from kaggle deepfake detection challenge: https://www.kaggle.com/c/deepfake-detection-challenge/data
Experimental steps-
1) run capture_img
2) run deepfake_detection_train
3) run model_play 
