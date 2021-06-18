# Diplomado_AI_Grupo07

Covid19 is definitely one of the most importants events in the history of mankind and  many approaches and efforts have been tried to create effective ways to detect the presence or ausence of it in a patient.

This project try to create a simplified model  to  COVID detection using images of radiology of the chest in contrast with more complex models founded in literature limited by lack of data using in our case a larger dataset ~ 5000 images(Covid/Normal/Pneumonia). The model is based on *"transfer Learnig"* using some state of the art neural networks like: ResNet101, DenseNet121, VGG19. 

<img src="https://www.researchgate.net/profile/Tawsifur-Rahman/publication/343094700/figure/fig1/AS:915554529460225@1595296618682/Sample-X-ray-image-from-the-dataset-COVID-19-X-ray-image-A-normal-X-ray-image-B.ppm" width="1000" height="270">




### Important aspects:
1. The data could be called "standard" because images are usually taken with the patient in a frontal position. 
2. Images used to train the model were validated by an expert radiologist, but its important to comment how difficult is to check if a patient is infected or not just by  a radiology image. 
3. The *"contrast"* aspect in those images can be a key point cuz some of them look   a little bit blurry, so *"equalization"* techniques were considered.
4. Even when the dataset is larger, some techniques like *"Data Augmentation"* are not discarded. 

## Relevant Tools and Libraries.
<img src="https://img.shields.io/badge/-Python-brightgreen"> |  <img src="https://img.shields.io/badge/-OpenCV-brightgreen"> | <img src="https://img.shields.io/badge/-Tensorflow-orange"> |  <img src="https://img.shields.io/badge/-Github-informational"> | <img src="https://img.shields.io/badge/-Sklearn-critical"> 

### Conclusions.
1.  The state of the art networks applied performs very well reaching almost a 97% of accuracy, recall and precision in the classifiation of images between two classes(Covid/ non covid). The results were validated with ROC and AUC curves as usual in the use case.
2.  The multiclass problem was tried, but was particularly difficult for the models to detect differences between Pneumonia and Covid images.

### Future works.
1. Focus in the multiclass classification problem  trying to consider more complex techniques like GANS, and specialized neural networks like ChestNet to the transfer process.
2. Create CNN visualizations to have an important  *"explainability level"* of what areas of the images use the model to the covid dection.

### INTEGRANTES: 
- Katherine Baquero
- David Beltran
- Johan Rodriguez
- Sebastian Torres


