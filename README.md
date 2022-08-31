Vertebra Segmentation
=====================
* [Segmentation Requirements](#segmentation-requirements)
* [Center+Coordconv U-Net](#center+coordconv-unet)
* [Centroid-UNet](#centroid-unet)
* [Coordconv U-Net](#coordconv-unet)
* [TransUNet](#Transunet)
* [U-Net](#unet)
* [Experiments](#experiments)

# Segmentation Requirements
* pytorch: 1.10.2+cu113
* torchvision: 0.11.3+cu113
* CUDA version: 11.6
* Python: 3.8.8

# Center+Coordconv UNet
Overall structure is that Coordconv layers and centroids are combined with the input layer of the model based on U-Net.    
Dataset was 1755 Vertebra X-ray image (Lateral view) in Severance Hospital for segmentation model.    
The proposed network was evaluated by 176 spine images and yielded an average Dice score of 0.9408.      
This network and related paper will be submitted in September, 2022.

Model Architecture   
![image](https://user-images.githubusercontent.com/48985628/187608509-aad9af10-031e-4bb0-a575-77b6f3144bca.png)

Results   
![image](https://user-images.githubusercontent.com/48985628/187622895-d23315ca-3dd1-43db-8ec3-60bdf70d6e26.png)

# Centroid UNet
This network detects the centroids of the vertebrae for the localization of the vertebra using U-Net.    
Thus, the centroids which are extracted from the Centroid UNet are added to the input channel of the segmentation model(Center+Coordconv UNet).    

Results    
![image](https://user-images.githubusercontent.com/48985628/187622462-5efc8433-5b42-488a-a17c-65977e567040.png)

# Coordconv UNet
Overall structure is that Coordconv layers are combined with the input layer of the model based on U-Net.     
Reference from the paper: [An intriguing failing of convolutional neural networks and the CoordConv solution](https://arxiv.org/abs/1807.03247)    
Reference from the paper : [CoordConv-Unet: Investigating CoordConv for Organ Segmentation](https://doi.org/10.1016/j.irbm.2021.03.002)     

# TransUNet
This network is a combination of vision transforemr and U-Net for Vertebra X-ray image segmentation. 
Reference from the paper: [TransUNet: Transformers Make Strong Encoders for Medical Image Segmentation](https://arxiv.org/pdf/2102.04306.pdf)

# UNet
U-Net, which is used as the base of the segmentation network, is an end-to-end convolutional network proposed for image segmentation in the biomedical field.
Reference from the paper: [U-Net: Convolutional Networks for Biomedical Image Segmentation](https://arxiv.org/abs/1505.04597)

U-Net Architecture    
![image](https://user-images.githubusercontent.com/48985628/187627436-58fa0f6b-082d-468c-8782-0c6f8b398936.png)

# Experiments
Comparison of results for several segmentation networks    
|Networks|Dice Score|
|--------|----|
|Center+Coordconv UNet|0.9408|
|Coordconv UNet|0.9362|
|TransUNet|0.9243|
|UNet|0.9117|

