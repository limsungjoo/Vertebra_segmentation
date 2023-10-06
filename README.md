Vertebra Segmentation
=====================
I have studied various methods to improve segmentation performance for Vertebra X-ray images.
The following models have been developed and the results of the models were compared.    
[Paper] "Deep Learning-based Spine Segmentation Technique Using the Center Point of the Spine and Modified U-Net" in the Journal of the Korean Society for Medical and Biological Engineering (Vol. 44, No. 2), April 2023.
[Paper] Submitted for a paper titled “General Vertebral Body Radiography Segmentation using Coordconv Layers and Center Points with U-Net” to the journal PLOS ONE.

* [Segmentation Requirements](#segmentation-requirements)
* [Center_Coordconv UNet](#center_coordconv-unet)
* [Centroid-UNet](#centroid-unet)
* [Coordconv UNet](#coordconv-unet)
* [TransUNet](#transunet)
* [U-Net](#unet)
* [Experiments](#experiments)
----------------------

## Segmentation Requirements
* pytorch: 1.10.2+cu113
* torchvision: 0.11.3+cu113
* CUDA version: 11.6
* Python: 3.8.8

## Center_Coordconv UNet
Overall structure is that Coordconv layers and centroids are combined with the input layer of the model based on U-Net.    
Dataset was 1755 Vertebra X-ray image (Lateral view) in Severance Hospital for segmentation model.    
The proposed network was evaluated by 176 spine images and yielded an average Dice score of 0.9408.      
This network and related paper will be submitted in September, 2022.

#### Model Architecture   
![image](https://user-images.githubusercontent.com/48985628/187608509-aad9af10-031e-4bb0-a575-77b6f3144bca.png)

#### Results
<img src="https://user-images.githubusercontent.com/48985628/187634962-8abf4d0e-ad12-4824-af75-d2c513fc611b.png" width="600" height="600"/>

## Centroid UNet
This network detects the centroids of the vertebrae for the localization of the vertebra using U-Net.    
Thus, the centroids which are extracted from the Centroid UNet are added to the input channel of the segmentation model(Center+Coordconv UNet).    

#### Results    
![image](https://user-images.githubusercontent.com/48985628/187630961-d99647b8-3fd3-4044-9297-a5c4675899cf.png)

## Coordconv UNet
Overall structure is that Coordconv layers are combined with the input layer of the model based on U-Net.     
Reference from the paper: [An intriguing failing of convolutional neural networks and the CoordConv solution](https://arxiv.org/abs/1807.03247)    
Reference from the paper : [CoordConv-Unet: Investigating CoordConv for Organ Segmentation](https://doi.org/10.1016/j.irbm.2021.03.002)     

## TransUNet
This network is a combination of vision transforemr and U-Net for Vertebra X-ray image segmentation.     
Reference from the paper: [TransUNet: Transformers Make Strong Encoders for Medical Image Segmentation](https://arxiv.org/pdf/2102.04306.pdf)

## UNet
U-Net, which is used as the base of the segmentation network, is an end-to-end convolutional network proposed for image segmentation in the biomedical field.      
Reference from the paper: [U-Net: Convolutional Networks for Biomedical Image Segmentation](https://arxiv.org/abs/1505.04597)    

#### Model Architecture         
<img src="https://user-images.githubusercontent.com/48985628/187627436-58fa0f6b-082d-468c-8782-0c6f8b398936.png" width="600" height="400"/>

## Experiments
Comparison of results for several segmentation networks    
|Networks|Dice Score|
|:--------:|:----:|
|Center+Coordconv UNet w/ DA|0.9514|
|Center+Coordconv UNet|0.9494|
|Coordconv UNet|0.9152|
|TransUNet|0.9243|
|UNet|0.9002|

