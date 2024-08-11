# fire-Unet-NDWS
Using Unet, TransUnet, AttentionUnet, etc. for Wild Fire Spread Prediction

## Data
The original Next Day Wildfire Spread dataset can be downloaded from [here](https://www.kaggle.com/fantineh/next-day-wildfire-spread). 

The extended 2012-2023 dataset can be downloaded from Kaggle [here](https://www.kaggle.com/datasets/bronteli/next-day-wildfire-spread-north-america-2012-2023).

## Run Jupyter Notebook for AttentionUnet, ResNet50 and Unet
— Use Google colab
```
    1. Upload dataset to cloud drive
    
    2. Load data from drive 
    from google.colab import drive , drive.mount('/content/drive’)
   
    3. Install necessary packages
    !pip install tensorflow==2.11.0 keras==2.11.0 tensorflow-addons==0.18.0 vit-keras==0.1.0 tensorflow-hub
``` 
— Use condo environment (GPU Type: A100,CUDA Module: 11.7, Conda Module: anaconda3/2021.05)
```
    1. Setup env 
    !conda create -y -n myenv
  
    2. Active env
     conda activate myenv
    
    3. Install necessary packages
    !pip install tensorflow==2.11.0 keras==2.11.0 tensorflow-addons==0.18.0 vit-keras==0.1.0 tensorflow-hub
``` 


## References and Acknowledgements
Our code is based on the following repositories, we thank the authors for their excellent contributions.

[Attention UNET and its Implementation in TensorFlow](https://idiotdeveloper.com/attention-unet-and-its-implementation-in-tensorflow/)

[CNN_for_prediction_wildfire](https://www.kaggle.com/code/isyanbaevnagim/cnn-for-prediction-wildfire#Load-libraries)

[Image segmentation with a U-Net-like architecture](https://keras.io/examples/vision/oxford_pets_image_segmentation/)

[Swin U-net with Focal Modulation (ASUFM)](https://github.com/bronteee/fire-asufm?tab=readme-ov-file)

[TransUNet](https://github.com/Beckschen/TransUNet)

[UNET Implementation in TensorFlow using Keras API](https://idiotdeveloper.com/unet-implementation-in-tensorflow-using-keras-api/)

