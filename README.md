# fire-Unet-NDWS
Using Unet, TransUnet, AttentionUnet, etc. for Wild Fire Spread Prediction

## Data
The original Next Day Wildfire Spread dataset can be downloaded from [here](https://www.kaggle.com/fantineh/next-day-wildfire-spread). 

The extended 2012-2023 dataset can be downloaded from Kaggle [here](https://www.kaggle.com/datasets/bronteli/next-day-wildfire-spread-north-america-2012-2023).

The fire size and fire spread speed subset of dataset can be downloaded from [here](https://www.kaggle.com/datasets/zhiminv/next-day-wildfire-spread-subset-of-dataset/data).

## Run Jupyter Notebook for AttentionUnet, ResNet50 and Unet
1. Use Google colab
```
    from google.colab import drive

    drive.mount('/content/drive’)

    !pip install tensorflow==2.11.0 keras==2.11.0 tensorflow-addons==0.18.0 vit-keras==0.1.0 tensorflow-hub
``` 
2. Use condo environment (GPU Type: A100,CUDA Module: 11.7, Conda Module: anaconda3/2021.05)
```

    !conda create -y -n myenv
  
    conda activate myenv
     
    !pip install tensorflow==2.11.0 keras==2.11.0 tensorflow-addons==0.18.0 vit-keras==0.1.0 tensorflow-hub
``` 

## Run Jupyter Notebook to test 6 features
1. Change  INPUT_FEATURES (e.g. INPUT_FEATURES = ['elevation', 'th', 'sph', 'pr', 'NDVI', 'PrevFireMask'])
2. Change num_in_channels from 12 to 6 in the dataset setup for train, val, and test
3. Change the input_shape from (32,32,12) to (32,32,6) in the model

## Run Jupyter Notebook to test subset base on fire size and fire spread speed

1. Change the test dataset path for corresponding .tfrecords subset(find example in Unet_test_subset_of_dataset)
2. Load h5 file and run test function

## References and Acknowledgements
Our code is based on the following links, we thank the authors for their excellent contributions.

[Attention UNET and its Implementation in TensorFlow](https://idiotdeveloper.com/attention-unet-and-its-implementation-in-tensorflow/)

[CNN_for_prediction_wildfire](https://www.kaggle.com/code/isyanbaevnagim/cnn-for-prediction-wildfire#Load-libraries)

[Image segmentation with a U-Net-like architecture](https://keras.io/examples/vision/oxford_pets_image_segmentation/)

[ResNet50 in TensorFlow using Keras API ](https://keras.io/api/applications/resnet/)

[Swin U-net with Focal Modulation (ASUFM)](https://github.com/bronteee/fire-asufm?tab=readme-ov-file)

[TransUNet](https://github.com/Beckschen/TransUNet)

[UNET Implementation in TensorFlow using Keras API](https://idiotdeveloper.com/unet-implementation-in-tensorflow-using-keras-api/)

