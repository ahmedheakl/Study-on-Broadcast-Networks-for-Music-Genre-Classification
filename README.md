# Study-on-Broadcast-Networks-for-Music-Genre-Classification
This repo contains the model details and implementation mentioned in our paper.

For each code file, we used python 3.9 and tensorflow 2.6.

Due to the increased demand for music streaming/recommender services and the recent developments of music information retrieval frameworks, Music Genre
Classification (MGC) has attracted the community’s attention. However, convolutional-based approaches are known to lack the ability to efficiently encode
and localize temporal features. In this paper, we study the broadcast-based neural networks aiming to improve the localization and generalizability under
a small set of parameters (about 180k) and investigate twelve variants of broadcast networks discussing the effect of block configuration, pooling method,
activation function, normalization mechanism, labelsmoothing, channel interdependency, LSTM block inclusion, and variants of inception schemes.
Our computational experiments using relevant datasets such as GTZAN, Extended Ballroom, HOMBURG, and Free Music Archive (FMA) show the stateof-the-art
classification accuracies in MGC. Our approach offers insights and the potential to enable compact and generalizable broadcast networks for music classification.

# Usage
First, you need to process your data and convert it to a spectrogram. We used librosa framework for this task. 
You will find the code implementation for the preprocessing in ```BBNN-Preprocessing-GTZAN-tf.ipynb``` file for GTZAN dataset and ```BBNN-Preprocessing-FMA-tf.ipynb``` for FMA dataset. 

To use our model's implementation, just download the implementation files and retrain the model on any of the following datasets:
. GTZAN
. FMA
. HOMBURG
. Extended Ballroom 

# Output 
According to your label encoding in the preprocessing stage, the model outputs a one hot vector with dimension `(num_classes, 1)` for each sample input.

# Model Architecture
The model is an improvement over Broadcast networks with lowered number of parameters an higher classification accuracy: 
![Model architecture](https://github.com/ahmedheakl/Study-on-Broadcast-Networks-for-Music-Genre-Classification/blob/main/Our-model.png)

