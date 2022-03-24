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
