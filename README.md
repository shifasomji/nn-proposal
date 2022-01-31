Title: Accurate Detection of Breast Cancer

Project Description
I will focus on creating neural network (NNs) that is able to accurately detect breast cancer from tissue images. There are many publicly available datasets of tissue images available online. Each pixel in the tissue image will be classified as "nucleus", "boundary", or "not nucleus and not boundary". After this classification, the neural network will then create a black and white image of only the nuclei. 

After segmenting the nuclei, features need to be computed. I envision using features like the nuclei's shape, area, and perimter. Nuclei in tissue images of an invasive breast cancer lesion will be larger and more distorted than nuclei in normal tissue images. By computing these features, the neural network will be able to predict when a tissue image is normal or when it shows the presence of a breast cancer lesion. 

The main goal of this project is to create a tool that helps doctors during their diagnoses. 

longer-term goal is to use a NN to enable better navigation in complex environments for a robot with multiple modes of locomotion (i.e., wheeled and flight). The results from this work will feed into an on-going project where we are trying to "cross the reality gap" that exists between simulation and the real world.

Project Goals
Create a dataset for training a NN for navigation.
Explore methods for adding "noise" to the dataset.
Train a NN that is able to navigate any procedurally generated maze.