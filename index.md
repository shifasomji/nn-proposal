<!---
**Project Description:**

I will focus on creating neural network (NNs) that is able to accurately detect breast cancer from tissue images. There are many publicly available datasets of tissue images available online. Each pixel in the tissue image will be classified as "nucleus", "boundary", or "not nucleus and not boundary". After this classification, the neural network will then create a black and white image of only the nuclei. 

After segmenting the nuclei, features need to be computed. I envision using features like the nuclei's shape, area, and perimter. Nuclei in tissue images of an invasive breast cancer lesion will be larger and more distorted than nuclei in normal tissue images. By computing these features, the neural network will be able to predict when a tissue image is normal or when it shows the presence of a breast cancer lesion. 

Here is an example of a benign tissue image. 
<img src="benign.JPG" alt="benign" width="300"/>

Here is an example of an invasive tissue image.
<img src="invasive.JPG" alt="invasive" width="300"/>

The main goal of this project is to create a tool that helps doctors during their diagnoses. Doctors will be able to check their diagnoses of a patient's tissue sample. This tool will be extremely helpful in making sure that doctors do not misdiagnose a patient or even recommend treatment when no treatment is needed. 


**Project Goals:**
1. Create a neural network that segments each pixel into nucleus, boundary, or not nucleus and not boundary.
2. Compute various features.
3. Train the NN to be able to detect invasive vs. benign breast cancer lesions by looking at tissue images. 
-->

**Introduction:**

Classifying breast cancer tissue images as cancerous or benign is an essential part of breast cancer research and diagnosis. Often, this classification is done by hand, despite the fact that classifying by hand takes a lot of time and can be inaccurate. Additionally, cancer tissue images can look very different depending on the image, furthur complicating classification efforts. These factor led us to try and develop a neural network to classify breast cancer tissue images as benign or malignant, with the hope of achieving similar accuracy to human classification (We should see if there is a stat for how accurate humans are).

There are many existing approaches to this problem, which we hope to build off of. A [paper](https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0177544) titled *Classification of breast cancer histology images using Convolutional Neural Networks* uses a convolutional neural network to classify breast cancer. This paper uses breast biopsy images. They extracted features, mostly features about a single nucleus like its color, shape, and density. They performed binary classification (arcinoma and non-carcinoma) which had an accuracy of 83.3%, and additionally they classified in four classes (normal tissue, benign lesion, in situ carcinoma and invasive carcinoma) which had an accuracy of 77.8%. 

A [paper](https://academic.oup.com/jnci/article/111/9/916/5307077?login=true) titled *Stand-Alone Artificial Intelligence for Breast Cancer Detection in Mammography: Comparison With 101 Radiologists* compared the performance of radiologists to that of an AI system in detecting breast cancer. The AI system used deep learning convolutional networks, feature classifiers, and image analysis algorithms to detect calcification and soft tissue lesions. Based on the presence of these features, the likelihood of cancer was determined. They found that the AI system performed about as well as the radiologists, although they did mention that such a system requires further investigation.

Neural networks are also used in many other pathology images. A [paper](https://pubmed.ncbi.nlm.nih.gov/27563488/) titled *UDeep learning for digital pathology image analysis: A comprehensive tutorial with selected use cases* explores different cases where neural networks have been used to classify pathology images. They used an open source framework (Caffe) with a singular network architecture, and where able to achieve high accuracy. In their lymphoma classification network, for example, they had a classification accuracy of 0.97 across 374 images. 

Our network will differ from these previous approaches in a variety of ways. Unlike these papers, our network will classify the images directly instead of extracting features from the images which it uses to classify, instead classifying the images directly. Like these papers, we will use a convolutional neural network to perform the classification.

Some of the challenges we forsee are time limitations in our algorithm, as well as difficulties in the size of the images. We do not expect different error rates for different sub-groups in the data. We anticipate that the neural network will perform the same for all of the sub-groups. We hope our model will be able to achieve a high degree of accuracy, comparable to a human. It might be difficult for the general public to interpret the results. Thus, it will be important to make sure that only doctors and trained pathologists can access the results and interpret them in a correct manner. To protect individuals' privacy, we will make sure to anonymize all tissue images and delete all images after the completion of the project. 

**Methods:**

For our software, we plan on using PyTorch. PyTorch is an open source machine learning library and we will easily be able to learn how to use PyTorch. Speicially, we will get a lot of our code from this tutorial initially: https://pytorch.org/tutorials/beginner/blitz/cifar10_tutorial.html and then modify it to achieve the best results. 

We plan on using [this dataset](https://web.inf.ufpr.br/vri/databases/breast-cancer-histopathological-database-breakhis/), which we found from one of our related works. This dataset has 8,000 images, of which 2,500 are benign and 5,500 are malignant. All of these images have dimensions of 700 x 460 pixels. Each image is a 3 channel RGB picture in PNG format. 

Load the image data into a numpy array using the pillow python package. Convert the numpy array into a tensor. 

Normalize the tensors to be in the range /[-1,1/]. Create a training set and a validation set. 

Define a convolutional neural network using PyTorch's built in model that takes 3 channel images (3 initial inputs). Additionally, define a loss function. 

Train the network, and experiment with different epochs. Test the model using the validation set. 

