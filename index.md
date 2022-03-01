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

Classifying breast cancer tissue images by hand takes a lot of time and can be inaccurate. Cancer tissue images can look very different depending on the image, so hopefully our neural network will be able to classify these differences and identify central characteristics indicative of cancer. 

Many previous attempts only used a few layers, so we hope to improve on their work by using more layers for our neural network. Currently, a manual alternative is being used, which may be more accurate, but we hope that we can train our network to be as accurate as manually labeling images. Due to variations in images, it would be difficult to create a program to classify the images without using machine learning, since there are so many different cases.

Some of the challenges we forsee are time limitations in our algorithm, as well as difficulties in the size of the images. We do not expect different error rates for different sub-groups in the data. We anticipate that the neural network will perform the same for all of the sub-groups. We hope our model will be able to achieve a high degree of accuracy, comparable to a human.

It might be difficult for the general public to interpret the results. Thus, it will be important to make sure that only doctors and trained pathologists can access the results and interpret them in a correct manner. To protect individuals' privacy, we will make sure to anonymize all tissue images and delete all images after the completion of the project. 

There are many existing approaches to this problem. A [paper](https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0177544) titled *Classification of breast cancer histology images using Convolutional Neural Networks* uses a convolutional neural network to classify breast cancer. This paper, in contrast to the previous one, uses actual breast biopsy images instead of ultrasound images. They extracted features, mostly features about a single nucleus like its color, shape, and density. This model only had an 78% accuracy rate. 

A [blog](http://andrewjanowczyk.com/use-case-1-nuclei-segmentation/) titled *Use Case 1: Nuclei Segmentation* walks through the process of using deep learning for nuceli segmentation on H&E stained estrogen receptor positive (ER+) breast cancer images. They used Matlab to create patches for the images, bash to create a database and train the network, and then python was used to generate output images showing the probability each pixel is a nuceli. The code and dataset is provided, and they were able to achieve similar results as hand segmentation. 

A [paper](https://academic.oup.com/jnci/article/111/9/916/5307077?login=true) titled *Stand-Alone Artificial Intelligence for Breast Cancer Detection in Mammography: Comparison With 101 Radiologists* compared the performance of radiologists to that of an AI system in detecting breast cancer. The AI system used deep learning convolutional networks, feature classifiers, and image analysis algorithms to detect calcification and soft tissue lesions. They found that the AI system performed about as well as the radiologists, although they did mention that such a system requires further investigation.

We hope to use the insights and results from these papers in our own project. 

