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

There are many existing approaches to this problem, which we hope to build off of. A [paper](https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0177544) titled *Classification of breast cancer histology images using Convolutional Neural Networks* uses a convolutional neural network to classify breast cancer. This paper uses breast biopsy images. They extracted features, mostly features about a single nucleus like its color, shape, and density. They performed binary classification (carcinoma and non-carcinoma) which had an accuracy of 83.3%, and additionally they classified in four classes (normal tissue, benign lesion, in situ carcinoma and invasive carcinoma) which had an accuracy of 77.8%. 

A [paper](https://academic.oup.com/jnci/article/111/9/916/5307077?login=true) titled *Stand-Alone Artificial Intelligence for Breast Cancer Detection in Mammography: Comparison With 101 Radiologists* compared the performance of radiologists to that of an AI system in detecting breast cancer. The AI system used deep learning convolutional networks, feature classifiers, and image analysis algorithms to detect calcification and soft tissue lesions. Based on the presence of these features, the likelihood of cancer was determined. They found that the AI system performed about as well as the radiologists, although they did mention that such a system requires further investigation.

Neural networks are also used in many other pathology images. A [paper](https://pubmed.ncbi.nlm.nih.gov/27563488/) titled *UDeep learning for digital pathology image analysis: A comprehensive tutorial with selected use cases* explores different cases where neural networks have been used to classify pathology images. They used an open source framework (Caffe) with a singular network architecture, and where able to achieve high accuracy. In their lymphoma classification network, for example, they had a classification accuracy of 0.97 across 374 images. 

Our network will differ from these previous approaches in a variety of ways. Unlike these papers, our network will classify the images directly instead of extracting features from the images which it uses to classify, instead classifying the images directly. Like these papers, we will use a convolutional neural network to perform the classification.

Some of the challenges we forsee are time limitations in our algorithm, as well as difficulties in the size of the images. We do not expect different error rates for different sub-groups in the data. We anticipate that the neural network will perform the same for all of the sub-groups. We hope our model will be able to achieve a high degree of accuracy, comparable to a human. It might be difficult for the general public to interpret the results. Thus, it will be important to make sure that only doctors and trained pathologists can access the results and interpret them in a correct manner. To protect individuals' privacy, we will make sure to anonymize all tissue images and delete all images after the completion of the project. 

**Methods:**

For our software, we plan on using PyTorch. PyTorch is an open source machine learning library and we will easily be able to learn how to use PyTorch. In particular, we will get a lot of our code from this [tutorial initially](https://pytorch.org/tutorials/beginner/blitz/cifar10_tutorial.html) and then modify it to achieve the best results. 

We plan on using [this dataset](https://web.inf.ufpr.br/vri/databases/breast-cancer-histopathological-database-breakhis/), which we found from one of our related works. This dataset has 8,000 images, of which 2,500 are benign and 5,500 are malignant. All of these images have dimensions of 700 x 460 pixels. Each image is a 3 channel RGB picture in PNG format. 

Once we have downloaded our dataset, we will use PyTorch's [ImageFolder](https://pytorch.org/vision/main/generated/torchvision.datasets.ImageFolder.html) class which prepares a dataset that is structured in folders. In order to use this class, we had to write a separate Python script that splits all our data into a test and a train folder. We have two classes in our dataset - benign and malignant.

Next, we use PyTorch's random_split() function, as well as the [DataLoader](https://pytorch.org/docs/stable/data.html#torch.utils.data.DataLoader) class. This enables us to randomly split the dataset into a training set and a validation set.

Then, we define a convolutional neural network using PyTorch's built in model that takes 3 channel images (3 initial inputs). Our neural network has 3 CNN blocks, where each block contains two convolution layers and one max-pooling layer. After applying the convolution, we use a flatten layer to convert the tensor from 3D to one-dimensional. 

Lastly, we will train the network and test the model using the validation set. 

**Discussion:**

*Expected Results*

We expect our results to be similar to those of the related papers we read. In particular, we will measure the AUC, which represents the area underneath the ROC curve. An ROC curve (receiver operating characteristic curve) is a graph showing the performance of a classification model at all classification thresholds. It usually plots two parameters - the false positive and true positive rate. If our classification model has a high accuracy, the ROC curve will have a larger area underneath the curve (AUC). The following picture contrasts a few ROC curves.

<p align="center">
<img src="ROC-curve.png" alt="roc" width="400"/>
</p>

We see that as the model improves, the area underneath the curve increases as well until we reach a perfect classifier that has an AUC of 1. 

The AUC provides an aggregate measure of performance across all possible classification thresholds. AUC represents the probability that a random positive example is positioned to the right of a random negative example. A model that has a 100% accuracy rate will have an AUC of 1 and similarly, a model that has a 0% accuracy rate will have an AUC of 0. We expect our AUC to be around 0.8, which is close to the AUC value of the other related papers. 

We will also measure the precision, recall, sensitivity, and accuracy of our model. Ideally, we will have high precision and accuracy, as this will signify that our model is able to predict the class type of any breast cancer tissue sample. 

*Interpretation and Evaluation of our Results:*

To evaluate our results, we can use a confusion matrix. This will tell us the accuracy, false positive rate, false negative rate, and recall. We can use these metrics to compare results with the other papers we looked at during the literature search. Using this comparison, we can then try to hypothesize about reasons for the difference, such as whether the size of the network was the dominant factor, whether it was the optimizer that was being used, or if there were other reasons for the difference. 

Additionally, we will aim to also split up the results by different factors if possible (such as age/race), and see if the results vary significantly between categories. This could tell us if, for example, our neural network was being trained to learn features relating to one age/race more than another.

*Ethics:*

There are a few ethical issues related to our project. As a patient's breast cancer tissue images are fairly personal and private information, it is important that all data used to train or test our model is immediately deleted after it is no longer needed. It is also important that patients agree to let their data be used by our model. 

**Results:**

**Reflection/Future Work:**



