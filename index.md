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

**Introduction Outline:**
1. Introductory paragraph: What is the problem and why is it relevant to the audience attending THIS CONFERENCE? Moreover, why is the problem hard, and what is your solution? 

2. Background paragraph: Elaborate on why the problem is hard, critically examining prior work, trying to tease out one or two central shortcomings that your solution overcomes.

3. Transition paragraph: What keen insight did you apply to overcome the shortcomings of other approaches? Structure this paragraph like a syllogism: Whereas P and P=>Q, infer Q.

4. Details paragraph: What technical challenges did you have to overcome and what kinds of validation did you perform?

5. Assessment paragraph: Assess your results and briefly state the broadly interesting conclusions that these results support. 

**Ethical Sweep**

General Questions:
1. Should we even be doing this?
2. What might be the accuracy of a simple non-ML alternative?
3. What processes will we use to handle appeals/mistakes?
4. How diverse is our team?

We think this is an important project that will help improve the accuracy of breast cancer detection from tissue images. 

Data Questions:
1. Is our data valid for its intended use?
2. What bias could be in our data? (All data contains bias.)
3. How could we minimize bias in our data and model?
4. How should we “audit” our code and data?

Impact Questions:
1. Do we expect different errors rates for different sub-groups in the data?

We do not expect different error rates for different sub-groups in the data. We anticipate that the neural network will perform the same for all of the sub-groups. 

2. What are likely misinterpretations of the results and what can be done to prevent those misinterpretations?

It might be difficult for the general public to interpret the results. Thus, it will be important to make sure that only doctors and trained pathologists can access the results and interpret them in a correct manner. 

3. How might we impinge individuals' privacy and/or anonymity?

To protect individuals' privacy, we will make sure to anonymize all tissue images and delete all images after the completion of the project. 