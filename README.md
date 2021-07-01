# CNN Multi-Class Classification Model to classify different Skin Cancer types

## **Problem Statement :** 
To build a CNN based model which can accurately detect Melanoma. Melanoma is a type of cancer that can be deadly if not detected early. It accounts for 75% of skin cancer deaths. A solution which can evaluate images and alert the dermatologists about the presence of melanoma has the potential to reduce a lot of manual effort needed in diagnosis.

For knowing more about Melanoma [click here](https://en.wikipedia.org/wiki/Melanoma).

 ![alt text]( https://www.mayoclinic.org/-/media/kcms/gbs/patient-consumer/images/2013/11/19/10/03/ds00575-melanoma-pictures-for-self-examination.jpg)

### **Python Libraries Used** 

Here is a list of Python Libraries used along with their version numbers :

 |Library                         |   Version                         |
|-------------------------------|-----------------------------|
|   `Tensorflow`           |    2.3.2           |
|`Numpy`            |1.18.1           |
|`matplotlib`|1.0.1|
|`seaborn `|0.10.0|

### **Input Datasets** 
The dataset consists of 2357 images of malignant and benign oncological diseases, which were formed from the International Skin Imaging Collaboration (ISIC). All images were sorted according to the classification taken with ISIC, and all subsets were divided into the same number of images, with the exception of melanomas and moles, whose images are slightly dominant.
The data set contains the following diseases:

1. Actinic keratosis
2. Basal cell carcinoma
3. Dermatofibroma
4. Melanoma
5. Nevus
6. Pigmented benign keratosis
7. Seborrheic keratosis
8. Squamous cell carcinoma
9. Vascular lesion

The image datasets were loaded using the glob function. There are 2239 images in the train directory and 118 images in the test directory. Within these folders there are sub-folders corresponding to each disease type.

### **Implementation in Python Notebook** 
#### *Dataset Creation*
Created train & validation dataset from the train directory with a batch size of 32 and image sizes of  to 180*180 using the Keras Pre-processing module. The training data was split into 80:20 ratio for the train: validation split.
#### *Dataset visualisation* 
Let's visualize one instance of all the nine classes present in the dataset .

![Melanoma Types](https://user-images.githubusercontent.com/10894854/124117092-566a8f00-da8d-11eb-9717-b6169e2fda79.JPG)

#### *Model Building & training :*
#### **MODEL 1: Creating a CNN model, which can accurately detect 9 classes present in the dataset.** 

The model consists of two convolution blocks with a max pool layer in each of them. Each of these layers is activated by a relu function. We flatten the output from the max pool layer before feeding them into the dense layers. There's a fully connected layer with 128 units on top of it that is activated by a softmax activation function. Softmax activation function gives the best results for multi- class classification and hence using the same. We will normalize the data in the model. Model is compiled using Adam Optimiser and Sparse Categorical Cross entropy Loss Function.

**Findings from the Model**

Here are the findings from the results :-
![model1](https://user-images.githubusercontent.com/10894854/124118897-7ac76b00-da8f-11eb-98df-02c40b83609e.JPG)

- The training accuracy steadily increases with each epoch and at the end of 20 epochs it peaks at ~88%
- The Training loss decreases steadily with each epoch
- The Validation accuracy initially increases but after around 5 epochs oscillates around +/- 50%
- The Validation Loss decreases initially but shows an upward trend after 5 epochs
- From this observations, we can clearly say that the model is over-fitting on the data.

To overcome this we will apply two strategies here :

1. Augmentation - Try Rotating/Zooming/Flipping the images and check if the over-fitting exists
2. Class Imbalance -- Address the class imbalance issue by adding additional images to each class.

#### **MODEL 2: Data Augmentation** 

We will try a few transformations in the dataset and run the model with this.

Here are the transformations :
1. Flip the image Horizontally
2. A slight random rotation of the image
3. Zoom the images slightly

We will use the same model as above. The only addition will be few dropout layers in between.
We are hoping that these dropout layers will help reduce the over-fitting in the model.

**Findings from the Model**

Here are the findings from the second model :
![model2](https://user-images.githubusercontent.com/10894854/124118929-83b83c80-da8f-11eb-8596-feb98d475beb.JPG)

- There is a significant drop in the Training error rate. Earlier the training rate touched ~90% but in this model the accuracy dropped to ~53%.
- The training loss is gradually decreasing and is lessor in absolute values than the first model.
- There is not much improvement in the validation accuracy compared to previous.
- The Validation losses are much lesser.

Hence, we can safely conclude that the problem of over-fitting has been completely mitigated here. But this has cost us a significant drop in the accuracy levels. We will address that in our third model.

**Plotting Class Distribution in the training dataset**

Many times real life datasets can have class imbalance, one class can have proportionately higher number of samples compared to the others. Class imbalance can have a detrimental effect on the final model quality. Hence as a sanity check it becomes important to check what is the distribution of classes in the data.

![plot1](https://user-images.githubusercontent.com/10894854/124118938-861a9680-da8f-11eb-96ad-667dad7e692c.JPG)

#### **MODEL 3: Rectifying the Class Imbalance** 
We will use a python package known as [Augmentor](https://augmentor.readthedocs.io/en/master/) to add more samples across all classes so that none of the classes have very few samples.

To use Augmentor, the following general procedure is followed:

1. Instantiate a Pipeline object pointing to a directory containing your initial image data set.
2. Define a number of operations to perform on this data set using your Pipeline object.
3. Execute these operations by calling the Pipeline’s sample() method.

In our case , we will add 500 samples per class to make sure that none of the classes are sparse.

So, now we have added 500 images to all the classes to maintain some class balance. We can add more images as we want to improve training process.

**Findings from the Model**

![model3](https://user-images.githubusercontent.com/10894854/124118936-85820000-da8f-11eb-873f-7eeaa744b483.JPG)

We reached a training accuracy of 90% and a validation accuracy of 80%. The Validation Accuracy has improved considerably and the gap between the training and validation accuracy has got bridged, and it stands at only 10% now.

