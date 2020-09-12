# 360° Imaging for Navigation Assistance for the Visually Impaired   

## Project Description 👁
Over 1.5 million Canadians and around 300 million people worldwide suffer from vision impairment or blindness. Of these, about half require external assistance for common tasks in their daily lives. This project is about the development of an innovative navigation system that utilizes 360° imaging technology along with deep learning methods and smartphone sensors to provide meaningful quality-of-life improvements to the visually impaired community.  
The work described here, along with the code in this repository, focuses on the task of __intersection crossing guidance__, an essential part of the global navigation system. 

<!-- ## Project supervisor
Professor Jeremy R. Cooperstock, Department of Electrical and Computer Engineering, Centre for Intelligent Machines, McGill University, jer@cim.mcgill.ca  
http://www.cim.mcgill.ca -->

## Credit
Initial code development for the intersetion-crossing app (Android_App/WSApplication) by Roger Girgis, currently a Ph.D. student at MILA

## Repo description 📝
NOTE: for each sub-task/item, the corresponding code is given in the form of both separate functions (separate Python files) and a comprehensive Jupyter Notebook structured sequentially. Code should be ran in an appropriate virtual environment set-up with the necessary dependencies (see  Deep Learning section).
  
This repository contains code for:
* __Extracting images__ - using the Google Street View API - from the lat-long intersection coordinates dataset put together by processing open online city databases 
* __Labelling__ these images and adding them to the Google Street View image dataset
* __Deep Learning__ - comprehensive work (model initialization, training, validation, testing)
* __Model compilation__ for compatibility with the lab's Android app

In addition, the Android Studio project code of the functioning app is accessible and sample images can be found.

Each task is covered below in more detail. 

### Extracting intersection crossing images using the Google Street View API
The Google Street View Static API is used in order to obtain ("scrape") several images from each intersection in our datasets. Two datasets are used: a small dataset containing intersection coordinates in the Ottawa area and a much larger dataset (65673 intersections) containing coordinates in the greater Montreal area.   
A few notes:
- User needs a valid Google Street View Static API access key
- Can modify the API optional request parameters (image size, heading, field of view, pitch, radius...) to best mimic the intersection crossing action desired
- Code is provided for reading a user-specified number of intersection lat-long coordinates and downloading the corresponding images (several images per intersection), given target API parameters

The code performing these tasks is contained in the __image-extraction-and-labelling__ folder in _image-extraction.py_. 
 

### Labelling images and adding them to the Google Street View image dataset
Images downloaded are processed and labelled. The processing consists of dividing the images into 7 vertical action spaces, making for an 8-label prediction problem (7 action spaces + the unknown action). By virtue of symmetry, the following data augmentation step is performed: each labelled image can be mirrored around its central vertical axis and the resulting flipped image can be automatically labelled with the corresponding inverse action (i.e., swapping left-to-right with right-to-left).   
A couple notes:
- __Open CV__ is necessary for image processing and labelling
- Code is provided for labelling images and merging the resulting data into the existing Google Street View image dataset  

The code performing these tasks is contained in the __image-extraction-and-labelling__ folder in _image-labelling.py_

The full code for image extraction __and__ labelling can also be found in _comprehensive-Notebook.ipynb_.


### Deep Learning
Transfer Learning is performed using several pretrained models (SqueezeNet, DenseNet, ResNet, VGG, MobileNet...) whose architectures are modified to fit our needs. Feature extraction or fine-tuning can be performed depending on the task at hand. Code for Deep Learning in both PyTorch and Tensorflow (version 2 __and__ version 1) is available.   
In order for the Deep Learning model to be successfully ported to the current Android app, the following should be observed:
- Deep Learning models must be obtained from Tensorflow
- (Re)trained models must be compatible with/obtained from Tensorflow versions ≤ 1.13 
- The Frozen graph of the model must be generated and used
