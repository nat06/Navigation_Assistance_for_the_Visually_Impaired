# 360° Imagining for Navigation Assistance for the Visually Impaired

## Project Description
Over 1.5 million Canadians and around 300 million people worldwide suffer from vision impairment or blindness. Of these, about half require external assistance for common tasks in their daily lives. This project is about the development of an innovative navigation system that utilizes 360° imaging technology along with deep learning methods and smartphone sensors to provide meaningful quality-of-life improvements to the visually impaired community.  
The work described here, along with the code in this repo, focuses on the task of intersection crossing guidance, an essential part of the global navigation system. 

## Repo description
NOTE: for each sub-task/item, the corresponding code is given in the form of both separate functions (separate Python files) and a comprehensive Jupyter Notebook, structured sequentially. Code should be ran in an appropriate virtual environment set-up with the necessary dependencies.
  
This repository contains code for:
* Extracting images - using the Google Street View API - from the lat-long intersection coordinates dataset put together by processing open online city databases 
* Labeling these images and adding them to the Google Street View image dataset
* Deep Learning - comprehensive work (model initialization, training, validation, test)
* Model compilation for compatibility with the lab's Android app

Each task is covered below in more details. 

### Extracting intersection crossing images using the Google Street View API
toMention: need GSV API key, need to have downloaded ... etc
(Montreal, Ottawa)
need Open CV !!!
can modify the query parameters to mimic the intersection crossing action as desired (heading, size, field of view, pitch, radius...)

### Labeling images and adding them to the Google Street View image dataset
toMention: 7 vertical strips (+0 so 8 classes) 
image augmentation... based on symmetry
  
The virtual environment should be configurated with the following dependencies:
