### IMPORTING NECESSARY LIBRARIES ###
from __future__ import absolute_import, division, print_function
import tensorflow as tf
print("Tensorflow version: ", tf.__version__)
import sys
import shutil
import os
import csv
import google_streetview.api
import google_streetview.helpers
import cv2
print("OpenCV version: ", cv2. __version__)
import numpy as np
import glob
from pathlib import Path
import pandas as pd

#### CREATING THE ARCHITECTURE

path = "GoogleStreetView_images"
DIR_UNLABELLED = "all_frames/"
DIR_ARCHIVE = DIR_UNLABELLED+"archives/"
DIR_LABELLED = "labelled_data/"

# Mapping on keyboard obtained through trial and error. For Mac 
actions = {
    '0': 48,
    '1': 49,
    '2': 50,
    '3': 51,
    '4': 52,
    '5': 53,
    '6': 54,
    '7': 55,
}

## code bit dealing with folder architecture
# if there's no 'archives' folder, create one 
if not os.path.isdir(path + "/" + DIR_ARCHIVE):
    os.mkdir(os.path.join(path, DIR_ARCHIVE))
init_idx = len(glob.glob(os.path.join(path, DIR_ARCHIVE)+"*.jpg"))
print(init_idx)

if not os.path.isdir(path + "/" + DIR_LABELLED):
    os.mkdir(os.path.join(path, DIR_LABELLED))
    print("maintain composure")
for i in range(len(actions)):
    if not os.path.isdir(os.path.join(path, DIR_LABELLED+str(i))):
        os.mkdir(os.path.join(path, DIR_LABELLED+str(i)))

#### IMAGE DOWNLOADING USING GOOGLE STREET VIEW API & INTERSECTION COORDINATES DATASET(S)

def ottawaDatasetProcess(){
    ## This code block is for the Ottawa dataset ##
    # (straightforward): 
    # dataset @: https://open.ottawa.ca/datasets/7ca6b8a8e36d4bada0bd166755c147cb_0?geometry=-76.784%2C45.150%2C-74.827%2C45.488&page=16

    ## NOTE 1: Ottawa dataset is already fully contained in our Google Street View dataset.
    ## NOTE 2: Must add your own personal or professional access key.

    colnames = ['OBJECTID_1', 'INSTALLATION_YEAR', 'LOCATION_X', 'LOCATION_Y', 'LAT', 'LONG']
    coordinatesData = pd.read_csv('./Downloads/2019_Pedestrian_Crossover_Locations.csv', names=colnames)

    latList = coordinatesData.LAT.tolist()
    latList.pop(0)
    longList = coordinatesData.LONG.tolist()
    longList.pop(0)
    print(len(latList), " intersection coordinates contained in dataset")

    for i in range(len(latList)):
        print(i)
        apiargs = {
            'location': latList[i] + ',' + longList[i],
            'size' : '640x640',
            'heading': '0;90;180;270',
            'key': ''  ## add your access key    
        }
        print(apiargs['location'])
        
        api_list = google_streetview.helpers.api_list(apiargs)
        results = google_streetview.api.results(api_list)

        try:
            path = './GoogleStreetView_images/all_frames/' + str(i)
            os.mkdir(path)
        except OSError:
            print ("Creation of the directory %s failed" % path)
        results.download_links('./GoogleStreetView_images/all_frames/' + str(i), 'metadata.json')
}

def montrealDatasetProcess(){
    ## this code block is for the greater Montreal area dataset ##
    ## dataset: VOI_INTERSECTION_S_N12_data.csv
    ## NOTE 1: Very large dataset (65673 intersection coordinates). As of today (September 6th, 2020) only 1211 intersections have 
    ## been processed (image extraction) and labelled (~1.84%).
    ## Hence, if performing new image extraction & labelling, skip the first 1211 rows of the dataset.
    ## NOTE 2: Must add your own personal or professional access key.

    colnames2 = ['ID_VOI_INTERSECTION','DATECONSTRUCTION','DATECONSTRUCTIONPREC_REF','DATERESURFACAGE',
                'MATERIAUINTER_REF', 'POSITION_REF', 'PROPRIETAIRE_REF', 'TYPEFONDATION_REF', 'TYPEINTERSECTION_REF',
                'TYPEUSAGECYCLABLE_REF','DATE_VERSION','GEOMETRY']

    coordinatesData = pd.read_csv('./Downloads/VOI_INTERSECTION_S_N12_data.csv', names=colnames2, 
                                skiprows=1211, nrows=200)

    polygonList = coordinatesData.GEOMETRY.tolist()
    latLongList = [] 

    for k in polygonList:
        if k == 'GEOMETRY':
            # do nothing
            print()
        else:
            temp = k.split('((')[1]
            temp2 = temp.split(',')[0] 
            temp3 = temp2.split(' ')
            latLongList.append(temp3[1] + ',' + temp3[0])

    print(len(latLongList), " intersection coordinates contained in dataset")

    for i in range(len(latLongList)):
        print(i)
        apiargs = {
            'location': latLongList[i],
            'size' : '640x640',
            'heading': '0;90;180;270',
            'key': ''  ## add your access key    
        }
        print(apiargs['location'])
        
        api_list = google_streetview.helpers.api_list(apiargs)
        results = google_streetview.api.results(api_list)

        try:
            path = './GoogleStreetView_images/all_frames/' + str(i)
            os.mkdir(path)
        except OSError:
            print ("Creation of the directory %s failed" % path)
        results.download_links('./GoogleStreetView_images/all_frames/' + str(i), 'metadata.json')
}

def moveImages(){
	## rename and move all files to the all_frames folder ##
	rootDir = './GoogleStreetView_images/all_frames/'
	subDirs = os.listdir(rootDir)
	numImagesLeftToLabel = len(glob.glob(os.path.join(path, DIR_UNLABELLED)+"*.jpg"))
	k = numImagesLeftToLabel + 1 # variable for renaming
	count = 0
	for sub in subDirs: # iterate through the sub_directories 
        if sub == '.DS_Store':
            # do nothing
            print()
        elif sub == 'archives':
            # do nothing
            print()
        else:
            files = os.listdir(rootDir + sub)
            print(files)
            for file in files: # iterate through the files in the sub_directory
                if file == 'metadata.json':
                    # do nothing
                    print()
                else:
                    os.rename(rootDir + sub + '/' + file, rootDir + str(k) + '.jpg') # renames & moves the photo
                    count += 1
                    k = k + 1
            shutil.rmtree(rootDir + sub)
    print(count, " images moved to all_frames")
}

## call the functions as desired.