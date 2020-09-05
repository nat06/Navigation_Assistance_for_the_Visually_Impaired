#!/usr/bin/env python
# coding: utf-8

# In[1]:


import cv2
import numpy as np
import glob
import os
print("Libraries imported.")


# In[2]:


path = "GoogleStreetView_images"
DIR_UNLABELLED = "all_frames/"
print(DIR_UNLABELLED)
DIR_ARCHIVE = DIR_UNLABELLED+"archives/"
print(DIR_ARCHIVE)
DIR_LABELLED = "labelled_data/"
print(DIR_LABELLED)


# In[3]:


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
for read_key in actions.values():
    print(read_key)
    
# if there's no 'archives' folder, create one 
if not os.path.isdir(path + "/" + DIR_ARCHIVE):
    print("swaggy")
    os.mkdir(os.path.join(path, DIR_ARCHIVE))
    print("no time for games here Swaggy")
init_idx = len(glob.glob(DIR_ARCHIVE+"*.jpg"))
print("grab your loot.")
print(init_idx)

if not os.path.isdir(path + "/" + DIR_LABELLED):
    os.mkdir(os.path.join(path, DIR_LABELLED))
    print("maintain composure")
for i in range(len(actions)):
    if not os.path.isdir(os.path.join(path, DIR_LABELLED+str(i))):
        os.mkdir(os.path.join(path, DIR_LABELLED+str(i)))

all_fnames = sorted(glob.glob(os.path.join(path, DIR_UNLABELLED)+"*.jpg"))
print("Number of Frames: ", len(all_fnames))


# In[4]:


for img_idx, fname in enumerate(all_fnames):
    print(img_idx, "  ", fname)
    if img_idx % 100 == 0:
        print("Number of completed images: ", str(init_idx+img_idx+1))
    image = cv2.imread(fname)
    image = cv2.resize(image, (684, 684))

    # make a copy to not save lines
    orig_img = cv2.resize(image.copy(), (224, 224))

    # Put vertical lines
    x = image.shape[1]//12
    line = x
    for _ in range(7):
        cv2.line(image, (line, 0), (line, image.shape[0]), (0, 0, 0), 2)
        line += int(2 * x)

    labelled = 0
    cv2.imshow(fname, image)
    while labelled == 0:
        read_key = 0xFF & cv2.waitKey(0)
        print(read_key)
        if read_key in actions.values():
            label = int([key for (key, value) in actions.items() if value == read_key][0])
            new_fname = os.path.join(path, DIR_LABELLED)+str(label)+"/%05d.jpg" % (img_idx+init_idx)
            cv2.imwrite(new_fname, orig_img)

            if label != 0 or label != 4:
                rotated_img = cv2.flip(orig_img, 1)
                new_label = len(actions) - label
                new_fname = os.path.join(path, DIR_LABELLED)+str(new_label)+"/%05d.jpg" % (img_idx+init_idx)
                cv2.imwrite(new_fname, rotated_img)

            os.rename(fname, os.path.join(path, DIR_ARCHIVE)+"%05d.jpg" % (img_idx+init_idx))
            labelled = 1

    cv2.destroyAllWindows()


# In[ ]:




