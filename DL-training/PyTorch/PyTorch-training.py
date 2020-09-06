# importing necessary modules
from __future__ import print_function
from __future__ import division
import torch 
import sys
import numpy as np
import torch.nn as nn
import torch.optim as optim
import torchvision
from torch.utils.data import Dataset, DataLoader
from torchvision.datasets import ImageFolder
import matplotlib.pyplot as plt
import time
import os, shutil
from shutil import copyfile
import copy
from PIL import Image
import csv
import random
print("All necessary modules imported.")

def load_model(model_Name, num_classes, feature_extract, dict_path=None):
    print("dict_path is None: " + str(dict_path==None))
    if model_Name == "SqueezeNet":
        model = torchvision.models.squeezenet1_0(pretrained=False)
        PATH = dict_path + "/squeezenet1_0-a815701f.pth"
        if (dict_path != None):
            model.load_state_dict(torch.load(PATH))
            print("Pretrained model successfully loaded.")
        set_parameters_that_require_grad(model, feature_extract)
        model.classifier[1] = nn.Conv2d(512, num_classes, kernel_size=(1,1), stride=(1,1))
        model.num_classes = num_classes
        input_size = 224
    elif model_Name == "mobilenet_v2":
        model = torchvision.models.mobilenet_v2(pretrained=False)
        PATH = dict_path + "/mobilenet_v2-b0353104.pth"
        if (dict_path != None):
            model.load_state_dict(torch.load(PATH))
            print("Pretrained model successfully loaded.")
        set_parameters_that_require_grad(model, feature_extract)        
        model.classifier[1] = nn.Linear(1280, num_classes)
        model.num_classes = num_classes
        input_size = 224
    elif model_Name == "resnet50":
        model = torchvision.models.resnet50(pretrained=False)
        PATH = dict_path + "/resnet50-19c8e357.pth"
        if (dict_path != None):
            model.load_state_dict(torch.load(PATH))
            print("Pretrained model successfully loaded.")
        set_parameters_that_require_grad(model, feature_extract)
        # reshaping the network
        num_in_features = model.fc.in_features
        model.fc = nn.Linear(num_in_features, num_classes) # adding Linear layer at the end
        input_size = 224
    elif model_Name == "inception_v3": 
#       Inception v3
#       Be careful, expects (299,299) sized images and has auxiliary output
        model = torchvision.models.inception_v3(pretrained=False)
        PATH = dict_path + "/inception_v3_google-1a9a5a14.pth"
        print("Starting to load...")
        if (dict_path != None):
            model.load_state_dict(torch.load(PATH))
            print("Pretrained model successfully loaded.")
        input_size = 299
    elif model_Name == "vgg11_bn":
        model = torchvision.models.vgg11_bn(pretrained=False)
        PATH = dict_path + "/vgg11_bn-6002323d.pth"
        if (dict_path != None):
            model.load_state_dict(torch.load(PATH))
            print("Pretrained model successfully loaded.")
        set_parameters_that_require_grad(model, feature_extract)
        num_in_features = model.classifier[6].in_features
        model.classifier[6] = nn.Linear(num_in_features, num_classes)
        input_size = 224
    else:
        raise Exception("Model not recognized. Exiting.")
    return model, input_size

def set_parameters_that_require_grad(model, feature_extract):
    if feature_extract: # if we're in feature extracting mode
        for param in model.parameters():
            param.requires_grad = False # freeze pretrained model parameters

# function loading labeled dataset (expecting labelling structure)
def load_dataset(data_path, transforms):
    data_set = ImageFolder(data_path, transforms)
    return data_set

class LabeledDataset(Dataset):
    def __init__(self, dataset, transform=None):
        self.dataset = dataset
        self.transform = transform
    
    def __getitem__(self, index):
        if self.transform:
            x = self.transform(dataset[index][0])
        else:
            x = dataset[index][0]
        y = dataset[index][1]
        return x, y
    
    def __len__(self):

        return len(dataset)

##  training function  ##
def train_model(model, dataloaders, criterion, optimizer, num_epochs=25, is_inception=False):
    since = time.time()

    val_acc_history = []
    
    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0
    best_epoch = 0
    valResultsPerEpoch = {}
    
    for epoch in tqdm(range(num_epochs)):
        print("epoch #", epoch)
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10) 
        
        valResultsPerEpoch[epoch] = []
        
        # Each epoch has a training and validation phase
        for phase in tqdm(['train', 'val']):
            if phase == 'train':
                model.train()  # Set model to training mode
            else:
                model.eval()   # Set model to evaluate mode

            running_loss = 0.0
            running_corrects = 0
            
            # Iterate over data.
            for inputs, labels in tqdm(dataloaders[phase]):
                inputs = inputs.to(device)
                labels = labels.to(device)
                
                # zero the parameter gradients
                optimizer.zero_grad()
                
                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == 'train'):
                    # Get model outputs and calculate loss
                    # Special case for inception because in training it has an
                    # auxiliary output. In train mode we calculate the loss by
                    # summing the final output and the auxiliary output but in
                    # testing we only consider the final output.
                    if is_inception and phase == 'train':
                        outputs, aux_outputs = model(inputs)
                        loss1 = criterion(outputs, labels)
                        loss2 = criterion(aux_outputs, labels)
                        loss = loss1 + 0.4*loss2
                    else:
                        print("(Not Inception)")
                        outputs = model(inputs)
                        loss = criterion(outputs, labels)

                    _, preds = torch.max(outputs, 1)
                    
                    ####
                    if phase == 'val':
                        print("preds-labels:")
                        diff = torch.abs(preds-labels) # returns a tensor of absolute value of differences
                        diff_list = diff.tolist()
                        valResultsPerEpoch[epoch].extend(diff_list)
                        print("DONE.")
                    ####
                    
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()
                    
                # statistics
                # addition for each dataloader batch
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)
            
            epoch_loss = running_loss / len(dataloaders[phase].dataset)
            epoch_acc = running_corrects.double() / len(dataloaders[phase].dataset)
                
            print('{} Loss: {:.4f} Acc: {:.4f}'.format(phase, epoch_loss, epoch_acc))
            
            # deep copy the model
            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_epoch = epoch
                best_model_wts = copy.deepcopy(model.state_dict())
            if phase == 'val':
                val_acc_history.append(epoch_acc)
                
        print()
        
    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
    print('Best epoch was epoch #', best_epoch)
    print('Best val Acc: {:4f}'.format(best_acc))   
    
    # load best model weights
    model.load_state_dict(best_model_wts)
    return model, val_acc_history, valResultsPerEpoch, best_epoch

### function returns the number of files in the subdirectory structure ###
def howManyFiles(path):
    notebook_path = os.path.dirname(os.path.realpath('__file__'))
    rootDir = notebook_path + path
    subDirs = os.listdir(rootDir)
    count = 0
    for sub in subDirs:
        files = os.listdir(rootDir + sub)
        count += len(files)
    print(path, " contains ", count, " images.")
    return count

howManyFiles('/GoogleStreetView_images/labelled_data')

def plot_it(hist, num_epochs):
    ohist = []
    ohist = [h.cpu().numpy() for h in hist]
    
    plt.title("Validation Accuracy vs. Number of Training Epochs")
    plt.xlabel("Training Epochs")
    plt.ylabel("Validation Accuracy")
    plt.plot(range(1,num_epochs+1),ohist,label="Pretrained")
    plt.ylim((0,1.))
    plt.xticks(np.arange(1, num_epochs+1, 1.0))
    plt.legend()
    plt.savefig('validation_accuracy_graph.png')
    plt.show()

def main():
    # proxy of the main function to be implemented
    phoneData_dir = "./Rogers_Corpus"
    data360_dir   = "./CV-Aid-for-Visually-Impaired/data_processing/labelled_360/labelled_data"
    streetViewData_dir = "./GoogleStreetView_images/labelled_data_already_scp-ed"

    # Ratio of training data to validation data
    train_ratio = 0.9

    # Number of classes in the dataset (class 0 being unknown)
    num_classes = 8 # for google street view data

    # Batch size for training (change depending on how much memory you have)
    batch_size = 64

    # Number of epochs to train for
    num_epochs = 20

    # Flag for feature extracting. When False, we finetune the whole model,
    # when True we only update the reshaped (final) layer params
    feature_extract = False

    # Hyperparameters for models (learning rate and momentum)
    learning_rate = 0.001
    momentum = 0.9

    # enter desired pretrained model
    modelName = "mobilenet_v2"

    modelX, input_size = load_model(modelName, num_classes, feature_extract, "./raw_pretrained_models")
    print("Model imported.")
    
    data_transform = transforms.Compose([
        transforms.Resize(input_size),
        transforms.ToTensor(),
        # normalize using ImageNet's mean & standard deviation values
        transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                             std=[0.229, 0.224, 0.225])
    ])
    
    print("Initializing Datasets and Dataloaders...")
    
    # creating training and validation datasets
    print("input size: ", input_size)
    
    streetView_data_set = load_dataset(streetViewData_dir, data_transform) # 8 labels
    print("-----------")
    print(streetView_data_set)
    print(streetView_data_set.class_to_idx)    
    print("streetView_data_set size: ", len(streetView_data_set))
    
    train_sizeStreetView = int(train_ratio * len(streetView_data_set))
    val_sizeStreetView = len(streetView_data_set) - train_sizeStreetView
    training_StreetView, val_StreetView = torch.utils.data.random_split(streetView_data_set,
                                                                        [train_sizeStreetView, val_sizeStreetView])
    
    print("length training_StreetView: ", len(training_StreetView), " images")
    print(type(training_StreetView))
    print("length val_StreetView: ",len(val_StreetView), " images")
    print(type(val_StreetView))
    
    imageStreetView_datasets = {'train': training_StreetView, 'val': val_StreetView }
        
    dataloadersStreetView_dict = {x: DataLoader(imageStreetView_datasets[x], batch_size=batch_size, shuffle=True) 
                                  for x in ['train', 'val']}

    print(dataloadersStreetView_dict['train'])
    print("length of trainStreetView dataloader: ", len(dataloadersStreetView_dict['train']), " batches") 
    
    modelX = modelX.to(device)

    params_to_update = modelX.parameters()
    if feature_extract:
        params_to_update = []
        for name, param in modelX.named_parameters():
            if param.requires_grad is True:
                params_to_update.append(param)
                print("\t", name)
    else:
        for name, param in modelX.named_parameters():
            if param.requires_grad is True:
                print("\t", name)

    # Observe that all parameters are being optimized
    optimizerX = optim.SGD(params_to_update, lr=learning_rate, momentum=momentum)
    
    # Setup the loss function
    criterion = nn.CrossEntropyLoss()
    
    # Train and evaluate
    modelX, hist, valResultsPerEpoch, bestEpoch = train_model(modelX, dataloadersStreetView_dict, criterion,
                                 optimizerX, num_epochs=num_epochs,
                                 is_inception=(modelName == "inception"))
    
    print("best epoch was: ", bestEpoch)
    print("valResultsPerEpoch for this epoch:")
    print(valResultsPerEpoch[bestEpoch])
    
    torch.save(modelX, "./bestModel-Jul09.pth")
    
    plot_it(hist, num_epochs)
     
#     sys.stdout.close()
#     plot_it(hist, num_epochs)

if __name__ == "__main__":
    main()

### check best trained model on a new random validation set ###
accuracy = running_corrects.double() / len(dataloadersStreetView_dict['val'].dataset)
print(accuracy)
