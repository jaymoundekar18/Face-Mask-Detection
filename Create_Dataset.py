#Import required libraries
import os 
from shutil import copy
import random

old_path = "data/"        # Downloaded dataset path

# Creating new dataset directory
os.mkdir("DATASET/Train")
os.mkdir("DATASET/Test")
os.mkdir("DATASET/Val")

os.mkdir("DATASET/Train/with_mask")
os.mkdir("DATASET/Train/without_mask")

os.mkdir("DATASET/Test/with_mask")
os.mkdir("DATASET/Test/without_mask")

os.mkdir("DATASET/Val/with_mask")
os.mkdir("DATASET/Val/without_mask")

#-------------------------------------------------------------------------

# Function to split the data 
def split_data(Source, Training, Testing, Validation, Split_size):
  files = []
  
  for fname in os.listdir(Source):
    file = Source + fname
    
    if os.path.getsize(file) > 0:
      files.append(fname)
    
    else:
      print(fname + " is zero length, so ignoring..")
    
  training_length = int(len(files) * Split_size)
  testing_length = int((len(files)-training_length)*0.5)
  validation_length = int((len(files)-training_length)*0.5)
  shuffled_set = random.sample(files, len(files))
  training_set = shuffled_set[0:training_length]
  testing_set = shuffled_set[training_length:-testing_length]
  validation_set = shuffled_set[-testing_length:]

  for fname in training_set:
    this_file = Source + fname
    copy(this_file, Training)
  
  for fname in testing_set:
    this_file = Source + fname
    copy(this_file, Testing)

  for fname in validation_set:
    this_file = Source + fname
    copy(this_file, Validation)


#-------------------------------------------------------------------------

# For With Mask Images
Source = "data/with_mask/"                      
Training = "DATASET/Train/with_mask"            
Testing = "DATASET/Test/with_mask"              
Validation = "DATASET/Val/with_mask"            
Split_size = 0.7

split_data(Source,Training,Testing,Validation,Split_size)

#-------------------------------------------------------------------------

# For Without Mask Images
Source = "data/without_mask/"                          
Training = "DATASET/Train/without_mask"                
Testing = "DATASET/Test/without_mask"                  
Validation = "DATASET/Val/without_mask"               
Split_size = 0.7

split_data(Source,Training,Testing,Validation,Split_size)

#-------------------------------------------------------------------------

print("Train", len(os.listdir("DATASET/Train/with_mask/")), "," ,len(os.listdir("DATASET/Train/without_mask/")))

print("Test", len(os.listdir("DATASET/Test/with_mask/")), "," ,len(os.listdir("DATASET/Test/without_mask/")))

print("Val", len(os.listdir("DATASET/Val/with_mask/")), "," ,len(os.listdir("DATASET/Val/without_mask/")))
