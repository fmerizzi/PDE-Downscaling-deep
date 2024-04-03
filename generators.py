import numpy as np
import matplotlib.pyplot as plt
import cv2
import os
import keras.utils

#max and mins for normalization purposes

vhtrain0_max = 25.128353
vhtrain0_min = -28.335325
vhtrain1_max = 29.295961
vhtrain1_min = -25.943642
vhtrain2_max = 104154.54
vhtrain2_min = 97233.336

eratrain0_max = 13.444138
eratrain0_min = -16.445755
eratrain1_max = 17.889359
eratrain1_min = -18.977646
eratrain2_max = 104217.875
eratrain2_min = 89486.81

target_min = -0.9143369
target_max = 1.2363808


class DataGeneratorMemmap(keras.utils.Sequence):
    def __init__(self, era5_path, vhrea_path, target_image_path, ephem_path, num_images, sequential=False, batch_size=24, vhrrea_flag = True):
        # create memory-mapped files era5, vhrea and target
        self.era5 = np.memmap(era5_path, dtype='float32', mode='r', offset = 128, shape=(3, num_images, 5, 9))
        self.vhrrea = np.memmap(vhrea_path, dtype='float32', mode='r', offset = 128, shape=(3, num_images, 44, 84))
        self.image_target = np.memmap(target_image_path, dtype='float32', mode='r',offset = 128, shape=(num_images, 130, 256))
        # its so small I don't neep to map it on disk
        self.ephem = np.load(ephem_path)
        #  the flag discriminate between era5 and vhrrea
        self.vhrrea_flag = vhrrea_flag           
        # set boolean for sequential or random dataset
        self.sequential = sequential
        # counter for keeping track of seuquential generator 
        self.counter = 0
        # set sequence len 
        self.sequence = 4
        # number of features (todo make it really modular) 
        self.num_features = 3
        self.batch_size = batch_size
        self.num_samples = num_images
        self.num_batches = int(np.ceil(self.num_samples / self.batch_size))
        #output dims
        self.dimA = 128 
        self.dimB = 256

    def __len__(self):
        return self.num_batches

    # min max normalization 
    def min_max_normalize(self, arr, max, min):
        return (arr - min) / (max - min)

    # call to restart the sequential counter  
    def counter_reset(self):
        self.counter = 0
    
    def __getitem__(self, idx):
        # prepare the resulting arrays for input and output
        inputs =  np.zeros((self.batch_size, self.dimA, self.dimB, self.sequence * self.num_features))
        ephem = np.zeros((self.batch_size,6))
        outputs = np.zeros((self.batch_size, self.dimA, self.dimB,1))
        
        # random path 
        if(self.sequential == False):  
            #compose the batch one element at the time (this is vectorizable, but may be valuable to keep it this way)
            for i in range(self.batch_size):
                # get a random number in range 
                random = np.random.randint(0, (self.num_samples - self.sequence)) 
                # get the low_res items, 2 past, 1 present 1 future & normalization
                if(self.vhrrea_flag):
                    #data selection
                    items = self.vhrrea[:, random + 1:random + self.sequence + 1] / 1
                    #normalization
                    items[0] = self.min_max_normalize(items[0],vhtrain0_max,vhtrain0_min)
                    items[1] = self.min_max_normalize(items[1],vhtrain1_max,vhtrain1_min)
                    items[2] = self.min_max_normalize(items[2],vhtrain2_max,vhtrain2_min)
                else: 
                    #data selection
                    items = self.era5[:, random + 1:random + self.sequence + 1] / 1
                    #normalization
                    items[0] = self.min_max_normalize(items[0],eratrain0_max,eratrain0_min)
                    items[1] = self.min_max_normalize(items[1],eratrain1_max,eratrain1_min)
                    items[2] = self.min_max_normalize(items[2],eratrain2_max,eratrain2_min)
                    #possible switch for coherence 
                    #items[[0,1]] = items[[1,0]]

                #aggregate the features on the channel axis
                items = items.reshape((-1, items.shape[-2], items.shape[-1]))
                items = np.expand_dims(items, axis=-1)
                items = np.swapaxes(items, 0, 3)

                #generate correct ephem 
                ephem[i] = self.ephem[:,random]
                #upscale the images via bilinear to 128x256
                for k in range(self.sequence * self.num_features):
                    inputs[i, :, :, k] = cv2.resize(items[0, :, :, k], (self.dimB, self.dimA), interpolation=cv2.INTER_LINEAR)
                # get the target high res results 
                outputs[i,:,:,0] = self.image_target[random + self.sequence - 1, :-2]
                #normalization outputs 
                outputs[i,:,:,0] = self.min_max_normalize(outputs[i,:,:,0], target_max, target_min)
    
        # sequential path 
        if(self.sequential == True):
            for i in range(self.batch_size):
                # get the new sequence (+1 on the last)
                if(self.vhrrea_flag):
                    items = self.vhrrea[:, self.counter + 1:self.counter + self.sequence + 1] / 1
                    #normalization
                    items[0] = self.min_max_normalize(items[0],vhtrain0_max,vhtrain0_min)
                    items[1] = self.min_max_normalize(items[1],vhtrain1_max,vhtrain1_min)
                    items[2] = self.min_max_normalize(items[2],vhtrain2_max,vhtrain2_min)
                else: 
                    items = self.era5[:, self.counter + 1:self.counter + self.sequence + 1, 1:6] / 1
                    #normalization
                    items[0] = self.min_max_normalize(items[0],eratrain0_max,eratrain0_min)
                    items[1] = self.min_max_normalize(items[1],eratrain1_max,eratrain1_min)
                    items[2] = self.min_max_normalize(items[2],eratrain2_max,eratrain2_min)
                    #items[[0,1]] = items[[1,0]]
                    
                items = items.reshape((-1, items.shape[-2], items.shape[-1]))
                items = np.expand_dims(items, axis=-1)
                items = np.swapaxes(items, 0, 3)

                #generate correct ephem 
                ephem[i] = self.ephem[:,self.counter]
                #upscale the images via bilinear to 256x256 
                for k in range(self.sequence * self.num_features):
                    inputs[i, :, :, k] = cv2.resize(items[0, :, :, k], (self.dimB, self.dimA), interpolation=cv2.INTER_LINEAR)
                # get the target high res results 
                outputs[i,:,:,0] = self.image_target[self.counter + self.sequence - 1, :-2]
                #normalization outputs 
                outputs[i,:,:,0] = self.min_max_normalize(outputs[i,:,:,0], target_max, target_min)
                self.counter = self.counter + 1      

        return (inputs,ephem), outputs
        