import os
path_work = "/projects/EVS-Sisyphe/Paudisio/Database"
os.chdir(path_work)

import numpy as np
import pandas as pd
from collections import Counter
import torch
from torch.utils.data import DataLoader
from torch.utils.data import TensorDataset
import sys
from main_function_beta import main_fct
from data_pair.fct_data_extraction import (little_pair, images_pair)



#Path for S2 and L8 data
path_1               = './2023/crop/S2_bis_crop'    #Sentinel-2 data
path_2               = './2023/crop/L8_bis_crop'    #HLS data

outpath_1, outpath_2 = main_fct.SL_create_csv(path_1,path_2)    #Create a 2 csv files associating a class to each image depending on the acquisition date


pair_dict                 =  images_pair.paths_pair(outpath_1, outpath_2)
csv_pair_path             = "2023/path_pair_test.csv"
#Save a csv containing path_pair
images_pair.write_dict(pair_dict,csv_pair_path)

main_fct.tensor_creator(csv_pair_path,"./test1",number_files=None )    #"./test1" is the path of the folder wich will contain all the output, here the HR_LR folders



def dataset_creation(path_1, save=1):
    #path_1 should correspond to the path of the folder containing all the folder of pair
    path_centerline = "./2023/RCT_raster_centerline.tif"

    dir = os.listdir(path_1)
    n = 1
    for file in dir:
        path_2 = os.path.join(path_1, file)
        print(file)
        for file_name in os.listdir(path_2):
            if "HLS" in file_name:
                tmp_HLS = os.path.join(path_2,file_name)

        for file_name in os.listdir(path_2):
            if "Sentinel2" in file_name:
                tmp_S2 = os.path.join(path_2,file_name)

                little_pair_object = little_pair(path_centerline,300,3, overlap=1, percentage_overlap=0.7)
                # little_pair_object = little_pair(path_centerline,300,3) #117195 data obtained

                little_pair_object.crop_pair_hv(tmp_S2, tmp_HLS)

                """
                Check if littler_pair_object. are not considered empty
                """
                if len(little_pair_object.square_crop_dict_1) and len(little_pair_object.square_crop_dict_2) :

                    tmp_target = torch.cat(little_pair_object.square_crop_dict_1)
                    tmp_train = torch.cat(little_pair_object.square_crop_dict_2)

                #Group the different list of crop obtained
                if n>1 :
                    tensor_target = torch.cat((tensor_target, tmp_target), dim=0)
                    tensor_train  = torch.cat((tensor_train, tmp_train), dim=0)
                else :
                    tensor_target = tmp_target
                    tensor_train = tmp_train

                    n = n + 1

        print(f"Size of dataset : {len(tensor_train)}")

        dataset = TensorDataset(tensor_train, tensor_target)

        if save :
            torch.save(dataset, "./LandsatHLS_Sentinel2_dataset_2Version.pth")


# if __name__ == "__main__":
path_1 = "./test1"
# dataset_creation(path_1,1)
dataset_creation(path_1)


# spectral_tensor = torch.load("total_dataset_WithShuffle_2703.pth",weights_only=False)
# loader = DataLoader(dataset, batch_size=2)
# dinv.utils.plot(spectral_tensor[0])
#os.chdir("/projects/EVS-Sisyphe/Paudisio/Database")


# Normaliser les données
# Clipping input data to the valid range for imshow with RGB data ([0..1] for floats or [0..255] for integers). Got range [-9999.0..-9999.0].

#Normalize iteratively the data
# for i in range(len(train)):
#     tmp = target[i]
#     target[i] = normalize(tmp)