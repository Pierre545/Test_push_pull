import os
path_work = "C:/Users/pC/Desktop/STAGE_ENSL/2023_out"
os.chdir(path_work)

import numpy as np
import pandas as pd
from collections import Counter
import torch
from torch.utils.data import DataLoader
from torch.utils.data import TensorDataset
import sys
from verify_overlap_mod import (verify_overlap_condition , visualize_comparison)
from main_function_original import main_fct
from fct_data_extraction import (little_pair, images_pair)



#Path for S2 and L8 data
path_1               = 'C:/Users/pC/Desktop/STAGE_ENSL/2023/S2_bis_crop'    #Sentinel-2 data
path_2               = 'C:/Users/pC/Desktop/STAGE_ENSL/2023/L8_bis_crop'    #HLS data

outpath_1, outpath_2 = main_fct.SL_create_csv(path_1,path_2)    #Create a 2 csv files associating a class to each image depending on the acquisition date


pair_dict                 =  images_pair.paths_pair(outpath_1, outpath_2)
csv_pair_path             = "C:/Users/pC/Desktop/STAGE_ENSL/2023_out/path_pair_test.csv"

#Save a csv containing path_pair
images_pair.write_dict(pair_dict,csv_pair_path)

main_fct.tensor_creator(csv_pair_path,"C:/Users/pC/Desktop/STAGE_ENSL/2023_out/test1",number_files=None )    #"./test1" is the path of the folder wich will contain all the output, here the HR_LR folders



def dataset_creation(path_1, save=1):
    #path_1 should correspond to the path of the folder containing all the folder of pair
    path_centerline = "C:/Users/pC/Desktop/STAGE_ENSL/RCT_raster_centerline.tif"

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
                all_coords = []
                if len(little_pair_object.square_crop_dict_1) and len(little_pair_object.square_crop_dict_2) :
                    all_coords.extend(little_pair_object.final_coords)
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
        return np.array(all_coords)





BATCH_SIZE = 300
SCALE = 3
OVERLAP_VAL = 0.7


if __name__ == "__main__":
    
    path_centerline = "C:/Users/pC/Desktop/STAGE_ENSL/RCT_raster_centerline.tif"
    outpath_1, outpath_2 = main_fct.SL_create_csv(path_1, path_2)
    pair_dict = images_pair.paths_pair(outpath_1, outpath_2)
    csv_pair_path = "C:/Users/pC/Desktop/STAGE_ENSL/2023_out/path_pair_test.csv"
    images_pair.write_dict(pair_dict, csv_pair_path)
    

    # Positions de Distance Euclidienne
    coords_DE = verify_overlap_condition(path_centerline, BATCH_SIZE, SCALE, OVERLAP_VAL, visualize=True)    

    # dataset creation
    coords_real = dataset_creation("C:/Users/pC/Desktop/STAGE_ENSL/2023_out/test1")
    
    
    # COMPARAISON
    coords_real_unique = np.unique(coords_real, axis=0)  # pour ne garder que les positions géographiques distinctes
    print(f"Nombre de positions  via crop : {len(coords_real_unique)}")
    print(f"Nombre de positions  via DE : {len(coords_DE)}")
    
    visualize_comparison(path_centerline, coords_DE, coords_real_unique, BATCH_SIZE, OVERLAP_VAL)
    
    
        