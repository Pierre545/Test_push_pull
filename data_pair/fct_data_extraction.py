from wave import Error
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import csv
import rasterio
import torch
import math
import deepinv as dinv
import random

class raster_data():
    @staticmethod
    def array2raster(raster, array, outpath):
        kwargs = raster.meta
        kwargs.update(dtype=rasterio.float32, count=1)

        with rasterio.open(outpath, 'w', **kwargs) as dst:
            dst.write_band(1, array.astype(rasterio.float32))
    @staticmethod
    def import_raster_data(path2raster):
        raster = rasterio.open(path2raster)
        array = raster.read(1)

        return (array)


class little_pair(raster_data):
    def __init__(self, path2raster_centerline, batch_size, scale=None, overlap=None, percentage_overlap=None ):
        """
        :param path2raster_centerline: raster corresponding to the centerline of the tensor considered
        :param batch_size: size of the batch to be considered
        :param scale: pay attention to the shape of the current tensor cropped and the shape of the centerline
        (ex: tensor.shape=[1,1,100,100] centerline.shape=[300,300], the value of scale will be 3)
        :param overlap: no input if accepted, else 1
        :param percentage_overlap: if overlap not accepted "overlap=1", you can specify the max percentage of overlaping accepted between two images between [0,1]
        : ex: you choose 0.1 that means that you accept an overlapping of equal to batch_size*(1-0.1), so 90% of overlapping
        """
        

        # self.square_crop_dict_1 = {}
        # self.square_crop_dict_2 = {}
        self.square_crop_dict_1 = []
        self.square_crop_dict_2 = []
        self.r_array = []
        self.c_array = []
        self.selected_r_array = []
        self.selected_c_array = []
        self.overlap = overlap
        self.scale = scale
        self.batch_size = batch_size
        self.centerline = little_pair.import_raster_data(path2raster_centerline)
        # self.centerline = self.centerline[::self.scale,::self.scale]
        self.final_coords = []

        if percentage_overlap is None:
            self.percentage_overlap = 1
        else:
            self.percentage_overlap = percentage_overlap

        self.batch_index()

    def batch_index(self):
        r_array_tmp, c_array_tmp = np.where(self.centerline == 1)

        half_batch = int(self.batch_size / 2)
        upper_c_limit = len(c_array_tmp)-1-half_batch
        upper_r_limit = len(r_array_tmp)-1-half_batch
        self.r_array  = []
        self.c_array  = []

        for i in range(len(r_array_tmp)):
            #Pay attention that batch do not overlap border
            if r_array_tmp[i] > half_batch and c_array_tmp[i] > half_batch and r_array_tmp[i] < upper_r_limit and c_array_tmp[i] < upper_c_limit:
                self.r_array.append(r_array_tmp[i])
                self.c_array.append(c_array_tmp[i])

    def no_overlap(self, a, b):
        "Logique de fonctionnement : Si  un point (R, C) est selectionné, le code supprime tous les autres candidats" 
        "qui partagent la même bande de lignes a< R < b OU la même bande de colonnes a < C < b "
        "Elle supprime tous les points dont la ligne ou la colonne est entre a et b."
        " Si les points de centerline sont alignés horizontalement ou verticalement, "
        "un seul point sélectionné peut vider des centaines de candidats sur toute la longueur de l'image,"
        " même s'ils sont loin physiquement" 
        "Si on valides un patch , on ne pourra plus jamais prendre de patches sur "
        "la même ligne ou la même colonne, même s'ils sont à l'autre bout de l'image."

        "Resultat : Nombre de points via DE : 82 "
        "Nombre de points via Crop  : 1  "
        indices_2_remove = []
        for index, value in enumerate(self.r_array):    #Remove index of batch horizontally superposed
            if a <= value <= b:
                indices_2_remove.append(index)
        self.r_array = [value for idx, value in enumerate(self.r_array) if idx not in indices_2_remove]
        self.c_array = [value for idx, value in enumerate(self.c_array) if idx not in indices_2_remove]

        indices_2_remove = []
        for index, value in enumerate(self.c_array):    #Remove index of batch vertically superposed
            if a <= value <= b:
                indices_2_remove.append(index)
        self.r_array = [value for idx, value in enumerate(self.r_array) if idx not in indices_2_remove]
        self.c_array = [value for idx, value in enumerate(self.c_array) if idx not in indices_2_remove]

    ######## Fonction no_overlap avec la methode de distance euclidienne
    " Les resultats sont beaucoup plus meilleur avec cette methode"
    " Elle calcule la distance directe (en ligne droite) entre le nouveau centre et les candidats."
    " Elle ne supprime que les points situés à l intérieur dun rayon min_dist."
    
    " Resultats : Nombre de points via DE  : 82 "
    " Nombre de points via Crop  : 86  "

    # def no_overlap(self, last_r, last_c):
    #     min_dist = self.batch_size * (1 - self.percentage_overlap)
    
    #     indices_2_keep = []
    #     for i in range(len(self.r_array)):
    #         # Calcul de la distance euclidienne avec le point que l'on vient de cropper
    #         dist = math.sqrt((self.r_array[i] - last_r)**2 + (self.c_array[i] - last_c)**2)
    #         if dist >= min_dist:
    #             indices_2_keep.append(i)
            
    #         # On ne garde que les points qui sont en dehors du cercle d'exclusion
    #     self.r_array = [self.r_array[i] for i in indices_2_keep]
    #     self.c_array = [self.c_array[i] for i in indices_2_keep]
    
    
    def crop_pair_hv(self, path2tensor_1, path2tensor_2):
        assert torch.load(path2tensor_1,weights_only=True).shape[2]//torch.load(path2tensor_2,weights_only=True).shape[2] ==  self.scale, "This code only works with scale equal to the dimension ratio between the two images"

        spectral_tensor_1 = torch.load(path2tensor_1, weights_only=True)
        spectral_tensor_2 = torch.load(path2tensor_2, weights_only=True)

        d, n, r, c = spectral_tensor_2.size()
        half_size = self.batch_size // 2
        batch_max = c // self.batch_size
        nb_data = 0
        i = -1

        while  len(self.r_array)>0 :
            # print(f" Indice number {i} / Total number of indices {len(self.r_array)}")
            i = i + 1
            # print(f"size of r_array {len(self.r_array)} valeur de i {i}")

            tmp_1 = self.r_array[i]
            tmp_2 = self.c_array[i]

            # For Sentinel data
            r_tensor_1 = tmp_1 - half_size
            r_tensor_2 = tmp_1 + half_size
            c_tensor_1 = tmp_2 - half_size
            c_tensor_2 = tmp_2 + half_size
            square_crop_1 = torch.zeros(1,n,self.batch_size, self.batch_size)
            square_crop_1[0,:,:,:] = spectral_tensor_1[0, :, r_tensor_1:r_tensor_2, c_tensor_1:c_tensor_2]

            # For Landsat data
            r_tensor_1 = r_tensor_1//self.scale
            r_tensor_2 = r_tensor_2//self.scale
            c_tensor_1 = c_tensor_1//self.scale
            c_tensor_2 = c_tensor_2//self.scale
            square_crop_2 = torch.zeros(1, n, self.batch_size//self.scale, self.batch_size//self.scale)
            square_crop_2[0, :, :, :] = spectral_tensor_2[0, :, r_tensor_1:r_tensor_2, c_tensor_1:c_tensor_2]

            """
            Calculate the percentage of exploitable data in the images
            """
            percentage_zero_crop1 = (torch.count_nonzero(square_crop_1) * 100)/ (square_crop_1.shape[2]*square_crop_1.shape[3]*n)
            percentage_zero_crop2 = (torch.count_nonzero(square_crop_2) * 100)/ (square_crop_2.shape[2]*square_crop_2.shape[3]*n)
            percentage_nine_crop2 = (torch.sum(square_crop_2 == -9999) * 100) / (square_crop_2.shape[2]*square_crop_2.shape[3]*n)
            percentage_inf = (torch.sum(torch.isinf(square_crop_1)) * 100) / (square_crop_2.shape[2]*square_crop_1.shape[3]*n)

            if percentage_zero_crop1 >= 99 and percentage_zero_crop2 >= 99 and percentage_nine_crop2 <= 1 and percentage_inf <= 1:
                self.square_crop_dict_1.append(square_crop_1)
                self.square_crop_dict_2.append(square_crop_2)

                # dinv.utils.plot([square_crop_1[0, :3, :, :].unsqueeze(0)],titles=f"Sentinel-2 - date {path2tensor_1[40:47]}")
                # dinv.utils.plot([square_crop_2[0, :3, :, :].unsqueeze(0)],titles=f"Landsat - date {path2tensor_2[44:51]}")
                self.final_coords.append((tmp_1, tmp_2)) 
   
                nb_data = nb_data + 1

                if self.overlap is not None:

                    self.selected_r_array = self.r_array[i] + self.batch_size * self.percentage_overlap
                    self.selected_c_array = self.c_array[i] + self.batch_size * self.percentage_overlap
                    #little_pair.no_overlap(self, self.selected_r_array, self.selected_c_array)

                    self.no_overlap(tmp_1, tmp_2)
                    i = -1
                    # i = 0 ?
                    # print(f"new r_data size {len(self.r_array)}")

            # torch.save((self.square_crop_dict_1),"./dict_S")
            # torch.save((self.square_crop_dict_2),"./dict_L")
            if i >= len(self.r_array)-1:
                print(f"Searched index: {i} but remaining indices are: {len(self.r_array)}")
                break

        print(f"Total number of cropped images: {nb_data}")






# Cette classe a de l'intérêt lorsque l'on utilise pas les données Harmonizé de Landsat (HLS)
class data_crop():
    @staticmethod
    def touch(path_new_file):
            with open(path_new_file, 'a'):
                os.utime(path_new_file, None)


    #This class should be used in an Conda environment with gdal installed
    def crop(self,path_2):
        # Utilisation d'une API gdal pour effectuer le crop sur notre zone d'intéret gdalwarp -cutline <polygon> -crop_to_cutline <input> <output>

        # Création d'un fichier bash pour executer les opérations ave l'API gdal (besoin d'etre avec l'interpreter conda)
        data_crop.touch("/home/paudisio/Desktop/2023/processing_S2.sh")
        f = open("/home/paudisio/Desktop/2023/processing_S2.sh", "a")

        data_crop.touch("/home/paudisio/Desktop/2023/processing_L8.sh")
        g = open("/home/paudisio/Desktop/2023/processing_L8.sh", "a")


        directory_1 = '/home/paudisio/Desktop/2023/L8_bis'
        for i in os.listdir(directory_1):
            path_11 = directory_1 + "/" + i

            #Création d'un dossier pour chaque date d'acquisition
            crop_folder_12 = "/home/paudisio/Desktop/2023/L8_bis_crop/" + i
            os.mkdir(crop_folder_12)

            for j in os.listdir(path_11):
                if ".tif" in j and ".xml" not in j:

                    path_12 = path_11 + "/" + j
                    g.writelines("gdalwarp -cutline" + " " + "/home/paudisio/Desktop/2023/Lhasa_S2_extent.gpkg" + " " + "-crop_to_cutline" + " " + path_12 + " " + crop_folder_12 +"/"+ j + "\n")
        g.close


        directory_2 = '/home/paudisio/Desktop/2023/S2_bis'
        for i in os.listdir(directory_2):
            path_21 = directory_2 + "/" + i

            #Création d'un dossier pour chaque date d'acquisition
            crop_folder_21 = "/home/paudisio/Desktop/2023/S2_bis_crop/" + i
            os.mkdir(crop_folder_21)
            for j in os.listdir(path_21):
                if ".xml" not in j:
                    path_22 = path_21 + "/" + j
                    f.writelines("gdalwarp -cutline" + " " + "/home/paudisio/Desktop/2023/Lhasa_S2_extent.gpkg" + " " + "-crop_to_cutline" + " " + path_22 + " " + crop_folder_21 +"/"+ j + ".tif" +"\n")
        f.close




class images_pair() :
    @staticmethod
    def paths_pair(csv_1, csv_2):

        df_1 = pd.read_csv(str(csv_1))
        df_2 = pd.read_csv(str(csv_2))

        keys_dict = ['class', 'path', 'data']
        for key in keys_dict:
            if key not in df_1.keys():
                print("Missing key :", key)
                return()

        try:
            i = 0
            pairs_dict = {}
            for class_1 in df_1['class']:
                path_1 = os.path.join(df_1['path'][i], df_1['data'][i])
                pairs_dict[str(path_1)] = []
                j = 0
                for class_2 in df_2['class']:
                    if class_1 == class_2:
                        path_2 = os.path.join(df_2['path'][j], df_2['data'][j])
                        # print(class_1, class_2)
                        # print(path_1, path_2)
                        pairs_dict[str(path_1)].append(str(path_2))

                    j = j + 1
                i = i + 1
        except ValueError:
            pass

        return (pairs_dict)

    @staticmethod
    def write_dict(dict_pair,out_path):
        """
        :param dict_pair should be the dictionnary containing the pair of path associated with file
        """

        try:
            with open(str(out_path),"w", newline="") as csvfile:
                writer = csv.writer(csvfile)
                # Write headers
                writer.writerow(["data_1", "data_2"])
                # Write data
                for data_1, data_2 in dict_pair.items():
                    writer.writerow([data_1, data_2])
            print("Output file for HR/LR pair :", out_path)

        except ValueError:
            print(ValueError)