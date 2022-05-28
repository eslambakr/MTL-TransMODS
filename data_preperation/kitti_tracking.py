import numpy as np
import os
import shutil
import cv2

# choose which data u need to prepare
# all: means the static and dynamic datasets.
# moving: means generate the dynamic objects only.
data_type = "moving"
ann_type = None
# Define directories
data_root = "/media/user/data/eslam/kitti_tracking_dataset/"
imgs_dir = data_root + "images/"
anns_dir = data_root + "ann/"

if data_type == "all":
    saving_root = data_root + "custom/"
    ann_type = "np_array_All_classes_Output"
elif data_type == "moving":
    saving_root = data_root + "moving/"
    ann_type = "np_array_moving_Output"
saving_imgs_dir = saving_root + "images/"
saving_anns_dir = saving_root + "ann/"
saving_instances_dir = saving_root + "instances/"
unique_counter = 0

# List the sub dirs
sub_folders = [x[0] for x in os.walk(anns_dir)]
for sub_folder in sub_folders:
    if ann_type in sub_folder:
        print(" Processing ---> ", sub_folder)
        # List the files in the sub dir
        files = []
        for (dirpath, dirnames, filenames) in os.walk(sub_folder):
            files.extend(filenames)
            break
        # Loop on the images
        for f in files:
            data = np.load(sub_folder + "/" + f, allow_pickle=True)
            npData = data['x']
            if len(npData) == 0:
                continue

            # Save each object in a separate mask
            if len(np.shape(npData)) == 2:
                num_of_obj = 1
            else:
                num_of_obj = np.shape(npData)[2]
            image_saved = False
            for obj in range(num_of_obj):
                if 1 in np.unique(npData, return_counts=True)[0]:
                    if not image_saved:
                        image_saved = True
                        # Copy and change the name of the images to be unique ones
                        shutil.copy(imgs_dir + sub_folder.split('/')[-2].split('_output_Data')[0]
                                    + "/image_02/data/" + f.split('.')[0] + ".png",
                                    saving_imgs_dir)
                        os.rename(saving_imgs_dir + f.split('.')[0] + ".png",
                                  saving_imgs_dir + sub_folder.split('/')[-2].split('_output_Data')[0]
                                  + '_' + f.split('.')[0] + ".png")

                    if num_of_obj == 1:
                        cv2.imwrite(saving_instances_dir + sub_folder.split('/')[-2].split('_output_Data')[0]
                                    + '_' + f.split('.')[0] + "_car_" + str(unique_counter) + ".png", npData*255)
                    else:
                        cv2.imwrite(saving_instances_dir + sub_folder.split('/')[-2].split('_output_Data')[0]
                                    + '_' + f.split('.')[0] + "_car_" + str(unique_counter) + ".png",
                                    npData[..., obj]*255)
                    unique_counter += 1

print("Total number of objects = ", unique_counter)
print("Done...!")
