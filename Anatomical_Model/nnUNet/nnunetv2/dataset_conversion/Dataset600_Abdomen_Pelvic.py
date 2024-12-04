import shutil
from batchgenerators.utilities.file_and_folder_operations import join, subfolders, maybe_mkdir_p, save_json
import os
import glob 
from tqdm import tqdm 
import shutil 
from nnunetv2.paths import nnUNet_raw
import SimpleITK as sitk 
import torch 

#when defining base environment variable nnUNet_raw samma matra rakhne and base chai nnUNet_raw_data rakhne
base = nnUNet_raw 
task_id = 600
task_name = "CT_PelvicAnatomical"

foldername = "Dataset%03.0d_%s" % (task_id, task_name)

out_base = join(base, foldername)
imagestr = join(out_base, "imagesTr")
imagests = join(out_base, "imagesTs")
labelstr = join(out_base, "labelsTr")
maybe_mkdir_p(imagestr)
maybe_mkdir_p(imagests)
maybe_mkdir_p(labelstr)

folder_raw = "/mnt/Enterprise2/shirshak/PENGWIN_OTHER_DATASETS/OVERALL_MERGED/"

all_cases = subfolders(folder_raw, join=True)

n_train_case = 0
n_test_case = 0

train_ct_names = sorted(glob.glob("/mnt/Enterprise2/shirshak/PENGWIN_OTHER_DATASETS/OVERALL_MERGED/Training/img/*.nii.gz"))
train_mask_names = sorted(glob.glob("/mnt/Enterprise2/shirshak/PENGWIN_OTHER_DATASETS/OVERALL_MERGED/Training/label/*.nii.gz"))
test_ct_names = sorted(glob.glob("/mnt/Enterprise2/shirshak/PENGWIN_OTHER_DATASETS/OVERALL_MERGED/Testing/img/*.nii.gz"))


# nnunet doesnot change the direction of image so......
def change_direction(orig_image):
    new_img = sitk.DICOMOrient(orig_image,'RAS')
    return new_img


if __name__ == '__main__':
    # print(train_ct_names[200])
    # print(train_mask_names[200])
    # start enumeration from 1 
    for index, (train_img_name, train_label_name) in enumerate(tqdm(zip(train_ct_names, train_mask_names)), 1):
        # print(train_img_name)
        # print(join(imagestr, "CT_PELVIC1K_" + str(index).zfill(3) + "_0000.nii.gz")) # zfill for filling up to 3 values
        # print(train_label_name)
        # print(join(labelstr, "CT_PELVIC1K_" + str(index).zfill(3) + ".nii.gz"))
        
        ct_scale_img = sitk.ReadImage(train_img_name)
        ct_scale_img = change_direction(ct_scale_img)
        sitk.WriteImage(ct_scale_img, join(imagestr, "CT_PELVIC1K_" + str(index).zfill(3) + "_0000.nii.gz"))
        
        ct_scale_mask = sitk.ReadImage(train_label_name)
        ct_scale_mask = change_direction(ct_scale_mask)
        sitk.WriteImage(ct_scale_mask, join(labelstr, "CT_PELVIC1K_" + str(index).zfill(3) + ".nii.gz"))
        n_train_case += 1
    
    for index, test_img_name in enumerate(tqdm(test_ct_names), 1):
        # print(test_img_name)
        # print(join(imagests, str(index).zfill(3) + "_0000.nii.gz"))
        ct_test_img = sitk.ReadImage(test_img_name)
        ct_test_img = change_direction(ct_test_img)
        sitk.WriteImage(ct_test_img, join(imagests, "CT_PELVIC1K_" + str(index).zfill(3) + "_0000.nii.gz"))
        n_test_case += 1


    json_dict = {}
    json_dict['name'] = "CT fracture all 54 res"
    json_dict['tensorImageSize'] = "4D"
    json_dict['reference'] = "CT fracture all 54 res"
    json_dict['release'] = "2022.11.03"

    json_dict['labels'] = {
        "background":0,
        "Sacrum" : 1,
        "Left Hipbone": 2,
        "Right Hipbone" : 3,
        "Lumber Spine" : 4
    }

    json_dict['channel_names'] = {0: "CT"}

    json_dict['numTraining'] = n_train_case
    json_dict['numTest'] = n_test_case
    json_dict['file_ending'] = ".nii.gz"
    json_dict['test'] = []

    save_json(json_dict, os.path.join(out_base, "dataset.json"))

# Environment Variables 
# export nnUNet_raw="/mnt/Enterprise2/shirshak/NewFracSegNet/Anatomical_Model/dataset/nnUNet_raw"
# export nnUNet_preprocessed="/mnt/Enterprise2/shirshak/NewFracSegNet/Anatomical_Model/dataset/nnUNet_preprocessed"
# export nnUNet_results="/mnt/Enterprise2/shirshak/NewFracSegNet/Anatomical_Model/dataset/nnUNet_results"


# export nnUNet_raw="/mnt/Enterprise2/shirshak/NewFracSegNet/Anatomical_Model/dataset/nnUNet_raw"
# export nnUNet_preprocessed="/mnt/Enterprise2/shirshak/NewFracSegNet/Anatomical_Model/dataset/nnUNet_preprocessed"
# export nnUNet_results="/mnt/Enterprise2/shirshak/NewFracSegNet/Anatomical_Model/dataset/nnUNet_results"
