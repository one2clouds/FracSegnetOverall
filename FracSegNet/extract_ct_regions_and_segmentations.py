'''

coding by erick liu, 6/16/2023

input: image(.nii.gz) finished by anatomical segment
output: Each label is stored separately as a separate image(.nii.gz) for annotation
'''
import os
from nnunet.basicFunc import *
import SimpleITK as sitk
import pandas as pd
import glob 
from tqdm import tqdm 
from batchgenerators.utilities.file_and_folder_operations import join, maybe_mkdir_p


# input: ct_origin_fileName,ct_label_fileName,label output:frac_Grayscale==label
def extractSingleFrac(ct_scale_img, ct_label_img, label):
    # ct_origin_img = sitk.ReadImage(ct_origin_fileName)
    # ct_label_img = sitk.ReadImage(ct_label_fileName)
    ct_origin_arr = sitk.GetArrayFromImage(ct_scale_img)  # get array from image
    ct_label_arr = sitk.GetArrayFromImage(ct_label_img)  # get array from image
    frac_Grayscale_img = ct_origin_arr.copy()
    frac_Grayscale_label = ct_label_arr.copy()

    if label == 1:
        frac_Grayscale_img[ct_label_arr < 1] = 0
        frac_Grayscale_img[ct_label_arr > 10] = 0
        frac_Grayscale_label[ct_label_arr < 1] = 0
        frac_Grayscale_label[ct_label_arr > 10] = 0
    if label == 2:
        frac_Grayscale_img[ct_label_arr < 10] = 0
        frac_Grayscale_img[ct_label_arr > 20] = 0
        frac_Grayscale_label[ct_label_arr < 10] = 0
        frac_Grayscale_label[ct_label_arr > 20] = 0
    if label == 3:
        frac_Grayscale_img[ct_label_arr < 20] = 0
        frac_Grayscale_img[ct_label_arr > 30] = 0
        frac_Grayscale_label[ct_label_arr < 20] = 0
        frac_Grayscale_label[ct_label_arr > 30] = 0

    frac_Grayscale_img, frac_Grayscale_label = sitk.GetImageFromArray(frac_Grayscale_img), sitk.GetImageFromArray(frac_Grayscale_label)

    frac_Grayscale_img.SetDirection(ct_scale_img.GetDirection())
    frac_Grayscale_img.SetSpacing(ct_scale_img.GetSpacing())
    frac_Grayscale_img.SetOrigin(ct_scale_img.GetOrigin())

    frac_Grayscale_label.SetDirection(ct_label_img.GetDirection())
    frac_Grayscale_label.SetSpacing(ct_label_img.GetSpacing())
    frac_Grayscale_label.SetOrigin(ct_label_img.GetOrigin())

    return frac_Grayscale_img, frac_Grayscale_label



def saveDiffFrac(fileName, labelName):
    # load image data
    splitName = os.path.split(fileName) # this splits filename from the directory name.. Eg..('/mnt/Enterprise2/shirshak/PENGWIN_TASK/PENGWIN_CT/part1', '001.mha')
    split_labelName = os.path.split(labelName)
    # print(name) # ['001', '']

    # ct_scale_img = pelvisOriginData(fileDir, maskName)
    #
    # ct_label_img = sitk.ReadImage(maskName)
    ct_origin_img = sitk.ReadImage(fileName)
    ct_label_img = sitk.ReadImage(labelName)

    # print(np.unique(sitk.GetArrayFromImage(ct_label_img)))

    # label = 1: Sacrum / 2: Left Hip / 3:Right Hip
    # =============================================================================================
    # ======================extract the single fracture and Rescale Intensity======================
    # =============================================================================================
    frac_sacrum_img, frac_sacrum_label = extractSingleFrac(ct_origin_img, ct_label_img, 1)
    frac_LeftIliac_img, frac_LeftIliac_label = extractSingleFrac(ct_origin_img, ct_label_img, 2)
    frac_RightIliac_img, frac_RightIliac_label = extractSingleFrac(ct_origin_img, ct_label_img, 3)

    # Before this i have made a directory nifty_preprocessed_into_fragments is specified location
    maybe_mkdir_p(os.path.join(os.path.split(splitName[0])[0], "nifti_fragments_LI_SA_RI", "images"))
    maybe_mkdir_p(os.path.join(os.path.split(splitName[0])[0], "nifti_fragments_LI_SA_RI", "labels"))

    sitk.WriteImage(frac_sacrum_img, os.path.join(os.path.split(splitName[0])[0],"nifti_fragments_LI_SA_RI","images", splitName[1].split('.mha', 2)[0] + '_SA_image.nii.gz'))
    sitk.WriteImage(frac_LeftIliac_img, os.path.join(os.path.split(splitName[0])[0],"nifti_fragments_LI_SA_RI","images", splitName[1].split('.mha', 2)[0] + '_LI_image.nii.gz'))
    sitk.WriteImage(frac_RightIliac_img, os.path.join(os.path.split(splitName[0])[0],"nifti_fragments_LI_SA_RI", "images",splitName[1].split('.mha', 2)[0] + '_RI_image.nii.gz'))

    sitk.WriteImage(frac_sacrum_label, os.path.join(os.path.split(split_labelName[0])[0], "nifti_fragments_LI_SA_RI", "labels", split_labelName[1].split('.mha', 2)[0] + '_SA_label.nii.gz'))
    sitk.WriteImage(frac_LeftIliac_label, os.path.join(os.path.split(split_labelName[0])[0], "nifti_fragments_LI_SA_RI", "labels", split_labelName[1].split('.mha', 2)[0] + '_LI_label.nii.gz'))
    sitk.WriteImage(frac_RightIliac_label, os.path.join(os.path.split(split_labelName[0])[0], "nifti_fragments_LI_SA_RI", "labels", split_labelName[1].split('.mha', 2)[0] + '_RI_label.nii.gz'))
    return 0


if __name__ == "__main__":
    ct_name = sorted(glob.glob("/mnt/Enterprise2/shirshak/PENGWIN_TASK/PENGWIN_CT/part*/*.mha"))
    mask_name = sorted(glob.glob("/mnt/Enterprise2/shirshak/PENGWIN_TASK/PENGWIN_CT/labels/*.mha"))
    # print(len(ct_name)) 100
    # print(len(mask_name)) 100
    for ct_name, mask_name in tqdm(zip(ct_name, mask_name)):
        saveDiffFrac(ct_name,mask_name)


# export nnUNet_raw_data_base="/home/shirshak/FracSegNet/dataset/nnUNet_raw"
# export nnUNet_preprocessed="/home/shirshak/FracSegNet/dataset/nnUNet_preprocessed"
# export RESULTS_FOLDER="/home/shirshak/FracSegNet/dataset/nnUNet_results"

# export nnUNet_raw_data_base="/home/shirshak/FracSegNet/dataset/nnUNet_raw"
# export nnUNet_preprocessed="/home/shirshak/FracSegNet/dataset/nnUNet_preprocessed"
# export RESULTS_FOLDER="/home/shirshak/FracSegNet/dataset/nnUNet_results"
