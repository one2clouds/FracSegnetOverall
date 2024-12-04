'''

coding by erick liu, 6/16/2023

input: image(.nii.gz) finished by anatomical segment
output: Each label is stored separately as a separate image(.nii.gz) for annotation
'''
import os
# from nnunet.basicFunc import *
import SimpleITK as sitk
import pandas as pd
import glob 
from tqdm import tqdm 
import numpy as np
import argparse

def maybe_mkdir_p(directory: str) -> None:
    os.makedirs(directory, exist_ok=True)

# nnunet doesnot change the direction of image so......
def change_direction(orig_image):
    new_img = sitk.DICOMOrient(orig_image,'RAS')
    return new_img


def extractSingleFrac(ct_scale_img, ct_label_img, label):

    ct_origin_arr = sitk.GetArrayFromImage(ct_scale_img)
    ct_label_arr = sitk.GetArrayFromImage(ct_label_img)
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


def saveDiffFrac(fileName, labelName, preprocessed_dir):
    # load image data
    splitName = os.path.split(fileName) 
    split_labelName = os.path.split(labelName)


    ct_origin_img = sitk.ReadImage(fileName)
    ct_origin_img = change_direction(ct_origin_img)

    ct_label_img = sitk.ReadImage(labelName)
    ct_label_img = change_direction(ct_label_img)

    # label = 1: Sacrum / 2: Left Hip / 3:Right Hip

    frac_sacrum_img, frac_sacrum_label = extractSingleFrac(ct_origin_img, ct_label_img, 1)
    frac_LeftIliac_img, frac_LeftIliac_label = extractSingleFrac(ct_origin_img, ct_label_img, 2)
    frac_RightIliac_img, frac_RightIliac_label = extractSingleFrac(ct_origin_img, ct_label_img, 3)


    maybe_mkdir_p(os.path.join(preprocessed_dir, "nifti_fragments_LI_SA_RI", "images"))
    maybe_mkdir_p(os.path.join(preprocessed_dir, "nifti_fragments_LI_SA_RI" ,"labels"))

    sitk.WriteImage(frac_sacrum_img, os.path.join(preprocessed_dir,"nifti_fragments_LI_SA_RI","images", splitName[1].split('.mha', 2)[0] + '_SA_image.nii.gz'))
    sitk.WriteImage(frac_LeftIliac_img, os.path.join(preprocessed_dir,"nifti_fragments_LI_SA_RI","images", splitName[1].split('.mha', 2)[0] + '_LI_image.nii.gz'))
    sitk.WriteImage(frac_RightIliac_img, os.path.join(preprocessed_dir,"nifti_fragments_LI_SA_RI","images",splitName[1].split('.mha', 2)[0] + '_RI_image.nii.gz'))

    sitk.WriteImage(frac_sacrum_label, os.path.join(preprocessed_dir,"nifti_fragments_LI_SA_RI","labels", split_labelName[1].split('.mha', 2)[0] + '_SA_label.nii.gz'))
    sitk.WriteImage(frac_LeftIliac_label, os.path.join(preprocessed_dir,"nifti_fragments_LI_SA_RI","labels", split_labelName[1].split('.mha', 2)[0] + '_LI_label.nii.gz'))
    sitk.WriteImage(frac_RightIliac_label, os.path.join(preprocessed_dir,"nifti_fragments_LI_SA_RI","labels", split_labelName[1].split('.mha', 2)[0] + '_RI_label.nii.gz'))
    
    return 0


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Process CT images to extract and save different fractures.')
    parser.add_argument('-i', '--input', required=True, help='Path to the input CT image and label')
    parser.add_argument('-o', '--output', required=True, help='Path to the output_file where imgs are written')

    args = parser.parse_args()

    ct_name = sorted(glob.glob(f"{args.input}/part*/*.mha"))
    mask_name = sorted(glob.glob(f"{args.input}/labels/*.mha"))
    # print(len(ct_name)) 100
    # print(len(mask_name)) 100
    for ct_name, mask_name in tqdm(zip(ct_name, mask_name)):
        saveDiffFrac(ct_name,mask_name, args.output)



# python3 extract_ct_regions.py -i /mnt/Enterprise2/shirshak/PENGWIN_TASK/PENGWIN_CT/ -o /mnt/Enterprise2/shirshak/PENGWIN_TASK/PENGWIN_CT/ 
