import pandas as pd
import glob 
from tqdm import tqdm 
import SimpleITK as sitk
import os 
import numpy as np


def change_direction(orig_image, new_img):
    if sitk.DICOMOrientImageFilter_GetOrientationFromDirectionCosines(orig_image.GetDirection()) == 'LPS':
        new_img = sitk.DICOMOrient(orig_image,'RAS')
    elif sitk.DICOMOrientImageFilter_GetOrientationFromDirectionCosines(orig_image.GetDirection()) == 'RAS':
        new_img.SetDirection(orig_image.GetDirection())
    else:
        print('Error while changing the direction of image')
        print(KeyError)
    return new_img

def check_voxels_resolution_and_size():
    img_list = sorted(glob.glob('/mnt/Enterprise2/shirshak/PENGWIN_TASK/PENGWIN_CT/part*/*.mha'))[:10]
    label_list = sorted(glob.glob('/mnt/Enterprise2/shirshak/PENGWIN_TASK/PENGWIN_CT/labels/*.mha'))[:10]

    # print(len(img_list))
    # print(len(label_list))

    columns = ['Name','Orientation', 'Re-Orientation']
    df = pd.DataFrame(columns=columns)

    for index, image_and_mask in enumerate(tqdm(zip(img_list, label_list))):
        temp_image_sitk = sitk.ReadImage(image_and_mask[0])
        temp_mask_sitk=sitk.ReadImage(image_and_mask[1])

        # print(temp_image_arr.shape)
        # print(voxel_size_image)
        # print(temp_mask_arr.shape)
        # print(voxel_size_mask)

        new_img = change_direction(temp_image_sitk, temp_image_sitk)
        new_mask = change_direction(temp_mask_sitk, temp_mask_sitk)

        df.loc[index+1, 'Name'] = os.path.split(image_and_mask[1])[1].split('.mha')[0]
        df.loc[index+1, 'Orientation'] = sitk.DICOMOrientImageFilter_GetOrientationFromDirectionCosines(temp_image_sitk.GetDirection()), sitk.DICOMOrientImageFilter_GetOrientationFromDirectionCosines(temp_mask_sitk.GetDirection())
        df.loc[index + 1, 'Re-Orientation'] = sitk.DICOMOrientImageFilter_GetOrientationFromDirectionCosines(new_img.GetDirection()), sitk.DICOMOrientImageFilter_GetOrientationFromDirectionCosines(new_mask.GetDirection())
    return df

if __name__ == '__main__':
    df = check_voxels_resolution_and_size()
    print(df)
    df.to_csv('zzz_tests/only_some_orientation.csv')
