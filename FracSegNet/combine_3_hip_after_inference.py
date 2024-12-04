import SimpleITK as sitk 
from monai.metrics import DiceMetric, HausdorffDistanceMetric
from tqdm import tqdm 
import torch 
from glob import glob 
import os 
import numpy as np
from monai.transforms import AsDiscrete
from scipy.ndimage import label


def separate_labels_for_non_connected_splitted_fragments(mask_arr: np.ndarray, for_which_classes: list, volume_per_voxel: float,
                                                   minimum_valid_object_size: int = None) -> np.ndarray:
    """
    gives separate label for non connected components other than main fragment in an mask array.
    :param image:
    :param for_which_classes: can be None. Should be list of int.
    :param minimum_valid_object_size: Only objects larger than minimum_valid_object_size will be removed.
    :return:
    """
    if for_which_classes is None:
        for_which_classes = np.unique(mask_arr)
        for_which_classes = for_which_classes[for_which_classes > 1] # Not taking background (0) and largest class (1)

    assert 0 not in for_which_classes, "background scannot be incorporated"
    assert 1 not in for_which_classes, "largest class, class 1  couldnot be incorporated, only small fragments can be incorporated"


    for c in for_which_classes: # for_which_classes = [2]
        mask = np.zeros_like(mask_arr, dtype=bool)
        mask = mask_arr == c


        # get labelmap and number of objects
        lmap, num_objects = label(mask.astype(int))

        # collect object sizes
        object_sizes = {}
        for object_id in range(1, num_objects + 1):
            object_sizes[object_id] = (lmap == object_id).sum() * volume_per_voxel

        # Removing smaller objects which is lesser than threshold
        if num_objects > 0:
            for object_id in range(1, num_objects + 1):
                    if minimum_valid_object_size is not None:
                        if object_sizes[object_id] < minimum_valid_object_size:
                            mask_arr[(lmap == object_id) & mask] = 0
                            del object_sizes[object_id]

        num_objects = len(object_sizes)

        if num_objects > 0:
            mask_value = 2 # Mask value for background is already 0, for largest bone is already 1, so starting from 2 for sub-fragments, and taking 2 for largest sub fragment bone...
            for _ in range(num_objects):
                # print('printing object sizesssssssss')
                # print(len(object_sizes))
                if len(object_sizes) > 1:
                    maximum_size = max(object_sizes.values()) 
                else:
                # For final component, comparing it with minimum valid object if it is not None. If it is none comparing it with 0
                    maximum_size = max(list(object_sizes.values())[0], [minimum_valid_object_size if minimum_valid_object_size is not None else 0])
                    
                for object_id in list(object_sizes.keys()):
                    # print(object_sizes[object_id])
                    # print(maximum_size)
                    if object_sizes[object_id] == maximum_size:
                        # mark that as label mask_value(2)
                        mask_arr[(lmap == object_id) & mask] = mask_value
                        # remove that object_sizes[object_id]
                        del object_sizes[object_id]
                        # print(f"Remove and deleted largest component {object_id}")
                        break
                    else:
                        continue 
                mask_value += 1   
    return mask_arr

def get_preds(mask_preprocessed_arr, base_value):
    mask = mask_preprocessed_arr.copy()
    mask[mask_preprocessed_arr == 1] = base_value + 1
    mask[mask_preprocessed_arr == 2] = base_value + 2
    mask[mask_preprocessed_arr == 3] = base_value + 3
    mask[mask_preprocessed_arr == 4] = base_value + 4
    mask[mask_preprocessed_arr == 5] = base_value + 5
    mask[mask_preprocessed_arr == 6] = base_value + 6
    return mask

def return_one_with_max_probability(li_mask, sa_mask, ri_mask, li_prob, sa_prob, ri_prob):

    # print(li_prob.shape) # (3, 128, 128, 128) #SInce there is background, class 1, class 2....
    # print(sa_prob.shape) # (3, 128, 128, 128)
    # print(ri_prob.shape) # (3, 128, 128, 128)

    li_prob = li_prob.max(axis=0)
    sa_prob = sa_prob.max(axis=0)
    ri_prob = ri_prob.max(axis=0)

    assert ri_mask.shape == li_mask.shape ==sa_mask.shape == li_prob.shape == ri_prob.shape == sa_prob.shape
    overall_mask = np.zeros_like(li_mask)

    for i in range(ri_mask.shape[0]):
        for j in range(ri_mask.shape[1]):
            for k in range(ri_mask.shape[2]):
                my_dict = {}
                # print(li_mask[i][j][k])
                if li_mask[i][j][k] != 0:
                    my_dict[li_mask[i][j][k]] = li_prob[i][j][k]

                if ri_mask[i][j][k] != 0:
                    my_dict[ri_mask[i][j][k]] = ri_prob[i][j][k]

                if sa_mask[i][j][k] != 0:
                    my_dict[sa_mask[i][j][k]] = sa_prob[i][j][k]

                # This even works if there is only one element in dict
                if len(my_dict) != 0 :
                    overall_mask[i][j][k] = max(my_dict, key=my_dict.get)
                else:
                    overall_mask[i][j][k] = 0
    return overall_mask 




if __name__ == '__main__':
    test_pred_names = sorted(glob('/mnt/Enterprise2/shirshak/NewFracSegNet/FracSegNet/output_folder_doing_preprocessing/*.nii.gz'))

    for index in range(0, len(test_pred_names), 3) :

        LI_img = sitk.ReadImage(test_pred_names[index])
        RI_img = sitk.ReadImage(test_pred_names[index+1])
        SA_img = sitk.ReadImage(test_pred_names[index+2])

        print(os.path.split(test_pred_names[index])[1][:3])

        mask_preprocessed_arr = separate_labels_for_non_connected_splitted_fragments(sitk.GetArrayFromImage(LI_img), for_which_classes=None, volume_per_voxel=float(np.mean(LI_img.GetSpacing(), dtype=np.float64)), minimum_valid_object_size=500)
        li_mask = get_preds(mask_preprocessed_arr, 10)

        mask_preprocessed_arr = separate_labels_for_non_connected_splitted_fragments(sitk.GetArrayFromImage(SA_img), for_which_classes=None, volume_per_voxel=float(np.mean(RI_img.GetSpacing(), dtype=np.float64)), minimum_valid_object_size=500)
        sa_mask = get_preds(mask_preprocessed_arr, 0)

        mask_preprocessed_arr = separate_labels_for_non_connected_splitted_fragments(sitk.GetArrayFromImage(RI_img), for_which_classes=None, volume_per_voxel=float(np.mean(SA_img.GetSpacing(), dtype=np.float64)), minimum_valid_object_size=500)
        ri_mask = get_preds(mask_preprocessed_arr, 20)

        overall_mask = np.zeros_like(li_mask)
        overall_mask = li_mask + sa_mask + ri_mask # return_one_with_max_probability(li_mask, sa_mask, ri_mask, li_prob = class_prob_frac_LeftIliac_arr, sa_prob = class_prob_frac_sacrum_arr, ri_prob=class_prob_frac_RightIliac_arr)

        print(np.unique(overall_mask, return_counts=True))
        overall_mask_img = sitk.GetImageFromArray(overall_mask)
        overall_mask_img.CopyInformation(LI_img)

        print(f"finished {os.path.split(test_pred_names[index])[1][:3]}")
        
        sitk.WriteImage(overall_mask_img, f'/mnt/Enterprise2/shirshak/NewFracSegNet/FracSegNet/output_folder_overall_preprocessing/{os.path.split(test_pred_names[index])[1][:3]}.nii.gz')





