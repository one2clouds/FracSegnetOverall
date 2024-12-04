import SimpleITK as sitk 
from monai.metrics import DiceMetric, HausdorffDistanceMetric
from tqdm import tqdm 
import torch 
from glob import glob 
import os 
import numpy as np
from monai.transforms import AsDiscrete



if __name__ == '__main__':
    
    test_label_names = sorted(glob("/mnt/Enterprise2/shirshak/PENGWIN_TASK/PENGWIN_CT/labels/*.mha"))[80:]
    test_pred_names = sorted(glob('/mnt/Enterprise2/shirshak/NewFracSegNet/FracSegNet/output_folder_overall_preprocessing/*.nii.gz'))

    for (test_label_name, test_pred_name) in zip(test_label_names, test_pred_names):
        print('-'*80)

        print(os.path.split(test_label_name)[1])
        print(os.path.split(test_pred_name)[1])

        gt_arr = sitk.GetArrayFromImage(sitk.ReadImage(test_label_name))
        predicted_arr = sitk.GetArrayFromImage(sitk.ReadImage(test_pred_name))

        num_class_onehot = max(len(np.unique(gt_arr)), len(np.unique(gt_arr)))

        y_preds = AsDiscrete(to_onehot= num_class_onehot)(torch.as_tensor(gt_arr).unsqueeze(0)).unsqueeze(0)
        y =  AsDiscrete(to_onehot= num_class_onehot)(torch.as_tensor(predicted_arr).unsqueeze(0)).unsqueeze(0)

        print(y_preds.shape)
        print(y.shape)

        dice_score = DiceMetric(include_background=False)(torch.as_tensor(gt_arr).unsqueeze(0).unsqueeze(0), torch.as_tensor(predicted_arr).unsqueeze(0).unsqueeze(0))
        dice_score.nan_to_num(nan=0, posinf=0, neginf=0)

        hausdorff = HausdorffDistanceMetric(include_background=False, percentile=95)(torch.as_tensor(gt_arr).unsqueeze(0).unsqueeze(0), torch.as_tensor(predicted_arr).unsqueeze(0).unsqueeze(0))
        hausdorff.nan_to_num(nan=-100, posinf=-100, neginf=-100)

        print('-'*80)



