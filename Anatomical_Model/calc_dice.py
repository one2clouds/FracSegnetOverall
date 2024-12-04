import SimpleITK as sitk 
from monai.metrics import DiceMetric
from tqdm import tqdm 
import torch 
from glob import glob 
import os 

def  make_discrete_data(data):
    # print(data.unique())
    data = torch.from_numpy(data)
    discreted_data = torch.zeros_like(data)

    discreted_data[(data>0) & (data<=10)] = 1
    discreted_data[(data>10) & (data<=20)] = 2
    discreted_data[(data>20) & (data<=30)] = 3
    return discreted_data



if __name__ == '__main__':
    test_label_names = sorted(glob("/mnt/Enterprise2/shirshak/PENGWIN_TASK/PENGWIN_CT/labels/*.mha"))[240:]
    test_pred_names = sorted(glob('/mnt/Enterprise2/shirshak/NewFracSegNet/Anatomical_Model/output_folder/*.nii.gz'))

    sum = 0
    for (test_label_name, test_pred_name) in zip(test_label_names, test_pred_names):

        print(os.path.split(test_label_name)[1])
        print(os.path.split(test_pred_name)[1])

        print('-'*80)
        sitk_arr = make_discrete_data(sitk.GetArrayFromImage(sitk.ReadImage(test_label_name)))
        sitk_predicted = sitk.GetArrayFromImage(sitk.ReadImage(test_pred_name))

        dice_metric_calc = DiceMetric(reduction="mean")(torch.as_tensor(sitk_predicted).unsqueeze(0).unsqueeze(0), torch.as_tensor(sitk_arr).unsqueeze(0).unsqueeze(0))
        print(dice_metric_calc)
        sum += dice_metric_calc
        print('-'*80)

    print(f"Overall Dice {sum / len(test_pred_names)}")