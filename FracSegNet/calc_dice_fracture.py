import SimpleITK as sitk 
from monai.metrics import DiceMetric, HausdorffDistanceMetric
from tqdm import tqdm 
import torch 
from glob import glob 
import os 
import numpy as np
from monai.transforms import AsDiscrete



if __name__ == '__main__':
    test_label_names = sorted(glob("/mnt/Enterprise2/shirshak/NewFracSegNet/FracSegNet/dataset/nnUNet_raw/nnUNet_raw_data/Task601_CT_PelvicFrac150/labelsTs/*.nii.gz"))
    
    
    test_pred_names = sorted(glob('/mnt/Enterprise2/shirshak/NewFracSegNet/FracSegNet/new_output_folder_doing_preprocessing/*.nii.gz'))

    # Dice Score
    LI_main_sum, SA_main_sum, RI_main_sum = 0, 0, 0
    LI_rem_sum, SA_rem_sum, RI_rem_sum = 0, 0, 0

    LI_main_count, SA_main_count, RI_main_count = 0, 0, 0
    LI_rem_count, SA_rem_count, RI_rem_count = 0, 0, 0

    # Hausdorff Distance
    LI_main_hd_sum, SA_main_hd_sum, RI_main_hd_sum = 0, 0, 0
    LI_rem_hd_sum, SA_rem_hd_sum, RI_rem_hd_sum = 0, 0, 0

    LI_main_hd_count, SA_main_hd_count, RI_main_hd_count = 0, 0, 0
    LI_rem_hd_count, SA_rem_hd_count, RI_rem_hd_count = 0, 0, 0

    overall_sum = 0
    overall_hd = 0

    test_label_names = test_label_names
    test_pred_names = test_pred_names

    for (test_label_name, test_pred_name) in zip(test_label_names, test_pred_names):
        print('-'*80)
        gt_arr = sitk.GetArrayFromImage(sitk.ReadImage(test_label_name))
        predicted_arr = sitk.GetArrayFromImage(sitk.ReadImage(test_pred_name))

        y_preds = AsDiscrete(to_onehot= 3)(torch.as_tensor(gt_arr).unsqueeze(0)).unsqueeze(0)
        y =  AsDiscrete(to_onehot= 3)(torch.as_tensor(predicted_arr).unsqueeze(0)).unsqueeze(0)

        dice_score = DiceMetric(include_background=False)(y_preds, y)
        dice_score = dice_score.nan_to_num(nan=0, posinf=0, neginf=0) # because sometimes there is only main fragment and not remaining fragment, so it gives nan when calculating remaining fragment

        hausdorff_score = HausdorffDistanceMetric(include_background=False, percentile=95)(y_preds, y)
        hausdorff_score = hausdorff_score.nan_to_num(nan = -100, posinf=-100, neginf=-100)
        
        print(os.path.split(test_pred_name)[1].split('.nii.gz')[0])
        print("Dice")
        print(dice_score[0][0].item())
        print(dice_score[0][1].item())
        print("Hausdorff")
        print(hausdorff_score[0][0].item())
        print(hausdorff_score[0][1].item())


        if 'LI' in os.path.split(test_pred_name)[1]:
            # print("left iliac")
            # print(os.path.split(test_label_name)[1])
            # print(os.path.split(test_pred_name)[1])
            if dice_score[0][0].item() != 0:
                LI_main_sum += dice_score[0][0].item() 
                LI_main_count += 1
            if dice_score[0][1].item() != 0:
                LI_rem_sum += dice_score[0][1].item()
                LI_rem_count += 1

            if hausdorff_score[0][0].item() != -100:
                LI_main_hd_sum += hausdorff_score[0][0].item()
                LI_main_hd_count += 1
            if hausdorff_score[0][1].item() != -100:
                LI_rem_hd_sum += hausdorff_score[0][1].item()
                LI_rem_hd_count += 1

        elif 'SA' in os.path.split(test_pred_name)[1]:
            # print("sacrum")
            # print(os.path.split(test_label_name)[1])
            # print(os.path.split(test_pred_name)[1])
            if dice_score[0][0].item() != 0:
                SA_main_sum += dice_score[0][0].item() 
                SA_main_count += 1
            if dice_score[0][1].item() != 0:
                SA_rem_sum += dice_score[0][1].item()
                SA_rem_count += 1
            
            if hausdorff_score[0][0].item() != -100:
                SA_main_hd_sum += hausdorff_score[0][0].item()
                SA_main_hd_count += 1
            if hausdorff_score[0][1].item() != -100:
                SA_rem_hd_sum += hausdorff_score[0][1].item()
                SA_rem_hd_count += 1
            
        elif 'RI' in os.path.split(test_pred_name)[1]:
            # print("right iliac")
            # print(os.path.split(test_label_name)[1])
            # print(os.path.split(test_pred_name)[1])
            if dice_score[0][0].item() != 0:
                RI_main_sum += dice_score[0][0].item() 
                RI_main_count += 1
            if dice_score[0][1].item() != 0:
                RI_rem_sum += dice_score[0][1].item()
                RI_rem_count += 1

            if hausdorff_score[0][0].item() != -100:
                RI_main_hd_sum += hausdorff_score[0][0].item()
                RI_main_hd_count += 1
            if hausdorff_score[0][1].item() != -100:
                RI_rem_hd_sum += hausdorff_score[0][1].item()
                RI_rem_hd_count += 1

        print('-'*80)

    print('-'*80)
    print("Dice")
    print("Main")
    print(LI_main_sum) # 18.221632063388824
    print(SA_main_sum) # 17.32523012161255
    print(RI_main_sum) # 18.832050919532776
    print("Remaining")
    print(LI_rem_sum) # 11.60687381029129
    print(SA_rem_sum) # 6.0831557251513
    print(RI_rem_sum) # 10.772076904773712

    print("Counts")

    print("Main")
    print(LI_main_count) # 20
    print(SA_main_count) # 20
    print(RI_main_count) # 20
    print("Remaining")
    print(LI_rem_count) # 16
    print(SA_rem_count) # 9
    print(RI_rem_count) # 15

    print('-'*80)

    print("Hausdorff")
    print("Main")
    print(LI_main_hd_sum)
    print(SA_main_hd_sum)
    print(RI_main_hd_sum)
    print("Remaining")
    print(LI_rem_hd_sum)
    print(SA_rem_hd_sum)
    print(RI_rem_hd_sum)

    print("Counts")
    print("Main")
    print(LI_main_hd_count)
    print(SA_main_hd_count)
    print(RI_main_hd_count)

    print("Remaining")
    print(LI_rem_hd_count)
    print(SA_rem_hd_count)
    print(RI_rem_hd_count)

    
    print(f"LI Main Overall Dice : {LI_main_sum / LI_main_count}") # 0.9110816031694412
    print(f"SA Main Overall Dice : {SA_main_sum / SA_main_count}") # 0.8662615060806275
    print(f"RI Main Overall Dice : {RI_main_sum / RI_main_count}") # 0.9416025459766388

    print(f"LI Remaining Overall Dice : {LI_rem_sum / LI_rem_count}") # 0.7254296131432056
    print(f"SA Remaining Overall Dice : {SA_rem_sum / SA_rem_count}") # 0.6759061916834779
    print(f"RI Remaining Overall Dice : {RI_rem_sum / RI_rem_count}") # 0.7181384603182475

    print(f"LI Main Overall Hausdorff : {LI_main_hd_sum / LI_main_hd_count}")
    print(f"SA Main Overall Hausdorff : {SA_main_hd_sum / SA_main_hd_count}")
    print(f"RI Main Overall Hausdorff : {RI_main_hd_sum / RI_main_hd_count}")

    print(f"LI Remaining Overall Hausdorff : {LI_rem_hd_sum / LI_rem_hd_count}")
    print(f"SA Remaining Overall Hausdorff : {SA_rem_hd_sum / SA_rem_hd_count}")
    print(f"RI Remaining Overall Hausdorff : {RI_rem_hd_sum / RI_rem_hd_count}")

    overall_sum = (LI_main_sum/LI_main_count) + (SA_main_sum/SA_main_count) + (RI_main_sum/RI_main_count) + (LI_rem_sum/LI_rem_count) + (SA_rem_sum/SA_rem_count) + (RI_rem_sum/RI_rem_count)
    overall_sum /= 6

    overall_hd = (LI_main_hd_sum/LI_main_hd_count) + (SA_main_hd_sum/SA_main_hd_count) + (RI_main_hd_sum/RI_main_hd_count) + (LI_rem_hd_sum/LI_rem_hd_count) + (SA_rem_hd_sum/SA_rem_hd_count) + (RI_rem_hd_sum/RI_rem_hd_count)
    overall_hd /= 6

    print(f"Whole Overall Dice : {overall_sum}")
    print(f"Whole Overall Hausdorff : {overall_hd}")








