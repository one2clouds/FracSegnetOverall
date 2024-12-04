import sys 
from nnUNet.nnunetv2.inference.predict_from_raw_data import nnUNetPredictor
import os 
import torch 
from nnUNet.nnunetv2.imageio.simpleitk_reader_writer import SimpleITKIO
from batchgenerators.utilities.file_and_folder_operations import join
import numpy as np
import SimpleITK as sitk 
from glob import glob 

from monai.metrics import DiceMetric

from pathlib import Path 

def  make_discrete_data(data):
    # print(data.unique())
    data = torch.from_numpy(data)
    discreted_data = torch.zeros_like(data)

    discreted_data[(data>0) & (data<=10)] = 1
    discreted_data[(data>10) & (data<=20)] = 2
    discreted_data[(data>20) & (data<=30)] = 3
    return discreted_data

def run():
    # WE DON'T NEED TO CHANGE DIRN OF IMG HERE becoz monai transforms will do it.
    print("Just Started")
    sys.stdout.write('Just started \n')


    # For Anatomical Model USING NNunet Anatomical Model
    train_images_names = sorted(glob("/mnt/Enterprise2/shirshak/PENGWIN_TASK/PENGWIN_CT/part*/*.mha"))[80:]
    train_label_names = sorted(glob("/mnt/Enterprise2/shirshak/PENGWIN_TASK/PENGWIN_CT/labels/*.mha"))[80:]

    for (train_image_name, train_label_name) in zip(train_images_names, train_label_names):
        # print('-'*80)
        # print(os.path.split(train_image_name)[1].split('.mha')[0])
        # print('-'*80)
        img, props = SimpleITKIO().read_images([train_image_name])

        sample_img = sitk.ReadImage(train_image_name)
        sample_label_arr = make_discrete_data(sitk.GetArrayFromImage(sitk.ReadImage(train_label_name)))

        predictor = nnUNetPredictor(
            tile_step_size=0.5,
            use_gaussian=True,
            use_mirroring=True,
            perform_everything_on_device=True,
            device=torch.device('cuda', 0),
            verbose=True,
            verbose_preprocessing=True,
            allow_tqdm=True
        )
        predictor.initialize_from_trained_model_folder(
            join('/mnt/Enterprise2/shirshak/NewFracSegNet/Anatomical_Model/dataset2/nnUNet_results', 'Dataset605_CT_PelvicFrac150/nnUNetTrainer__nnUNetPlans__3d_lowres'),
            use_folds=(1,),
            checkpoint_name='checkpoint_best.pth',
        )

        predicted_segmentation, class_probabilities = predictor.predict_single_npy_array(img, props, None, None, True)
        print(np.unique(predicted_segmentation, return_counts=True)) # [0 1 2 3]

        pred_img = sitk.GetImageFromArray(predicted_segmentation)
        pred_img.SetDirection(sample_img.GetDirection())
        pred_img.SetSpacing(sample_img.GetSpacing())
        pred_img.SetOrigin(sample_img.GetOrigin())

        img_path = Path(f"/mnt/Enterprise2/shirshak/NewFracSegNet/Anatomical_Model/output_folder/{str(os.path.split(train_image_name)[1].split('.mha')[0])}.nii.gz")
        print(img_path)

        sitk.WriteImage(pred_img, img_path)

        print('-'*80)

        print(torch.as_tensor(predicted_segmentation).unsqueeze(0).unsqueeze(0).shape)
        print(torch.as_tensor(sample_label_arr).unsqueeze(0).unsqueeze(0).shape)

        dice_metric_calc = DiceMetric(reduction="mean")(torch.as_tensor(predicted_segmentation).unsqueeze(0).unsqueeze(0), torch.as_tensor(sample_label_arr).unsqueeze(0).unsqueeze(0))
        print(dice_metric_calc)
        print('-'*80)





if __name__ == "__main__":
    # import gdown 
    # gdown.download("https://drive.google.com/uc?id=1TDlfk8tGhMRIvk86nG8yspna2ZZ1-Lf0", join(RESOURCE_PATH, 'model_best.model'))
    raise SystemExit(run())
