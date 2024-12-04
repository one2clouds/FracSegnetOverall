import sys 
import nnunet
import os 
import torch 
from nnunet.training.network_training.nnUNetTrainer import nnUNetTrainer
from batchgenerators.utilities.file_and_folder_operations import join
from batchgenerators.utilities.file_and_folder_operations import load_pickle
from nnunet.training.model_restore import recursive_find_python_class #, load_model_and_checkpoint_files, restore_model

from collections import OrderedDict
import numpy as np
import SimpleITK as sitk 
from glob import glob 

from monai.metrics import DiceMetric
from pathlib import Path 
from nnunet.training.data_augmentation.default_data_augmentation import default_3D_augmentation_params


def restore_model(pkl_file, checkpoint=None, train=False, fp16=None):
    """
    This is a utility function to load any nnUNet trainer from a pkl. It will recursively search
    nnunet.trainig.network_training for the file that contains the trainer and instantiate it with the arguments saved in the pkl file. If checkpoint
    is specified, it will furthermore load the checkpoint file in train/test mode (as specified by train).
    The pkl file required here is the one that will be saved automatically when calling nnUNetTrainer.save_checkpoint.
    :param pkl_file:
    :param checkpoint:
    :param train:
    :param fp16: if None then we take no action. If True/False we overwrite what the model has in its init
    :return:
    """
    info = load_pickle(pkl_file)
    init = info['init']
    name = info['name']
    search_in = join(nnunet.__path__[0], "training", "network_training")
    tr = recursive_find_python_class([search_in], name, current_module="nnunet.training.network_training")


    if tr is None:
        raise RuntimeError("Could not find the model trainer specified in checkpoint in nnunet.trainig.network_training. If it "
                           "is not located there, please move it or change the code of restore_model. Your model "
                           "trainer can be located in any directory within nnunet.trainig.network_training (search is recursive)."
                           "\nDebug info: \ncheckpoint file: %s\nName of trainer: %s " % (checkpoint, name))
    assert issubclass(tr, nnUNetTrainer), "The network trainer was found but is not a subclass of nnUNetTrainer. " \
                                          "Please make it so!"


    trainer = tr(*init)

    if fp16 is not None:
        trainer.fp16 = fp16

    trainer.process_plans(info['plans'])
    saved_model = torch.load(checkpoint, map_location=torch.device('cpu'))

    new_state_dict = OrderedDict()

    for k, value in saved_model['state_dict'].items():
        key = k
        if key.startswith('module.'):
            key = key[7:]
        new_state_dict[key] = value

    trainer.initialize_network()
    trainer.network.load_state_dict(new_state_dict)
    return trainer

def run():
    # WE DON'T NEED TO CHANGE DIRN OF IMG HERE becoz monai transforms will do it.
    print("Just Started")
    sys.stdout.write('Just started \n')

    pkl = join(Path("/mnt/Enterprise2/shirshak/NewFracSegNet/FracSegNet/dataset/nnUNet_results/nnUNet/3d_fullres/Task601_CT_PelvicFrac150/nnUNetTrainerV2__nnUNetPlansv2.1/all"), 'model_best.model.pkl')
    checkpoint = pkl[:-4]
    train = False
    trainer = restore_model(pkl, checkpoint, train)
    # We also have to put value of data_aug_params from nnunet/training/data_augumentation/default_data_augumentation.py, and since our model is 3d full res model 
    trainer.data_aug_params = default_3D_augmentation_params
    # For Anatomical Model USING NNunet Anatomical Model 
    
    train_images_names = sorted(glob('/mnt/Enterprise2/shirshak/NewFracSegNet/FracSegNet/dataset/nnUNet_raw/nnUNet_raw_data/Task601_CT_PelvicFrac150/imagesTs/*.nii.gz'))
    train_label_names = sorted(glob('/mnt/Enterprise2/shirshak/NewFracSegNet/FracSegNet/dataset/nnUNet_raw/nnUNet_raw_data/Task601_CT_PelvicFrac150/labelsTs/*.nii.gz'))

    
    # predicted_arr = sitk.GetArrayFromImage(sitk.ReadImage('/mnt/Enterprise2/shirshak/NewFracSegNet/FracSegNet/output_folder_doing_preprocessing/output.nii.gz'))
    # gt_arr = sitk.GetArrayFromImage(sitk.ReadImage('/mnt/Enterprise2/shirshak/NewFracSegNet/FracSegNet/dataset/nnUNet_raw/nnUNet_raw_data/Task601_CT_PelvicFrac150/labelsTs/081_LI.nii.gz'))

    # print(DiceMetric(include_background=False)(torch.as_tensor(gt_arr).unsqueeze(0).unsqueeze(0), torch.as_tensor(predicted_arr).unsqueeze(0).unsqueeze(0)))

    for (train_image_name, train_label_name) in zip(train_images_names, train_label_names):
        # print('-'*80)
        # print(os.path.split(train_image_name)[1].split('.mha')[0])
        # print('-'*80)

        print(os.path.split(train_image_name)[1].split('.nii.gz')[0])
        trainer.preprocess_predict_nifti([train_image_name], output_file=f"/mnt/Enterprise2/shirshak/NewFracSegNet/FracSegNet/new_output_folder_doing_preprocessing/{os.path.split(train_image_name)[1].split('.nii.gz')[0]}.nii.gz", softmax_ouput_file=None, mixed_precision=True)


if __name__ == "__main__":
    # import gdown 
    # gdown.download("https://drive.google.com/uc?id=1TDlfk8tGhMRIvk86nG8yspna2ZZ1-Lf0", join(RESOURCE_PATH, 'model_best.model'))
    raise SystemExit(run())
