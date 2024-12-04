import shutil
from batchgenerators.utilities.file_and_folder_operations import join, subfolders, maybe_mkdir_p, save_json
import os
import tqdm 
from pathlib import Path

# export nnUNet_raw_data_base="/mnt/Enterprise2/shirshak/NewFracSegNet/FracSegNet/dataset/nnUNet_raw"
# export nnUNet_preprocessed="/mnt/Enterprise2/shirshak/NewFracSegNet/FracSegNet/dataset/nnUNet_preprocessed"
# export RESULTS_FOLDER="/mnt/Enterprise2/shirshak/NewFracSegNet/FracSegNet/dataset/nnUNet_results"


# we provide location for nnUNet_raw_data inside nnUNet_raw
# base = "/mnt/Enterprise2/shirshak/NewFracSegNet/FracSegNet/dataset/nnUNet_raw/nnUNet_raw_data"
base = "/mnt/Enterprise2/shirshak/NewFracSegNet/FracSegNet/dataset/nnUNet_raw/nnUNet_raw_data"

task_id = 602
task_name = "CT_PelvicFrac150"

foldername = "Task%03.0d_%s" % (task_id, task_name)

out_base = join(base, foldername)
imagestr = join(out_base, "imagesTr")
labelstr = join(out_base, "labelsTr")

maybe_mkdir_p(imagestr)

maybe_mkdir_p(labelstr)


folder_raw = Path("/mnt/Enterprise2/shirshak/PENGWIN_TASK/PENGWIN_CT/nifti_fragments_LI_SA_RI")

n_train_case = 0
n_test_case = 0


for filename in sorted(os.listdir(os.path.join(folder_raw, "images")))[:6]:
    print(filename)
    if "_RI_image.nii.gz" in filename:
        # print(os.path.join(folder_raw, "images", filename))
        # print(join(imagestr, filename.split("_RI_image.nii.gz")[0] + "_RI_0000.nii.gz"))
        shutil.copy(os.path.join(folder_raw, "images", filename), join(imagestr, filename.split("_RI_image.nii.gz")[0] + "_RI_0000.nii.gz"))

    if "_LI_image.nii.gz" in filename:
        shutil.copy(os.path.join(folder_raw, "images", filename), join(imagestr, filename.split("_LI_image.nii.gz")[0] + "_LI_0000.nii.gz"))

    if "_SA_image.nii.gz" in filename:
        shutil.copy(os.path.join(folder_raw, "images", filename), join(imagestr, filename.split("_SA_image.nii.gz")[0] + "_SA_0000.nii.gz"))

# We must make dataset into fixed classes so, we take preprocessed_nifti_fragments_LI_SA_RI and not nifti_fragments_LI_SA_RI because while running code 
# nnUNet_plan_and_preprocessing it gave error as there are classes 11 12 21 23 etc so we need fixed class, 
# and in preprocessed filder there are fixed class

folder_raw_for_labels = Path("/mnt/Enterprise2/shirshak/PENGWIN_TASK/PENGWIN_CT/preprocessed_nifti_fragments_LI_SA_RI")

for filename in sorted(os.listdir(os.path.join(folder_raw_for_labels, "labels")))[:6]:
    print(filename)
    if "_RI_label.nii.gz" in filename:
        # print(os.path.join(folder_raw_for_labels, "labels", filename))
        # print(join(labelstr, filename.split("_RI_label.nii.gz")[0] + "_RI.nii.gz"))
        shutil.copy(os.path.join(folder_raw_for_labels, "labels", filename), join(labelstr, filename.split("_RI_label.nii.gz")[0] + "_RI.nii.gz"))
        n_train_case += 1
    if "_LI_label.nii.gz" in filename:
        shutil.copy(os.path.join(folder_raw_for_labels, "labels", filename), join(labelstr, filename.split("_LI_label.nii.gz")[0] + "_LI.nii.gz"))
        n_train_case += 1
    if "_SA_label.nii.gz" in filename:
        shutil.copy(os.path.join(folder_raw_for_labels, "labels", filename), join(labelstr, filename.split("_SA_label.nii.gz")[0] + "_SA.nii.gz"))
        n_train_case += 1



train_patient_names = os.listdir(labelstr)

json_dict = {}
json_dict['name'] = "CT fracture all 54 res"
json_dict['description'] = "CT fracture all 54 res"
json_dict['tensorImageSize'] = "4D"
json_dict['reference'] = "CT fracture all 54 res"
json_dict['licence'] = ""
json_dict['release'] = "2022.11.03"
json_dict['modality'] = {
    "0": "CT",
}
json_dict['labels'] = {
    "0": "background",
    "1": "main fracture segment",
    "2": "segment 2",
    #"3": "segment 3"
}

json_dict['numTraining'] = n_train_case
json_dict['numTest'] = n_test_case
json_dict['training'] = [{'image': "./imagesTr/%s" % i, "label": "./labelsTr/%s" % i} for i in
                         train_patient_names]
json_dict['test'] = '' #["./imagesTs/%s" % i for i in test_patient_names]

save_json(json_dict, os.path.join(out_base, "dataset.json"))

# export nnUNet_raw_data_base="/mnt/Enterprise2/shirshak/NewFracSegNet/FracSegNet/dataset/nnUNet_raw"
# export nnUNet_preprocessed="/mnt/Enterprise2/shirshak/NewFracSegNet/FracSegNet/dataset/nnUNet_preprocessed"
# export RESULTS_FOLDER="/mnt/Enterprise2/shirshak/NewFracSegNet/FracSegNet/dataset/nnUNet_results"