import ast
from multiprocessing.pool import Pool
import numpy as np
from scipy.ndimage import label
import SimpleITK as sitk
from nnunet.utilities.sitk_stuff import copy_geometry
from batchgenerators.utilities.file_and_folder_operations import *

from typing import Union, Tuple, List
from batchgenerators.utilities.file_and_folder_operations import load_json, join
import argparse


def remove_all_but_largest_component_from_segmentation(segmentation: np.ndarray,
                                                       for_which_classes: Union[int, Tuple[int, ...],List[Union[int, Tuple[int, ...]]]],
                                                       background_label: int = 0) -> np.ndarray:
    
    if for_which_classes is None:
        for_which_classes, classes_counts = np.unique(segmentation, return_counts=True)
        for_which_classes = for_which_classes[for_which_classes > 0]
        classes_counts = classes_counts[1:] # remove the count of pixels of 0th background
    # print(for_which_classes)
    # print(classes_counts)
    max_class_count = -1
    for class_count in classes_counts:
        if class_count> max_class_count:
            max_class_count = class_count
    
    for unique_class_name, class_count in zip(for_which_classes, classes_counts):
        if class_count < max_class_count:
            segmentation[segmentation == unique_class_name] = 2
        if class_count == max_class_count:
            segmentation[segmentation == unique_class_name] = 1
    
    # print(np.unique(segmentation))
    return segmentation


def load_remove_save(input_file: str, output_file: str, for_which_classes: list,
                     minimum_valid_object_size: dict = None) -> None:
    # Only objects larger than minimum_valid_object_size will be removed. Keys in minimum_valid_object_size must
    # match entries in for_which_classes
    img_in = sitk.ReadImage(input_file)
    img_npy = sitk.GetArrayFromImage(img_in)

    image = remove_all_but_largest_component_from_segmentation(img_npy, for_which_classes)
    # print(input_file, "kept:", kept_size)
    img_out_itk = sitk.GetImageFromArray(image)
    img_out_itk = copy_geometry(img_out_itk, img_in)
    sitk.WriteImage(img_out_itk, output_file)


def load_postprocessing(json_file):
    '''
    loads the relevant part of the pkl file that is needed for applying postprocessing
    :param pkl_file:
    :return:
    '''
    a = load_json(json_file)
    if 'min_valid_object_sizes' in a.keys():
        min_valid_object_sizes = ast.literal_eval(a['min_valid_object_sizes'])
    else:
        min_valid_object_sizes = None
    return a['for_which_classes'], min_valid_object_sizes


def apply_postprocessing_to_folder(input_folder: str, output_folder: str, for_which_classes: list,
                                   min_valid_object_size:dict=None, num_processes=8):
    """
    applies removing of all but the largest connected component to all niftis in a folder
    :param min_valid_object_size:
    :param min_valid_object_size:
    :param input_folder:
    :param output_folder:
    :param for_which_classes:
    :param num_processes:
    :return:
    """
    maybe_mkdir_p(output_folder)
    p = Pool(num_processes)
    nii_files = subfiles(input_folder, suffix=".nii.gz", join=False)
    input_files = [join(input_folder, i) for i in nii_files]
    out_files = [join(output_folder, i) for i in nii_files]
    results = p.starmap_async(load_remove_save, zip(input_files, out_files, [for_which_classes] * len(input_files),
                                                    [min_valid_object_size] * len(input_files)))
    res = results.get()
    p.close()
    p.join()


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--input', required=True)
    parser.add_argument('-o', '--output', required=True)
    args = parser.parse_args()

    apply_postprocessing_to_folder(args.input, args.output, for_which_classes = None)



# python3 separate_into_main_and_remaining_fragments.py -i /mnt/Enterprise2/shirshak/PENGWIN_TASK/PENGWIN_CT/nifti_fragments_LI_SA_RI/labels -o /mnt/Enterprise2/shirshak/PENGWIN_TASK/PENGWIN_CT/preprocessed_nifti_fragments_LI_SA_RI/labels





















    # remove_all_but_largest_component_from_segmentation(sitk.GetArrayFromImage(sitk.ReadImage(os.path.join(input_folder, "001_LI_label.nii.gz"))), for_which_classes=None)

    
# def remove_all_but_the_largest_connected_component(image: np.ndarray, for_which_classes: list, volume_per_voxel: float,
#                                                    minimum_valid_object_size: dict = None):
#     """
#     removes all but the largest connected component, individually for each class
#     :param image:
#     :param for_which_classes: can be None. Should be list of int. Can also be something like [(1, 2), 2, 4].
#     Here (1, 2) will be treated as a joint region, not individual classes (example LiTS here we can use (1, 2)
#     to use all foreground classes together)
#     :param minimum_valid_object_size: Only objects larger than minimum_valid_object_size will be removed. Keys in
#     minimum_valid_object_size must match entries in for_which_classes
#     :return:
#     """
#     if for_which_classes is None:
#         for_which_classes = np.unique(image)
#         for_which_classes = for_which_classes[for_which_classes > 0]

#     assert 0 not in for_which_classes, "cannot remove background"
#     largest_removed = {}
#     kept_size = {}
#     for c in for_which_classes:
#         if isinstance(c, (list, tuple)):
#             c = tuple(c)  # otherwise it cant be used as key in the dict
#             mask = np.zeros_like(image, dtype=bool)
#             for cl in c:
#                 mask[image == cl] = True
#         else:
#             mask = np.zeros_like(image, dtype=bool)
#             mask = image == c
#         # get labelmap and number of objects
#         lmap, num_objects = label(mask.astype(int))

#         # collect object sizes
#         object_sizes = {}
#         for object_id in range(1, num_objects + 1):
#             object_sizes[object_id] = (lmap == object_id).sum() * volume_per_voxel

#         largest_removed[c] = None
#         kept_size[c] = None

#         if num_objects > 0:
#             # we always keep the largest object. We could also consider removing the largest object if it is smaller
#             # than minimum_valid_object_size in the future but we don't do that now.
#             maximum_size = max(object_sizes.values())
#             kept_size[c] = maximum_size

#             for object_id in range(1, num_objects + 1):
#                 # we only remove objects that are not the largest
#                 if object_sizes[object_id] != maximum_size:
#                     # we only remove objects that are smaller than minimum_valid_object_size
#                     remove = True
#                     if minimum_valid_object_size is not None:
#                         remove = object_sizes[object_id] < minimum_valid_object_size[c]
#                     if remove:
#                         image[(lmap == object_id) & mask] = 0
#                         if largest_removed[c] is None:
#                             largest_removed[c] = object_sizes[object_id]
#                         else:
#                             largest_removed[c] = max(largest_removed[c], object_sizes[object_id])
#     return image, largest_removed, kept_size
