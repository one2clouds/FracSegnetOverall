import torch 
import SimpleITK as sitk  


dismap_img = sitk.ReadImage('/mnt/Enterprise2/shirshak/NewFracSegNet/FracSegNet/distance_map_zzz_tests/distance_map.nii.gz')
dismap_arr = sitk.GetArrayFromImage(dismap_img)
dismap_arr = torch.as_tensor(dismap_arr)

print(dismap_arr.shape) # (401, 512, 512)
dismap_arr = dismap_arr.unsqueeze(0).unsqueeze(0)
dismap_arr_new = torch.nn.functional.interpolate(dismap_arr, size=(128,128,128)).squeeze(0).squeeze(0)
print(dismap_arr_new.shape)
dismap_new_img = sitk.GetImageFromArray(dismap_arr_new)
dismap_new_img.SetSpacing(dismap_img.GetSpacing())
dismap_new_img.SetDirection(dismap_img.GetDirection())
dismap_new_img.SetOrigin(dismap_img.GetOrigin())
sitk.WriteImage(dismap_new_img, '/mnt/Enterprise2/shirshak/NewFracSegNet/FracSegNet/distance_map_zzz_tests/dismap_new.nii.gz')