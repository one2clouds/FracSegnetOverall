from pathlib import Path 

import vedo 
import os 

from get_mesh import get_mesh_from_segmentation


def process_hip_bones(nifti_filename, offscreen=False, screenshot_out_dir="."):
    splitted_name = os.path.split(nifti_filename)[1].split(".mha")[0]

    mesh_obj = get_mesh_from_segmentation(nifti_filename, largest_component=False)
    print("after get_mesh from segmentation")
    vedo.show(
        mesh_obj.opacity(0.5),
        offscreen=offscreen,
    )
    print("after vedo show ")
    out_filename = Path(nifti_filename).with_suffix(".png")
    print("after submitting to out_filename")
    vedo.screenshot(str(Path(screenshot_out_dir) / out_filename.name))
    print("after doing screenshot")
    if offscreen:
        vedo.close()


if __name__ == "__main__":
    process_hip_bones(nifti_filename="/mnt/Enterprise2/shirshak/PENGWIN_TASK/PENGWIN_CT/labels/001.mha",offscreen=False, screenshot_out_dir="/home/shirshak/FracSegNet/FOR_VISUALIZATION/visualized_images/")