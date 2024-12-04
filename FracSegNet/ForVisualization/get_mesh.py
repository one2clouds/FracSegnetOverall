import vedo 
import SimpleITK as sitk
from typing import Optional 


def make_isotropic(
    img: sitk.Image,
    spacing: Optional[float] = None,
    interpolator=sitk.sitkNearestNeighbor,
):
    """
    Resample `img` so that the voxel is isotropic with given physical spacing
    The image volume is shrunk or expanded as necessary to represent the same physical space.
    Use sitk.sitkNearestNeighbour while resampling Label images,
    when spacing is not supplied by the user, the highest resolution axis spacing is used
    """
    # keep the same physical space, size may shrink or expand

    print(" make isotropic started")

    if spacing is None:
        spacing = min(list(img.GetSpacing()))

    resampler = sitk.ResampleImageFilter()
    new_size = [
        round(old_size * old_spacing / spacing)
        for old_size, old_spacing in zip(img.GetSize(), img.GetSpacing())
    ]
    output_spacing = [spacing] * len(img.GetSpacing())

    resampler.SetOutputSpacing(output_spacing)
    resampler.SetSize(new_size)
    resampler.SetOutputDirection(img.GetDirection())
    resampler.SetOutputOrigin(img.GetOrigin())
    resampler.SetTransform(sitk.Transform())
    resampler.SetInterpolator(interpolator)
    # resampler.SetDefaultPixelValue(img.GetPixelIDValue())
    resampler.SetDefaultPixelValue(0)

    print(" upto make isotropic")

    return resampler.Execute(img)


def get_volume(filename, largest_component=False, isotropic=True, reorient=False, orientation='PIR') -> vedo.Volume:
    print("get_volume started")
    sitk_volume = sitk.ReadImage(filename)
    if reorient:
        sitk_volume = sitk.DICOMOrient(sitk_volume, orientation)
    if largest_component:
        # get largest connected component
        sitk_volume = sitk.RelabelComponent(sitk.ConnectedComponent(
            sitk.Cast(sitk_volume, sitk.sitkUInt8),
        ), sortByObjectSize=True) == 1
    if isotropic:
        sitk_volume = make_isotropic(sitk_volume, 1.0)
    np_volume = vedo.Volume(sitk.GetArrayFromImage(sitk_volume))
    print("upto get volume")
    return np_volume


def get_mesh_from_segmentation(filename: str, largest_component=False, flying_edges=True, decimate=False, decimation_ratio=1.0, isosurface_value=1.0, smooth=20, reorient=False, orientation='PIR') -> vedo.Mesh:
    print("Before get mesh from segmentation")
    np_volume = get_volume(filename, largest_component, reorient=reorient, orientation=orientation)
    # isosurface_values = get_segmentation_labels(sitk_volume)
    mesh_obj: vedo.Mesh = np_volume.isosurface(value=isosurface_value-0.1, flying_edges=flying_edges)
    print("before mesh object fill hole")
    print(mesh_obj)
    # mesh_obj = mesh_obj.fill_holes()
    mesh_obj.smooth(niter=smooth)
    print("before decimate")
    if decimate:
        mesh_obj = mesh_obj.decimate(fraction=decimation_ratio)
    print(f"upto get mesh from segmentation {len(mesh_obj.vertices)}")

    return mesh_obj.cap()