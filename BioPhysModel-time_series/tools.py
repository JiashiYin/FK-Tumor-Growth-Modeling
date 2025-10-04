import nibabel as nib
import numpy as np


def writeNii(array, path="", affine=np.eye(4)):
    if path == "":
        path = "%dx%dx%dle.nii.gz" % np.shape(array)
    # Convert array to int32 before creating NIfTI image
    array = np.asarray(array, dtype=np.int32)
    nibImg = nib.Nifti1Image(array, affine)
    nib.save(nibImg, path)