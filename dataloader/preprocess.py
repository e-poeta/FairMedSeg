# preprocess.py
import torchio as tio
import numpy as np
import torch
import SimpleITK as sitk
from helpers.transforms import bias_correction_sitk,nlmeans_denoise_sitk,get_divisible_shape

def preprocess_dce_mri(image_paths, label_path=None,
                       apply_bias_correction=True,
                       apply_denoising=True,
                       apply_intensity_clipping=True,
                       apply_intensity_normalization=True,
                       lower_percentile=0.5,
                       upper_percentile=99.5,
                       resample_spacing=(0.75, 0.75, 1.0),
                       target_shape=(384, 384, 160)):
    """
    TorchIO-based preprocessing pipeline for a full DCE-MRI sequence.
    Applies preprocessing in the correct order:
    Bias correction → Denoising → Clipping → Normalization → Resampling

    Args:
        image_paths (list[str]): List of NIfTI paths to DCE phases (e.g., 3-5).
        label_path (str, optional): Path to the label image.
    Returns:
        dict: A dictionary with preprocessed 'image' and optionally 'label'.
    """
    phase_images = [tio.ScalarImage(p) for p in image_paths]
    label = tio.LabelMap(label_path) if label_path else None

    # Combine phases into multi-channel image
    merged_tensor = torch.cat([img.tensor for img in phase_images], dim=0)  # (C, D, H, W)
    # subject_dict will be constructed after denoising/bias correction if used

    # Step 1–3: Bias → Denoise → Clip
    transforms = []
    # Bias correction
    if apply_bias_correction:
        corrected_images = []
        for img in image_paths:
            image_sitk = sitk.ReadImage(img)
            image_sitk = sitk.Cast(image_sitk, sitk.sitkFloat32)
            corrected = bias_correction_sitk(image_sitk, otsu_threshold=True, shrink_factor=8)
            corrected_array = sitk.GetArrayFromImage(corrected)  # (H, W, D)
            corrected_array = np.transpose(corrected_array, (2, 1, 0))  # (D, H, W)
            corrected_tensor = torch.tensor(corrected_array).unsqueeze(0)
            corrected_images.append(corrected_tensor)
        merged_tensor = torch.cat(corrected_images, dim=0)
       
        
    if not apply_bias_correction:
        merged_tensor = torch.cat([img.tensor for img in phase_images], dim=0)
        
    if apply_denoising:
        denoised_images = []
        for c in range(merged_tensor.shape[0]):
            image_array = merged_tensor[c].numpy()
            image_array = np.transpose(image_array, (2, 1, 0))  # Back to (H, W, D) for SimpleITK
            image_sitk = sitk.GetImageFromArray(image_array)
            denoised = nlmeans_denoise_sitk(image_sitk)
            denoised_array = sitk.GetArrayFromImage(denoised)
            denoised_array = np.transpose(denoised_array, (2, 1, 0))  # Back to (D, H, W)
            denoised_tensor = torch.tensor(denoised_array).unsqueeze(0)
            denoised_images.append(denoised_tensor)
        merged_tensor = torch.cat(denoised_images, dim=0)           
    merged_tensor = merged_tensor.float()   
    subject_image = tio.ScalarImage(tensor=merged_tensor, affine=phase_images[0].affine)

    subject_dict = {'image': subject_image}
    if label is not None:
        subject_dict['label'] = label

    subject = tio.Subject(subject_dict)

    if apply_intensity_clipping:
        transforms.append(tio.Clamp(out_min=lower_percentile, out_max=upper_percentile))

    subject = tio.Compose(transforms)(subject)

    # Step 4: Normalize over all channels
    if apply_intensity_normalization:
        image_tensor = subject['image'].tensor  # shape: (C, D, H, W)
        mean = image_tensor.mean()
        std = image_tensor.std()
        subject['image'].set_data((image_tensor - mean) / std)

    # Step 5: Resample
    if resample_spacing is not None:
        subject = tio.Resample(resample_spacing, image_interpolation='bspline', label_interpolation='nearest')(subject)
        
    # Step 6: Resize to fixed shape for compatibility
    subject = tio.Resize(target_shape)(subject) #(320, 320, 128)
    
    return subject
