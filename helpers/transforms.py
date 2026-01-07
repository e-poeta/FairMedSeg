import os
import numpy as np
import nibabel as nib
import matplotlib.pyplot as plt
import plotly.express as px
import pandas as pd
import json
import seaborn as sns
import SimpleITK as sitk
from skimage.filters import threshold_otsu
from skimage import exposure
from skimage import restoration


# Loading functions for the MAMMA-MIA dataset
def load_multiphase_images(patient_id, images_dir):
    patient_path = os.path.join(images_dir, patient_id)
    image_files = sorted([f for f in os.listdir(patient_path) if f.endswith('.nii.gz')])
    images = [nib.load(os.path.join(patient_path, f)).get_fdata() for f in image_files]
    return np.stack(images)

def load_segmentation(patient_id, segmentations_dir):
    seg_path = os.path.join(segmentations_dir, f'{patient_id}.nii.gz')
    seg = nib.load(seg_path).get_fdata()
    return seg

def load_patient_json(patient_id, patient_info_dir):
    json_path = os.path.join(patient_info_dir, f'{patient_id}.json')
    with open(json_path, 'r') as file:
        data = json.load(file)
    return data

def extract_json_metadata(patient_info_dir):
    records = []
    for file_name in os.listdir(patient_info_dir):
        if file_name.endswith('.json'):
            with open(os.path.join(patient_info_dir, file_name), 'r') as file:
                data = json.load(file)
                patient_id = data['patient_id']
                clinical = data['clinical_data']
                imaging = data['imaging_data']
                lesion = data['primary_lesion']
                menopausal_status = clinical.get('menopausal_status', None)
                if menopausal_status in ['pre',
                                         'pre (<6 months since LMP AND no prior bilateral ovariectomy AND not on estrogen replacement)',
                                         'pre (< 6 months since LMP AND no prior bilateral ovariectomy AND not on estrogen replacement)']:
                    menopausal_status = 'pre'
                elif menopausal_status in ['peri (6-12 months since LMP AND no prior bilateral ovariectomy AND not on estrogen replacement)']:
                    menopausal_status = 'peri'

                record = {
                    'patient_id': patient_id,
                    'age': clinical.get('age', None),
                    'menopausal_status': menopausal_status,
                    'breast_density': clinical.get('breast_density', None),
                    'pcr': lesion.get('pcr', None),
                    'tumor_subtype': lesion.get('tumor_subtype', None),
                    'bilateral': imaging.get('bilateral', None),
                    'dataset': imaging.get('dataset', None),
                    'scanner_manufacturer': imaging.get('scanner_manufacturer', None),
                    'scanner_model': imaging.get('scanner_model', None),
                    'field_strength': imaging.get('field_strength', None)
                }
                records.append(record)

    return pd.DataFrame(records)

# Visualization functions 
def visualize_phases(images):
    n_phases = images.shape[0]
    mid_slice = images.shape[-1] // 2
    fig, axes = plt.subplots(1, n_phases, figsize=(4*n_phases, 4))

    for idx in range(n_phases):
        axes[idx].imshow(images[idx, :, :, mid_slice].T, cmap='gray', origin='lower')
        axes[idx].set_title(f'Phase {idx}')
        axes[idx].axis('off')

    plt.show()
    
def visualize_segmentation_overlay(image, segmentation, slice_index=None):
    if slice_index is None:
        slice_index = segmentation.shape[-1] // 2

    plt.figure(figsize=(6, 6))
    plt.imshow(image[:, :, slice_index].T, cmap='gray', origin='lower')
    plt.imshow(np.ma.masked_where(segmentation[:, :, slice_index].T == 0, segmentation[:, :, slice_index].T), 
               cmap='autumn', alpha=0.5, origin='lower')
    plt.title('Segmentation Overlay')
    plt.axis('off')
    plt.show()

def interactive_volume_viewer(volume):
    fig = px.imshow(volume, animation_frame=2, binary_string=True, labels=dict(animation_frame="Slice"))
    fig.update_layout(width=600, height=600)
    fig.show()
    
#Statistics
def segmentation_statistics(segmentation):
    voxel_count = np.sum(segmentation > 0)
    volume_mm3 = voxel_count  # Assuming voxel spacing of 1mm³, adjust if different spacing
    print(f"Segmentation voxel count: {voxel_count}")
    print(f"Estimated volume: {volume_mm3} mm³")
    
def explore_demographics(df):
    plt.figure(figsize=(10,4))

    plt.subplot(1,2,1)
    sns.histplot(df['age'].dropna(), bins=15, kde=True)
    plt.title('Age Distribution')

    plt.subplot(1,2,2)
    sns.countplot(y='menopausal_status', data=df)
    plt.title('Menopausal Status')

    plt.tight_layout()
    plt.show()
    
def clinical_correlation(df):
    numeric_df = df[['age', 'pcr', 'field_strength']].dropna()
    sns.pairplot(numeric_df, hue='pcr')
    plt.show()

def missing_data_summary(df):
    missing = df.isnull().sum().sort_values(ascending=False)
    missing_percentage = (missing / len(df)) * 100
    summary_df = pd.DataFrame({
        'Missing Values': missing,
        'Percentage': missing_percentage
    })
    print(f"Missing Data Summary for {df.shape[0]} records:")
    display(summary_df)
    



##Preprocessing
def bias_correction_sitk(image_sitk, otsu_threshold=False, shrink_factor=0):
    """Apply N4 Bias Correction."""
    if shrink_factor:
        # N4BiasFieldCorrectionImageFilter takes too long to run, shrink image
        mask_breast = sitk.OtsuThreshold(image_sitk, 0, 1)
        shrinked_image_sitk = sitk.Shrink(image_sitk, [shrink_factor] * image_sitk.GetDimension())
        shrinked_mask_breast = sitk.Shrink(mask_breast, [shrink_factor] * mask_breast.GetDimension())
        corrector = sitk.N4BiasFieldCorrectionImageFilter()
        tmp_image = corrector.Execute(shrinked_image_sitk, shrinked_mask_breast)
        log_bias_field = corrector.GetLogBiasFieldAsImage(image_sitk)
        corrected_image_sitk = image_sitk / sitk.Exp(log_bias_field)
    else:
        initial_img = image_sitk
        # Cast to float to enable bias correction to be used
        tmp_image = sitk.Cast(image_sitk, sitk.sitkFloat64)
        # Set zeroes to a small number to prevent division by zero
        tmp_image = sitk.GetArrayFromImage(tmp_image)
        tmp_image[tmp_image == 0] = np.finfo(float).eps
        tmp_image = sitk.GetImageFromArray(tmp_image)
        tmp_image.CopyInformation(initial_img)
        if otsu_threshold:
            maskImage = sitk.OtsuThreshold(tmp_image, 0, 1)
        # Apply image bias correction using N4 bias correction
        corrector = sitk.N4BiasFieldCorrectionImageFilter()
        if otsu_threshold:
            corrected_image_sitk = corrector.Execute(tmp_image, maskImage)
        else:
            corrected_image_sitk = corrector.Execute(tmp_image)
    return corrected_image_sitk

def nlmeans_denoise_sitk(image_sitk, patch_size=5, patch_distance=6, h=0.8):
    """
    Denoises a DCE-MRI image using Non-Local Means (NLMeans) filtering.
    
    Parameters:
        image_sitk (SimpleITK.Image): Input DCE-MRI image.
        patch_size (int): Size of the patches used for denoising.
        patch_distance (int): Maximal distance in pixels where to search patches used for denoising.
        h (float): Cut-off distance (higher h means more smoothing).
        
    Returns:
        SimpleITK.Image: Denoised DCE-MRI image.
    """
    # Convert SimpleITK image to NumPy array
    image_np = sitk.GetArrayFromImage(image_sitk)
    
    # Apply Non-Local Means denoising
    denoised_np = restoration.denoise_nl_means(
        image_np,
        patch_size=patch_size,
        patch_distance=patch_distance,
        h=h,
        fast_mode=True
    )
    
    # Convert denoised NumPy array back to SimpleITK image
    denoised_image_sitk = sitk.GetImageFromArray(denoised_np)
    denoised_image_sitk.CopyInformation(image_sitk)  # Preserve original metadata
    
    return denoised_image_sitk

def clip_image_sitk(image_sitk, percentiles=[1, 99]):
    """Clip intensity range of an image.

    Parameters
    image: ITK Image
        Image to normalize
    lowerbound: float, default -1000.0
        lower bound of clipping range
    upperbound: float, default 3000.0
        lower bound of clipping range

    Returns
    -------
    image : ITK Image
        Output image.

    """
    image_array = sitk.GetArrayFromImage(image_sitk)
    image_array = image_array.ravel()
    # Drop all zeroes from array
    image_array = image_array[image_array != 0]
    lowerbound = np.percentile(image_array, percentiles[0])
    upperbound = np.percentile(image_array, percentiles[1])
    # Create clamping filter for clipping and set variables
    filter = sitk.ClampImageFilter()
    filter.SetLowerBound(float(lowerbound))
    filter.SetUpperBound(float(upperbound))

    # Execute
    clipped_image_sitk = filter.Execute(image_sitk)

    return clipped_image_sitk

def zscore_normalization_sitk(image_sitk, mean, std):
    # Z-score normalization
    array = sitk.GetArrayFromImage(image_sitk) 
    normalized_array = (array - mean) / std
    zscored_sitk = sitk.GetImageFromArray(normalized_array)
    zscored_sitk.CopyInformation(image_sitk)
    return zscored_sitk

def resample_sitk(image_sitk, new_spacing=None, new_size=None,
                   interpolator=sitk.sitkBSpline, tol=0.00001):
    # Get original settings
    original_size = image_sitk.GetSize()
    original_spacing = image_sitk.GetSpacing()
   
    # ITK can only do 3D images
    if len(original_size) == 2:
        original_size = original_size + (1, )
    if len(original_spacing) == 2:
        original_spacing = original_spacing + (1.0, )

    if new_size is None:
        # Compute output size
        new_size = [round(original_size[0]*(original_spacing[0] + tol) / new_spacing[0]),
                    round(original_size[1]*(original_spacing[0] + tol) / new_spacing[1]),
                    round(original_size[2]*(original_spacing[2] + tol) / new_spacing[2])]

    if new_spacing is None:
        # Compute output spacing
        tol = 0
        new_spacing = [original_size[0]*(original_spacing[0] + tol)/new_size[0],
                       original_size[1]*(original_spacing[0] + tol)/new_size[1],
                       original_size[2]*(original_spacing[2] + tol)/new_size[2]]

    # Set and execute the filter
    ResampleFilter = sitk.ResampleImageFilter()
    ResampleFilter.SetInterpolator(interpolator)
    ResampleFilter.SetOutputSpacing(new_spacing)
    ResampleFilter.SetSize(np.array(new_size, dtype='int').tolist())
    ResampleFilter.SetOutputDirection(image_sitk.GetDirection())
    ResampleFilter.SetOutputOrigin(image_sitk.GetOrigin())
    ResampleFilter.SetOutputPixelType(image_sitk.GetPixelID())
    ResampleFilter.SetTransform(sitk.Transform())
    try:
        resampled_image_sitk = ResampleFilter.Execute(image_sitk)
    except RuntimeError:
        # Assume the error is due to the direction determinant being 0
        # Solution: simply set a correct direction
        # print('Bad output direction in resampling, resetting direction.')
        direction = (1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0)
        ResampleFilter.SetOutputDirection(direction)
        image_sitk.SetDirection(direction)
        resampled_image_sitk = ResampleFilter.Execute(image_sitk)

    return resampled_image_sitk

def get_divisible_shape(shape, k=16):
    """Round up spatial shape to be divisible by k."""
    return tuple(((s + k - 1) // k) * k for s in shape)