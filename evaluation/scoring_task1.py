"""
scoring_segmentation.py

MAMA-MIA Challenge – Task 1: Tumour Segmentation Scoring Script

This script evaluates predicted segmentations against ground-truth masks
using Dice Similarity Coefficient (DSC) and normalized 95% Hausdorff Distance (NormHD).
Additionally, it computes a performance score, a fairness score based on demographic subgroups,
and an overall ranking score that combines both metrics.

Expected Inputs:
- Clinical and imaging metadata (Excel file with 'dataset_info' sheet)
- Ground truth and predicted tumour segmentations

Expected Outputs:
- Per-patient segmentation results (CSV)
- Group-wise fairness scores
- Heatmaps visualizing average metrics by demographic groups

Author: Lidia Garrucho, Universitat de Barcelona
Date: April 2025
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import SimpleITK as sitk
from metrics import compute_segmentation_metrics

# ----------------------------
# Utility Functions
# ----------------------------

def plot_combined_segmentation_heatmaps(fairness_varibles_df, variable_1, variable_2, variable_3, name_1, name_2, name_3, 
                                        output_plot,  metric='DSC', cmap='coolwarm'):
    """
    Generate two side-by-side heatmaps visualizing the average metric across different demographic groups.

    Parameters:
        fairness_varibles_df (DataFrame): Dataframe with patient-wise metrics and demographic variables
        variable_1, variable_2, variable_3 (str): Column names for group variables
        name_1, name_2, name_3 (str): Pretty names for heatmap axes
        output_plot (str): Path to save the resulting figure
        metric (str): The metric to visualize (e.g., 'DSC', 'NormHD')
        cmap (str): Colormap
    """
    # Heatmap 1: Average DSC by Age and Breast Density
    pivot_metric_v12 = fairness_varibles_df.pivot_table(index=variable_1, columns=variable_2, values=metric, aggfunc='mean')
    # Heatmap 2: Average DSC by Age and Menopausal Status
    pivot_metric_v13 = fairness_varibles_df.pivot_table(index=variable_1, columns=variable_3, values=metric, aggfunc='mean')
    # Create a figure with two subplots side by side
    fig, axes = plt.subplots(1, 2, figsize=(18, 6))
    # Plot Heatmap 1
    sns.heatmap(pivot_metric_v12, annot=True, cmap=cmap, fmt=".3f", cbar_kws={'label': f'Average {metric}'}, ax=axes[0])
    axes[0].set_title(f'Average {metric} by {name_1} and {name_2}')
    axes[0].set_ylabel(f'{name_1} Group')
    axes[0].set_xlabel(f'{name_2}')
    # Plot Heatmap 2
    sns.heatmap(pivot_metric_v13, annot=True, cmap=cmap, fmt=".3f", cbar_kws={'label': f'Average {metric}'}, ax=axes[1])
    axes[1].set_title(f'Average {metric} by {name_1} and {name_3}')
    axes[1].set_ylabel(f'{name_1} Group')
    axes[1].set_xlabel(f'{name_3}')
    # Invert the order of the y axis values
    axes[0].invert_yaxis()
    axes[1].invert_yaxis()
    axes[1].invert_xaxis()
    # Adjust layout to prevent overlap
    plt.tight_layout()
    # Save the combined figure
    plt.savefig(output_plot)
    plt.close()

# ----------------------------
# Main Execution
# ----------------------------

if __name__ == '__main__':

    # Settings
    HD_MAX = 150
    alpha = 0.5  # Weight for balancing performance and fairness
    selected_fairness_variables = ['age', 'menopause', 'breast_density']
    # The challenge will also evaluate the breast density variable, but it is not included in all the training data

    # Define paths (modify as needed)
    exp_path = None  # Path to the experiment output directory
    data_dir = None  # Path to the data directory
    clinical_data_xlsx = None # Path to the clinical data
    split_csv_path = f'{data_dir}/train_test_splits.csv'  # Path to the train-test split CSV file
    gt_segmentations = f'{data_dir}/segmentations/expert' # Path to the ground truth expert segmentations
    json_info_files = f'{data_dir}/patient_info_files' # Path to the patient JSON info files
    pred_segmentations = f'{exp_path}/predictions' # Path to your predicted segmentations
    output_csv = f'{exp_path}/results_postcc_task1.csv'
    output_plots_dir = f'{exp_path}/plotscc'
    

    # Read clinical data and get the fairness groups
    clinical_df = pd.read_excel(clinical_data_xlsx, sheet_name='dataset_info')
    # For fairness_varibles_df, we will drop all the clinical_df columns except the selected_fairness_variables and patient_id
    fairness_varibles_df = clinical_df[['patient_id'] + selected_fairness_variables]
    # Modify age column values mapping them by age groups
    fairness_varibles_df.loc[:, 'age'] = pd.to_numeric(fairness_varibles_df['age'], errors='coerce').fillna(-1)
    fairness_varibles_df.loc[:, 'age'] = fairness_varibles_df['age'].apply(lambda x: x if x >= 0 else None)

    # Discretize into bins
    age_bins = [0, 40, 50, 60, 70, 100]
    age_labels = ['0-40', '41-50', '51-60', '61-70', '71+']
    fairness_varibles_df['age'] = fairness_varibles_df['age'].astype("object")
    fairness_varibles_df.loc[:, 'age'] = pd.cut(fairness_varibles_df['age'], bins=age_bins, labels=age_labels).astype(str)

    # # Map the menopausal status values to 'pre', 'post', and 'unknown'
    fairness_varibles_df.loc[:, 'menopause']  = fairness_varibles_df['menopause'].fillna('unknown')
    fairness_varibles_df.loc[:, 'menopause'] = fairness_varibles_df['menopause'].apply(lambda x: 'pre' if 'peri' in x else x)
    fairness_varibles_df.loc[:, 'menopause']  = fairness_varibles_df['menopause'].apply(lambda x: 'post' if 'post' in x else x)
    fairness_varibles_df.loc[:, 'menopause']  = fairness_varibles_df['menopause'].apply(lambda x: 'pre' if 'pre' in x else x)
    
    fairness_varibles_df = fairness_varibles_df.set_index('patient_id')
  

    # Create output directories if they do not exist
    os.makedirs(pred_segmentations, exist_ok=True)
    os.makedirs(output_plots_dir, exist_ok=True)

    # Read clinical data just for validation patient IDs
    split_df = pd.read_csv(split_csv_path)
    val_ids = split_df["test_split"].dropna().tolist()  
    patient_list =val_ids
    dice_scores = []
    hausdorff_distances = []
    fairness_varibles_df = fairness_varibles_df.loc[fairness_varibles_df.index.isin(patient_list)]

    for idx, patient_id in enumerate(patient_list):
        print(f'Processing patient {idx + 1}/{len(patient_list)}: {patient_id}')
        # Read the segmentation files
        gt_file = os.path.join(gt_segmentations, f'{patient_id}.nii.gz')
        # Read it with SimpleITk and convert to numpy
        itk_image = sitk.ReadImage(gt_file)
        gt_mask = sitk.GetArrayFromImage(itk_image)

        # Read the predicted segmentation
        pred_file = os.path.join(pred_segmentations, f'{patient_id}.nii.gz')
        # Read it with SimpleITk and convert to numpy
        itk_image = sitk.ReadImage(pred_file)
        pred_mask = sitk.GetArrayFromImage(itk_image)
        
        metrics = compute_segmentation_metrics(gt_mask, pred_mask, hd_max=HD_MAX)
        fairness_varibles_df = fairness_varibles_df.copy()
        fairness_varibles_df.loc[patient_id, 'DSC'] = metrics['DSC']
        fairness_varibles_df.loc[patient_id, 'NormHD'] = metrics['NormHD']
        dice_scores.append(metrics['DSC'])
        hausdorff_distances.append(metrics['NormHD'])
        print(f'Patient {patient_id} - DSC: {metrics["DSC"]:.4f}, NormHD: {metrics["NormHD"]:.4f}')

    print(f'Average Dice: {np.mean(dice_scores):.4f}')
    print(f'Average Hausdorff distance: {np.mean(hausdorff_distances):.4f}')
    # Export results
    fairness_varibles_df.reset_index().to_csv(output_csv, index=False)

    # Compute performance score (combining Dice and NormHD)
    performance_score = 0.5 * (np.mean(dice_scores) + (1 - np.mean(hausdorff_distances)))
    print(f'Performance score: {performance_score:.4f}')
    
    # Compute fairness score across selected variables
    fairness_score_dict = {}
    for variable in selected_fairness_variables:
        # Split the fairness_varibles_df into groups based on the values of the column 'variable'
        groups = fairness_varibles_df.groupby(variable)
        # Initialize lists to store group-level metrics
        dice_scores_groups = []
        norm_hd_scores_groups = []
        # Compute group-level averages for Dice and Hausdorff Distance
        for group in groups:
            group_name = group[0]

            if group_name == 'nan':
                continue  # skip groups with NaN label

            avg_dice = group[1]['DSC'].mean()
            avg_norm_hd = group[1]['NormHD'].mean()
            if pd.notnull(avg_dice) and pd.notnull(avg_norm_hd):
                dice_scores_groups.append(avg_dice)
                norm_hd_scores_groups.append(avg_norm_hd)

        # Compute disparities: max - min across groups
        if dice_scores_groups and norm_hd_scores_groups:
            dice_disparity = max(dice_scores_groups) - min(dice_scores_groups)
            normalized_hd_disparity = max(norm_hd_scores_groups) - min(norm_hd_scores_groups)
            fairness_score = 1 - 0.5 * (dice_disparity + normalized_hd_disparity)
            # Append fairness variable and disparity to the dictionary
            fairness_score_dict[variable] = fairness_score
        else:
            fairness_score_dict[variable] = np.nan 
        
    avg_fairness_score = sum([fairness_score_dict[variable] for variable in selected_fairness_variables])
    avg_fairness_score = avg_fairness_score/len(selected_fairness_variables)
    print(f'Average fairness score: {avg_fairness_score:.4f}')

    # Final ranking score: combination of performance and fairness
    ranking_score = (1 - alpha)*performance_score + alpha*avg_fairness_score
    print(f'Ranking score: {ranking_score:.4f}')

    # Compute standard deviations
    dice_std = np.std(dice_scores)
    hausdorff_std = np.std(hausdorff_distances)

    # Create summary DataFrame
    summary_df = pd.DataFrame({
        'Average Dice': [np.mean(dice_scores)],
        'Dice Std': [dice_std],
        'Average Hausdorff': [np.mean(hausdorff_distances)],
        'Hausdorff Std': [hausdorff_std],
        'Performance Score': [performance_score],
        'Average Fairness Score': [avg_fairness_score],
        'Ranking Score': [ranking_score]
    })

    # Save to CSV
    summary_csv = os.path.join(exp_path, 'summary_results_postcc.csv')
    summary_df.to_csv(summary_csv, index=False)
    print(f'Summary results saved to {summary_csv}')

    # Save fairness heatmaps
    output_plot = os.path.join(output_plots_dir, 'heatmap_dsc_combined.png')
    plot_combined_segmentation_heatmaps(fairness_varibles_df, 'age', 'breast_density', 'menopause', 'Age', 'Breast Density',
                                        'Menopausal Status', output_plot,  metric='DSC', cmap='coolwarm_r') 
    output_plot = os.path.join(output_plots_dir, 'heatmap_normhd_combined.png')
    plot_combined_segmentation_heatmaps(fairness_varibles_df, 'age', 'breast_density', 'menopause', 'Age', 'Breast Density',
                                        'Menopausal Status', output_plot,  metric='NormHD', cmap='coolwarm')

