from metrics.metrics import compute_segmentation_metrics_fast, compute_segmentation_metrics
import torch 
import numpy as np

def compute_performance_score(outputs, labels, columns_of_interest, val_metadata):
    """
    Computes the challenge scores based on the provided challenge results.

    Args:
        outputs (torch.Tensor): The model outputs.
        labels (torch.Tensor): The ground truth labels.
        wandb_run (wandb.Run, optional): The wandb run object for logging.

    Returns:
        dict: A dictionary with challenge IDs as keys and their corresponding scores as values.
    """
    alpha = 0.5  # Weighting factor - chosen by the challenge organizers
    
    fairness_score_dict = {}

    # Compute the segmentation metrics for each sample
    probs = torch.nn.functional.softmax(outputs, dim=1)
    preds = probs.argmax(dim=1, keepdim=True)
    
    for i in range(len(labels)):
        pred_np = (preds[i, 0].cpu().numpy() == 1).astype(np.uint8).squeeze()
        label_np = (labels[i].cpu().numpy() == 1).astype(np.uint8).squeeze()

        metrics = compute_segmentation_metrics(pred_np, label_np)
        val_metadata.loc[val_metadata.index[i], 'DSC'] = metrics['DSC']
        val_metadata.loc[val_metadata.index[i], 'NormHD'] = metrics['NormHD']

    # Return the average performance score
    performance_score = 0.5 * (val_metadata['DSC'].mean() + (1 - val_metadata['NormHD'].mean()))


    return performance_score, val_metadata, val_metadata['DSC'].mean() , val_metadata['NormHD'].mean()


def compute_fairness_ranking_score(val_metadata, columns_of_interest, avg_performance_score, alpha=0.5):
    
    fairness_score_dict = {}
    for column in columns_of_interest:
        
        groups = val_metadata.groupby(column)

        dice_scores_groups = []
        norm_hd_scores_groups = []

        for group in groups:
            group_name = group[0]
            avg_dice = group[1]['DSC'].mean()
            avg_norm_hd = group[1]['NormHD'].mean()
            dice_scores_groups.append(avg_dice)
            norm_hd_scores_groups.append(avg_norm_hd)

        # Compute disparities: max - min across groups
        dice_disparity = max(dice_scores_groups) - min(dice_scores_groups)
        normalized_hd_disparity = max(norm_hd_scores_groups) - min(norm_hd_scores_groups)
        fairness_score = 1 - 0.5 * (dice_disparity + normalized_hd_disparity)
        # Append fairness variable and disparity to the dictionary
        fairness_score_dict[column] = fairness_score

    avg_fairness_score = sum([fairness_score_dict[column] for column in columns_of_interest])
    avg_fairness_score = avg_fairness_score/len(columns_of_interest)
    #print(f'Average fairness score: {avg_fairness_score:.4f}')

    # Compute the ranking score 


    # Final ranking score: combination of performance and fairness
    ranking_score = (1 - alpha)*avg_performance_score + alpha*avg_fairness_score

    return avg_fairness_score, ranking_score