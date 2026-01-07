import torch
import torch.nn as nn
import torch.nn.functional as F
from nnunetv2.training.loss_functions.dice import DC_and_CE_loss
from monai.metrics import DiceMetric
from divexplorer import DivergenceExplorer


class DivergenceFairLoss(nn.Module):
    def __init__(self, sensitive_attributes, alpha=0.5, divergence_scores=None):
        super().__init__()
        self.alpha = alpha
        self.cross_entropy = nn.CrossEntropyLoss(reduction='none')  # we'll weight manually
        self.seg_loss = DC_and_CE_loss()  # original Dice + CE loss
        self.divergence_scores = divergence_scores or {}
        self.sensitive_columns = sensitive_attributes
  

    def _compute_dice_score(pred, true):
        """
        Compute the Dice score between predicted and true labels.
        
        Args:
            pred (numpy.ndarray): Predicted labels.
            true (numpy.ndarray): True labels.
            
        Returns:
            float: The Dice score.
        """
        dice_metric = DiceMetric(include_background=False, reduction="mean")
        pred_tensor = torch.tensor(pred).unsqueeze(0)  # Add batch dimension
        true_tensor = torch.tensor(true).unsqueeze(0)  # Add batch dimension
        return dice_metric(pred_tensor, true_tensor).item()


    def compute_the_subgroups(net_output, target, metadata):
        """
        Compute the subgroups based on the metadata and net output.
        
        Args:
            net_output (torch.Tensor): The output from the neural network.
            metadata (dataframe): Dataframe of metadata items for each sample.
        
        Returns:
            dict: A dictionary where keys are subgroup identifiers and values are their corresponding indices.
        
        """

        # 1. Extract the subgroups using DivExplorer 
        # Add the column prediction_dice_score calculating the Dice score from MONAI for each sample
        dice_scores = []
        for i in range(len(metadata)):
            pred = net_output[i].argmax(dim=0).cpu().numpy()  # Assuming net_output is a batch of predictions
            true = target[i].cpu().numpy()
            dice_score = _compute_dice_score(pred, true)
            dice_scores.append(dice_score)

        metadata['prediction_dice_score'] = dice_scores

        fp_explorer = DivergenceExplorer(metadata)
        subgroups = fp_explorer.get_pattern_divergence(
            quantitative_outcomes=['prediction_dice_score'],
            min_support = 0.005,
            attributes= self.sensitive_columns
        )     

    def compute_weights(self, metadata):
        weights = []
        for item in metadata:
            keys = self.get_subgroup_keys(item)
            max_div = max(abs(self.divergence_scores.get(k, 0.0)) for k in keys)
            weights.append(max_div)
        return torch.tensor(weights, dtype=torch.float32, device='cuda')

    def forward(self, net_output, target, metadata:dataframe):
        # 1. Main segmentation loss (Dice + CE)
        l_seg = self.seg_loss(net_output, target)

        # 2. Compute pixel-wise CE
        ce_per_pixel = self.cross_entropy(net_output, target)  # shape: (B, D, H, W)

        # 3. Compute sample weights
        weights = self.compute_weights(metadata)  # shape: (B,)
        weights = weights.view(-1, 1, 1, 1)  # broadcast to per-pixel

        # 4. Apply sample weights
        l_div = (weights * ce_per_pixel).mean()

        # 5. Final loss
        return self.alpha * l_seg + (1 - self.alpha) * l_div
