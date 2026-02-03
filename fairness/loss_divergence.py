import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from tqdm import tqdm
from monai.losses import DiceCELoss
from monai.metrics import DiceMetric
from divexplorer import DivergenceExplorer
from metrics.metrics import compute_segmentation_metrics, compute_segmentation_metrics_fast
import torchio as tio


class DivergenceFairLoss(nn.Module):
    def __init__(self, device, min_support=0.005, alpha=0.5, divergence_scores=None, sensitive_columns=None):
        super().__init__()
        self.alpha = alpha
        self.seg_loss = DiceCELoss(to_onehot_y=True,softmax=True)  # Dice + CE loss
        self.columns_of_interest = ['age', 'breast_density', 'menopause']  # columns of interest for divergence
        self.performance_name = 'performance_score'  # name of the performance score in metadata
        self.device = device
        self.dice_metric = DiceMetric(include_background=False, reduction="mean")
        self.min_support = min_support  # minimum support for subgroup discovery
       
    def compute_performance_score(self, model, val_loader):

        performance_scores = []
        self.dice_metric.reset()  # Reset the Dice metric


        for idx, batch in enumerate(tqdm(val_loader, desc=f"Loss Validation", total=len(val_loader), position=1, leave=False)):
            images = batch['image'].float().to(self.device) #[tio.DATA]
            labels = batch['label'].long().to(self.device) #[tio.DATA]
            # Extract metadata from the batch
            metadata = {}
            for col in self.columns_of_interest:
                metadata[col] = batch[col][0]
            new_df = pd.DataFrame([metadata])  
               
            try:
                val_metadata = pd.concat([val_metadata, new_df], ignore_index=True)
            except NameError:
                val_metadata = new_df
            with torch.cuda.amp.autocast():
                outputs = model(images)
            probs = torch.nn.functional.softmax(outputs, dim=1)
            preds = probs.argmax(dim=1, keepdim=True) 

            for i in range(len(labels)):
                # pred_np = (preds[i, 0].cpu().numpy() == 1).astype(np.uint8)
                # label_np = (labels[i].cpu().numpy() == 1).astype(np.uint8)

                # metrics = compute_segmentation_metrics(prediction_file=pred_np, reference_file=label_np)
                # performance_score = 0.5 * (metrics['DSC'] + (1 - metrics['NormHD'])) 
                # performance_scores.append(performance_score)
                label_tensor = (labels[i:i+1] == 1).float().cpu()
                pred_tensor = (preds[i:i+1, 0:1] == 1).float().cpu() 

                performance_score = self.dice_metric(pred_tensor, label_tensor)
                performance_scores.append(performance_score.item())

        return performance_scores, val_metadata

    def compute_weights(self, sorted_subgroups, train_metadata):
         # Compute the weights based on the divergence scores
        for i in tqdm(range(len(sorted_subgroups)), desc="Computing weights", leave=False):
            subgroup = sorted_subgroups.iloc[i]
            itemset = list(subgroup['itemset'])
            df_res = train_metadata.copy()

            # Filter the training metadata based on the subgroup itemset
                    # Filter the training metadata based on the subgroup itemset
            for condition in itemset:
                k, v = condition.split("=")
                df_res = df_res[df_res[k] == v]

            # Only assign weight if there are matches and they haven't been updated already
            if len(df_res) > 0:
                if train_metadata.loc[df_res.index, "weights"].values[0] == 1.0:
                            train_metadata.loc[df_res.index, "weights"] = abs(sorted_subgroups['performance_score_div'].values[i])
                else:
                    continue

        return train_metadata['weights']  # shape: [B]


    def finding_subgroups_weights(self, val_metadata, train_metadata):
         # Run divexplorer 
        fp_diver = DivergenceExplorer(val_metadata)
        subgroups = fp_diver.get_pattern_divergence(
            min_support=self.min_support,#0.005,
            quantitative_outcomes=[self.performance_name],
            attributes=self.columns_of_interest
        )
        # Sort the subgroups by performance score divergence -- note that if the performance metric is positive, we want to minimize the divergence
        divergence_metric = self.performance_name + '_div'
        sorted_subgroups = subgroups.sort_values(by=divergence_metric, ascending=True)

        weights = self.compute_weights(sorted_subgroups, train_metadata)

        return weights
    
    def calculate_divergence_loss(self, weights, labels, outputs):
        """
        Calculate the divergence loss based on the weights and performance scores.
        Args:
            weights: [B] - sample weights
            performance_scores: [B] - performance scores for each sample
            labels: [B, 1, D, H, W] - int labels
            preds: [B, 1, D, H, W] - predicted labels
        Returns:
            divergence loss value
        """
        # Convert to numpy for easier manipulation
        losses = []
        for i in range(len(labels)):
            # Calculate the Dice loss for each sample
            loss = self.seg_loss(outputs[i, :].unsqueeze(0), labels[i, :].unsqueeze(0))
            # Multiply by the weight 
            weighted_loss = weights.iloc[i] * loss.item()
            losses.append(weighted_loss)
        # Calculate the final divergence loss as the mean of the weighted losses
        #divergence_loss = np.mean(losses)
        divergence_loss = np.sum(losses)
        return divergence_loss


    def forward(self, model, outputs, labels, metadata, validation_set, train_metadata_weights):
        """
        Args:
            outputs: [B, C, D, H, W] - logits
            labels:  [B, 1, D, H, W] - int labels
            metadata: list of dicts with sample-level metadata
            performance_score: optional, not used here directly
        """
        
        # 1. Standard segmentation loss
        l_seg = self.seg_loss(outputs, labels)

        # 2. Compute the performance score of the model over the Validation set
        if validation_set:
            performance_scores, val_metadata = self.compute_performance_score(model=model, val_loader=validation_set)

        # 3. Add the computed performance score to metadata
        val_metadata[self.performance_name] = performance_scores

        # 4. Compute sample weights from divergence
        train_metadata = pd.DataFrame(metadata)
        if train_metadata_weights is not None:
            train_metadata['weights'] = train_metadata_weights
        else:
            train_metadata['weights'] = 1.0
        weights = self.finding_subgroups_weights(val_metadata=val_metadata, train_metadata=train_metadata)  # shape: [B]

        # 4. Calculate the divergence loss
        l_div = self.calculate_divergence_loss(weights=weights, labels=labels, outputs=outputs)

        # 5. Final loss
        return self.alpha * l_seg + (1 - self.alpha) * l_div, l_div, weights
