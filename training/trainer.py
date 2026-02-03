import os
import torch
from tqdm import tqdm
import wandb
import numpy as np
import pandas as pd
from monai.utils import set_determinism
from monai.metrics import DiceMetric
from monai.losses import DiceCELoss, HausdorffDTLoss, DiceFocalLoss
from metrics.challenge_scores import compute_performance_score, compute_fairness_ranking_score
import torchio as tio 

class Trainer:
    def __init__(
        self,
        model,
        train_loader,
        val_loader,
        loss_function,
        config: dict,
        wandb_run=None,
    ):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.loss_function = loss_function
        self.wandb_run = wandb_run
        self.config = config
        self.device = config["device"]
        self.use_divloss = config["divloss"]
        self.optimizer = torch.optim.Adam(model.parameters(), lr=config.get("learning_rate", 1e-4))
        self.dice_metric = DiceMetric(include_background=False, reduction="mean")
        self.train_metadata_weights = None
        self.train_metadata = pd.DataFrame()  # Initialize an empty DataFrame for training metadata
        #self.dice_loss = DiceCELoss(to_onehot_y=True,softmax=True)
        self.dice_loss = DiceFocalLoss(to_onehot_y=True,softmax=True,)
        self.hd_loss = HausdorffDTLoss(
            to_onehot_y=True,softmax=True
        )

        # Reproducibility
        set_determinism(seed=config.get("seed", 42))

        # Directory for saving models
        self.save_dir = config.get("save_dir", "outputs/checkpoints")
        os.makedirs(self.save_dir, exist_ok=True)

        self.best_dice = 0.0
        if "resume" in config and config["resume"]:
            checkpoint = torch.load(config["resume"], map_location=self.device, weights_only=True)
            self.model.load_state_dict(checkpoint["model_state_dict"])
            self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
            self.start_epoch = checkpoint.get("epoch", 1) + 1
            self.best_dice = checkpoint.get("val_dice", 0.0)
            print(f"Resumed training from epoch {self.start_epoch - 1} with best val dice {self.best_dice:.4f}")

    def _step(self, batch, batch_idx, training=True):
        images = batch["image"].to(self.device).float() 
        labels = batch["label"].to(self.device).long() 

        if training:
            self.optimizer.zero_grad()
            outputs = self.model(images)
            if self.use_divloss:
                metadata = {}
                for col in self.config['columns_of_interest']:
                    metadata[col] = batch[col]
                new_df = pd.DataFrame(metadata)  
                try:
                    self.train_metadata = pd.concat([self.train_metadata, new_df], ignore_index=True)
                except NameError:
                    self.train_metadata = new_df

                l_seg = self.dice_loss(outputs, labels)
                current_weights = self.train_metadata_weights[(batch_idx)* len(labels):(batch_idx+1)*len(labels)].reset_index(drop=True)
                div_loss = self.loss_function.calculate_divergence_loss(weights=current_weights, 
                                                                    labels=labels, 
                                                                    outputs=outputs) 
            
                loss = self.config['alpha'] * l_seg + (1 - self.config['alpha']) * div_loss
            else:
                loss = self.dice_loss(outputs, labels)            
            
            loss.backward()
            self.optimizer.step()
            if self.wandb_run:
                self.wandb_run.log({"train_loss_batch": loss.item()})
                if self.use_divloss:
                    self.wandb_run.log({
                        "div_loss": div_loss.item(),
                    })
            return loss.item()
        else:
            with torch.no_grad():
                outputs = self.model(images)
                probs = torch.nn.functional.softmax(outputs, dim=1)
                preds = probs.argmax(dim=1, keepdim=True)
                self.dice_metric(preds, labels)
            return None

    def train_epoch(self, epoch):
        self.model.train()
        running_loss = 0.0
        if epoch == 1 and self.use_divloss:
            self.train_metadata_weights = pd.DataFrame(len(self.train_loader.dataset) * [1.0], columns=['weights'])  # Initialize weights to 1.0 for the first epoch

        for batch_idx, batch in enumerate(tqdm(self.train_loader, desc=f"Training Epoch {epoch}", total=len(self.train_loader), position=0, leave=True)):
            loss = self._step(batch, batch_idx, training=True)
            running_loss += loss
            if self.wandb_run:
                self.wandb_run.log({
                    "epoch": epoch,
                    "batch": batch_idx
                })
        return running_loss / len(self.train_loader)

    def validate(self, epoch=None, compute_scores=False):

        performance_scores, DSC, NormHD = [], [], []
        val_metadata_with_performance = pd.DataFrame()  # Initialize an empty DataFrame for validation metadata
        self.model.eval()
        self.dice_metric.reset()  # important!

        with torch.no_grad():
            for batch_idx, batch in enumerate(tqdm(self.val_loader, desc=f"Validation Epoch {epoch}", total=len(self.val_loader), position=1, leave=True)):
                images = batch["image"].to(self.device).float() 
                labels = batch["label"].to(self.device).long()  

                val_metadata_dict = {}
                for col in self.config['columns_of_interest']:
                    val_metadata_dict[col] = batch[col]
                val_metadata = pd.DataFrame(val_metadata_dict)

                outputs = self.model(images)
                probs = torch.nn.functional.softmax(outputs, dim=1)
                preds = probs.argmax(dim=1, keepdim=True)


                self.dice_metric(preds, labels)
                if compute_scores:
                    performance_score, val_metadata_performance, dsc, normhd = compute_performance_score(
                        outputs=outputs,
                        labels=labels,
                        columns_of_interest=self.config['columns_of_interest'],
                        val_metadata=val_metadata,
                    )
                    performance_scores.append(performance_score)
                    DSC.append(dsc)
                    NormHD.append(normhd)
                    val_metadata_with_performance = pd.concat([val_metadata_with_performance, val_metadata_performance], ignore_index=True)
                    self.wandb_run.log({
                        "performance_score": performance_score,
                    })

                if self.wandb_run:
                    self.wandb_run.log({
                        "epoch": epoch,
                        "batch": batch_idx,
                        "val_batch_dice": self.dice_metric.aggregate().item(),
                    })


        # Compute the Dice score
        dice_score = self.dice_metric.aggregate().item()
        # Compute the average scores
        if len(performance_scores) > 0:
            avg_performance_score = np.mean(performance_scores)
            avg_dsc = np.mean(DSC)
            avg_normhd = np.mean(NormHD)
        # Compute the fairness and ranking scores
        if compute_scores:
            avg_fairness_score, ranking_score = compute_fairness_ranking_score(val_metadata=val_metadata_with_performance,
                                           columns_of_interest=self.config['columns_of_interest'],
                                           avg_performance_score=avg_performance_score)
            if self.wandb_run:
                self.wandb_run.log({
                    "epoch": epoch,
                    "val_performance_score": avg_performance_score,
                    "val_dice": avg_dsc,
                    "val_normhd": avg_normhd,
                    "val_fairness_score": avg_fairness_score,
                    "val_ranking_score": ranking_score  
                })

        else:
            if self.wandb_run:
                self.wandb_run.log({
                    "epoch": epoch,
                    "val_dice": dice_score,                    
                })    
        return dice_score
    
    # def _update_weights(self):
    #     performance_name = 'performance_score'  # name of the performance score in metadata
    #     performance_scores, val_metadata = self.loss_function.compute_performance_score(model=self.model, val_loader=self.val_loader)
    #     val_metadata[performance_name] = performance_scores

    #     self.train_metadata['weights'] = 1.0
    #     weights = self.loss_function.finding_subgroups_weights(val_metadata=val_metadata, train_metadata=self.train_metadata)

    #     self.train_metadata_weights = weights

    def _update_weights(self):
        performance_name = 'performance_score'

        # TEMPORARILY use multi-GPU for scoring
        # available_devices = [0, 1]  # <--- replace with your desired GPU indices (relative to visible devices)
        # if torch.cuda.device_count() > 1:
        #     print(f"Using DataParallel on GPUs: {available_devices}")
        #     model = torch.nn.DataParallel(self.model, device_ids=available_devices)
        # else:
        #     model = self.model

        self.model.eval()

        # Compute performance scores
        performance_scores, val_metadata = self.loss_function.compute_performance_score(
            model=self.model,
            val_loader=self.val_loader
        )

        # Store scores in metadata
        val_metadata[performance_name] = performance_scores

        self.train_metadata['weights'] = 1.0
        weights = self.loss_function.finding_subgroups_weights(
            val_metadata=val_metadata,
            train_metadata=self.train_metadata
        )

        self.train_metadata_weights = weights

        # Clear CUDA memory
        torch.cuda.empty_cache()


    def train(self):
        best_dice = 0.0 

        for epoch in range(1 + self.config["starting_epoch"], self.config["epochs"] + 1):
            train_loss = self.train_epoch(epoch=epoch)
            val_dice = None
            if self.use_divloss:
                print("Updating weights based on performance scores...")
                self._update_weights()
                # Re initiliaze the self.train_metadata
                self.train_metadata = pd.DataFrame()
                
            if epoch % self.config["val_interval"] == 0:
                val_dice = self.validate(epoch=epoch)
                print(f"Epoch {epoch}/{self.config['epochs']} - Loss: {train_loss:.4f} - Val Dice: {val_dice:.4f}")

                checkpoint = {
                    "epoch": epoch,
                    "model_state_dict": self.model.state_dict(),
                    "optimizer_state_dict": self.optimizer.state_dict(),
                    "train_loss": train_loss,
                    "val_dice": val_dice,
                }
                model_path = os.path.join(self.save_dir, f"{self.config['model_name']}_epoch{epoch}.pth")
                torch.save(checkpoint, model_path)

                if val_dice > best_dice:
                    best_dice = val_dice
                    best_path = os.path.join(self.save_dir, "best_checkpoint.pth")
                    torch.save(checkpoint, best_path)
                    print(f"Best model updated (Val Dice: {best_dice:.4f})")

            else:
                print(f"Epoch {epoch}/{self.config['epochs']} - Loss: {train_loss:.4f}")

            if self.wandb_run:
                self.wandb_run.log({
                    "epoch": epoch,
                    "train_loss": train_loss,
                })
                if val_dice is not None:
                    self.wandb_run.log({
                        "val_dice": val_dice,
                        "best_val_dice": best_dice,
                    })
