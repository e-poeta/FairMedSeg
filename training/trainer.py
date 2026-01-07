import os
import torch
from tqdm import tqdm
import wandb
from monai.utils import set_determinism
from monai.metrics import DiceMetric
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
        self.config = config
        self.device = config["device"]
        self.val_interval = config.get("val_interval", 1)
        self.optimizer = torch.optim.Adam(model.parameters(), lr=config.get("learning_rate", 1e-4))
        self.dice_metric = DiceMetric(include_background=False, reduction="mean")
        self.wandb_run = wandb_run

        # Reproducibility
        set_determinism(seed=config.get("seed", 42))

        # Directory for saving models
        self.save_dir = config.get("save_dir", "outputs/checkpoints")
        os.makedirs(self.save_dir, exist_ok=True)

        self.start_epoch = 1
        self.best_dice = 0.0
        if "resume" in config and config["resume"]:
            checkpoint = torch.load(config["resume"], map_location=self.device, weights_only=True)
            self.model.load_state_dict(checkpoint["model_state_dict"])
            self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
            self.start_epoch = checkpoint.get("epoch", 1) + 1
            self.best_dice = checkpoint.get("val_dice", 0.0)
            print(f"Resumed training from epoch {self.start_epoch - 1} with best val dice {self.best_dice:.4f}")

    def _step(self, batch, training=True):
        images = batch["image"].to(self.device).float() #batch["image"][tio.DATA].to(self.device).float()
        labels = batch["label"].to(self.device).long()  # o .float() si tu loss lo requiere#batch["label"][tio.DATA].to(self.device).long()
        if training:
            self.optimizer.zero_grad()
            outputs = self.model(images)
            loss = self.loss_function(outputs, labels)
            loss.backward()
            self.optimizer.step()
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
        for batch_idx, batch in enumerate(tqdm(self.train_loader, desc=f"Training Epoch {epoch}", total=len(self.train_loader), position=0, leave=True)):
            # Remove any non-numerical or problematic metadata keys before collate
            if isinstance(batch, dict):
                batch = {k: v for k, v in batch.items() if isinstance(v, torch.Tensor)}
            
            loss = self._step(batch, training=True)
            running_loss += loss
        return running_loss / len(self.train_loader)

    def validate(self, epoch):
        self.model.eval()
        self.dice_metric.reset()  # important!

        with torch.no_grad():
            for batch_idx, batch in enumerate(tqdm(self.val_loader, desc=f"Validation Epoch {epoch}", total=len(self.val_loader), position=1, leave=True)):
                if isinstance(batch, dict):
                    batch = {k: v for k, v in batch.items() if isinstance(v, torch.Tensor)}
                images = batch["image"].to(self.device).float() #batch["image"][tio.DATA].to(self.device).float()
                labels = batch["label"].to(self.device).long()  # o .float() si tu loss lo requiere#batch["label"][tio.DATA].to(self.device).long()

                outputs = self.model(images)
                probs = torch.nn.functional.softmax(outputs, dim=1)
                preds = probs.argmax(dim=1, keepdim=True)

                self.dice_metric(preds, labels)

        dice_score = self.dice_metric.aggregate().item()
        return dice_score

    def train(self):
        best_dice = self.best_dice
        for epoch in range(self.start_epoch, self.config["epochs"] + 1):
            train_loss = self.train_epoch(epoch)
            val_dice = None
            if epoch % self.val_interval == 0:
                val_dice = self.validate(epoch)
                print(f"Epoch {epoch}/{self.config['epochs']} - Loss: {train_loss:.4f} - Val Dice: {val_dice:.4f}")

                # Save everything needed to resume training
                checkpoint = {
                    "epoch": epoch,
                    "model_state_dict": self.model.state_dict(),
                    "optimizer_state_dict": self.optimizer.state_dict(),
                    "train_loss": train_loss,
                    "val_dice": val_dice,
                }
                model_path = os.path.join(self.save_dir, f"epoch{epoch}.pth")
                torch.save(checkpoint, model_path)
                
                if val_dice > best_dice:
                    best_dice = val_dice
                    best_path = os.path.join(self.save_dir, "best_checkpoint.pth")
                    torch.save(checkpoint, best_path)
                    print(f"Best model updated (Val Dice: {best_dice:.4f})")

            else:
                print(f"Epoch {epoch}/{self.config['epochs']} - Loss: {train_loss:.4f}")

            if self.wandb_run:
                log_data = {
                    "epoch": epoch,
                    "train_loss": train_loss,
                }
                if val_dice is not None:
                    log_data["val_dice"] = val_dice
                    log_data["best_val_dice"] = best_dice
                self.wandb_run.log(log_data)
