import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import warnings
from dataloader.load_monai_metadata import load_train_val_metadata
from configs.transforms_config import get_transforms
from fairness.loss_divergence import DivergenceFairLoss
from monai.data import Dataset, DataLoader, CacheDataset,PersistentDataset

from trainer import Trainer
from dataloader.data import get_dataloaders
from monai.utils import set_determinism
from monai.losses import DiceLoss, DiceCELoss
from monai.networks.nets import UNet, SwinUNETR, SegResNet
import wandb
import torch
import argparse

warnings.filterwarnings("ignore", message="Using TorchIO images without a torchio.SubjectsLoader")
warnings.filterwarnings("ignore", category=FutureWarning, module="monai.data.dataset")
# ---------------------- Argument Parsing ----------------------
parser = argparse.ArgumentParser(description="Training script for MAMMA-MIA")

parser.add_argument("--gpu", type=int, default=6, help="CUDA device index")
parser.add_argument( "--model",type=str,choices=["unet", "swinunetr", "segresnet"],default="unet",
                    help="Model type",)
parser.add_argument( "--exp_name", type=str, default="unet-baseline", 
                    help="WandB experiment name")
parser.add_argument("--batch_size", type=int, default=1,
                    help="Batch size for training")
parser.add_argument("--epochs", type=int, default=500,
                    help="Number of training epochs")
parser.add_argument("--val_interval", type=int,
                    default=1, help="Validation interval")
parser.add_argument("--monai", action="store_true", help="Use MONAI pipeline", default=True)
parser.add_argument("--divloss", action="store_true", help="Use divergence loss", default=True)
parser.add_argument("--resume", type=str, default=None, help="Name of the checkpoint")
parser.add_argument("--alpha", type=float, default=0.5, help="Weight for the divergence loss")
parser.add_argument("--columns_of_interest", type=str, nargs="+", default=["age", "breast_density", "menopause"], help="Columns of interest for divergence loss")
parser.add_argument("--load_checkpoint", action="store_true", help="Load model checkpoint", default=False)
parser.add_argument("--min_support", type=float, default=0.005, help="Minimum support for subgroup discovery")



args = parser.parse_args()
os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)


USE_MONAI_PIPELINE = args.monai  
USE_DIVLOSS = args.divloss  

# CONFIG
set_determinism(42)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
config = {
    "device": device,
    "epochs": args.epochs,
    "starting_epoch": 0,
    "learning_rate": 1e-4,
    "val_interval": args.val_interval,
    "project_name": "mamma-mia",
    "run_name": args.exp_name,
    "model_save_path": "models",
    "save_dir": f"outputs/{ args.exp_name}/checkpoints",
    "model_name": args.exp_name,
    "alpha": args.alpha,  # Weight for the divergence loss
    "divloss": USE_DIVLOSS,  # Use divergence loss
    "columns_of_interest": args.columns_of_interest,  # Columns of interest for divergence loss
}
preprocessing_config = {
    "apply_bias_correction": False,
    "apply_denoising": False,
    "apply_intensity_clipping": True,
    "apply_intensity_normalization": True,
    "resample_spacing": (1.0, 1.0, 1.0),
    "target_shape": (320, 320, 128),# (256, 256, 96)  (320, 320, 128) # (H, W, D) (272, 272, 128)
}

config.update(preprocessing_config)
# WANDB INIT
wandb_run = wandb.init(project="mama-mia", name=args.exp_name, config=config, job_type="train")

# DATA LOADERS
if USE_MONAI_PIPELINE:
    train_transform, val_transform, test_transform = get_transforms(config)
    train_files, val_files = load_train_val_metadata(
        # Modify the paths as needed
        image_root_dir=None,
        seg_root_dir=None,
        excel_metadata_path=None,
        split_csv_path=None,
        columns_of_interest=["age", "breast_density", "menopause"],
        phase_slice=3,  # 3: pre + 2 phases
    )

    train_ds = PersistentDataset(data=train_files, transform=train_transform, cache_dir="./persistent_cache_meta/train")
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=4)

    val_ds = PersistentDataset(data=val_files, transform=val_transform, cache_dir="./persistent_cache_meta/val")
    val_loader = DataLoader(val_ds, batch_size=1, num_workers=4)

else:
    train_loader, val_loader = get_dataloaders(
        # Modify the paths as needed
        image_root_dir=None,
        seg_root_dir=None,
        excel_metadata_path=None,
        split_csv_path=None,
        columns_of_interest=["age", "breast_density", "menopause"],
        phase_slice=3,
        batch_size=args.batch_size,
        num_workers=2,
        preprocessing_config=preprocessing_config,
    )

# MODEL
if args.model == "unet":
    model = UNet(
        spatial_dims=3,
        in_channels=3,
        out_channels=2,
        channels=(16, 32, 64, 128, 256),
        strides=(2, 2, 2, 2),
        num_res_units=2,
    ).to(device)
elif args.model == "swinunetr":
    model = SwinUNETR(img_size=config['target_shape'],
                        in_channels=3,
                      out_channels=2,
                      feature_size=48,
                      use_checkpoint=True,
                      )
    model = model.to("cuda:0")
    model = torch.nn.DataParallel(model, device_ids=[0, 1])  # Use multiple GPUs if available
elif args.model == "segresnet":
    model = SegResNet(
        blocks_down=[1, 2, 2, 4],     
        blocks_up=[1, 1, 1],         
        init_filters=32,             
        in_channels=3,              
        out_channels=2,              
        dropout_prob=0.1             
    ).to(device)
else:
    raise ValueError(f"Unknown model type: {args.model}")


if args.load_checkpoint:
    # Load the model from the specified checkpoint
    print(f"Loading model from checkpoint: {CHECKPOINT_PATH}")
    checkpoint = torch.load(CHECKPOINT_PATH, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    config['starting_epoch'] = checkpoint['epoch']  # Resume from the last epoch
    print(f"Resuming training from epoch {checkpoint['epoch']}")
    wandb.restore(CHECKPOINT_PATH)

if USE_DIVLOSS:
    # Use DivergenceFairLoss
    print("Using DivergenceFairLoss")
    loss_function = DivergenceFairLoss(min_support=args.min_support, device=device, alpha=config['alpha'], sensitive_columns=args.columns_of_interest)
else:
    # Use standard DiceCELoss
    print("Using standard DiceCELoss")
    loss_function = DiceCELoss(to_onehot_y=True,softmax=True)


# TRAINER
trainer = Trainer(
    model=model,
    train_loader=train_loader,
    val_loader=val_loader,
    loss_function=loss_function,
    config=config,
    wandb_run=wandb_run,
)
# START TRAINING
trainer.train()
wandb.finish()
