import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))


from inference import Model  
import argparse
from configs.transforms_config import get_transforms,get_postransforms
from dataloader.load_monai_metadata import load_train_val_metadata
from monai.data import Dataset, DataLoader, CacheDataset,PersistentDataset
import torch
import wandb
import warnings

warnings.filterwarnings("ignore", category=FutureWarning, module="monai.data.dataset")

parser = argparse.ArgumentParser(description="Test script for MAMMA-MIA")

parser.add_argument("--gpu", type=int, default=6, help="CUDA device index")
parser.add_argument( "--model",type=str,choices=["unet", "swinunetr", "segresnet"],default="unet",
                    help="Model type",)
parser.add_argument( "--exp_name", type=str, default="unetb1m", 
                    help="WandB experiment name")

args = parser.parse_args()
os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)

# CONFIG
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
config = {
    "device": device,
    "project_name": "mamma-mia",
    "run_name": args.exp_name,
    "save_dir": None,
}
preprocessing_config = {
    "apply_bias_correction": False,
    "apply_denoising": False,
    "apply_intensity_clipping": True,
    "apply_intensity_normalization": True,
    "resample_spacing": (1.0, 1.0, 1.0),
    "target_shape": (320, 320, 128)  
}
config.update(preprocessing_config)

train_transform,val_transform,test_transforms = get_transforms(config)

post_transforms = get_postransforms(test_transforms, config)

train_files, val_files = load_train_val_metadata(
        # Modify the paths as needed
        image_root_dir=None,
        seg_root_dir=None,
        excel_metadata_path=None,
        split_csv_path=None,
        columns_of_interest=["age", "breast_density", "menopause"],
        phase_slice=3,  # 3: pre + 2phases
    )

test_ds = CacheDataset(data=val_files, transform=test_transforms, cache_rate=1.0, num_workers=4)
 
model = Model(
    dataset=test_ds,
    model_type="unet",             
    exp_name=args.exp_name,
    device="cuda",
    patient_info_dir=None, #Modify as needed
    post_transforms=post_transforms,
)


# Run prediction
model.predict_segmentation(config["save_dir"])