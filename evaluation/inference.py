import os
import torch
import numpy as np
import SimpleITK as sitk
import json
import nibabel as nib
import scipy.ndimage as ndi

from monai.data import Dataset, DataLoader, decollate_batch
from monai.networks.nets import UNet, SwinUNETR,SegResNet


def load_patient_json(patient_id, patient_info_dir):
    json_path = os.path.join(patient_info_dir, f"{patient_id}.json")
    with open(json_path, "r") as file:
        data = json.load(file)
    return data


class Model:
    def __init__(
        self,
        dataset,
        model_type="unet",
        exp_name="unetb1m",
        device="cuda",
        patient_info_dir=None,
        post_transforms=None,
    ):
        self.dataset = dataset
        self.model_type = model_type
        self.exp_name = exp_name
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.target_shape = (320, 320, 128)
        self.patient_info_dir = patient_info_dir
        self.post_transforms = post_transforms

    def load_model(self):
        if self.model_type == "unet":
            model = UNet(
                spatial_dims=3,
                in_channels=3,
                out_channels=2,
                channels=(16, 32, 64, 128, 256),
                strides=(2, 2, 2, 2),
                num_res_units=2,
            )
        elif self.model_type== "segresnet":
            model = SegResNet(
                blocks_down=[1, 2, 2, 4],     
                blocks_up=[1, 1, 1],         
                init_filters=32,             
                in_channels=3,              
                out_channels=2,              
                dropout_prob=0.1             
            )
        elif self.model_type == "swinunetr":
            model = SwinUNETR(
                img_size=self.target_shape,
                in_channels=3,
                out_channels=2,
                feature_size=48,
                use_checkpoint=False,
            )
        else:
            raise ValueError(f"Unknown model type: {self.model_type}")
        model_path = f"/home/vargas/mammaMia/training/outputs/{self.exp_name}/checkpoints/best_checkpoint.pth"
        model.load_state_dict(
            torch.load(model_path, map_location=self.device, weights_only=True)[
                "model_state_dict"
            ]
        )
        model.to(self.device)
        model.eval()
        return model


    def post_process_segmentation(self, pred_seg, affine, patient_id):
        patient_info = load_patient_json(patient_id, self.patient_info_dir)
        coords = patient_info["primary_lesion"]["breast_coordinates"]
        print(f"Processing patient {patient_id} with coordinates: {coords}")
        # Step 1: Apply bounding box mask of the brest 
        mask = np.zeros_like(pred_seg)
        mask[
            coords["z_min"]:coords["z_max"],  # x -> z
            coords["y_min"]:coords["y_max"],  # y -> y
            coords["x_min"]:coords["x_max"]   # z -> x
        ] = pred_seg[
            coords["z_min"]:coords["z_max"],
            coords["y_min"]:coords["y_max"],
            coords["x_min"]:coords["x_max"]
        ]

        nib_seg_post = nib.Nifti1Image(mask, affine)
        nib.save(nib_seg_post, os.path.join(self.output_dir + "_post", f"{patient_id}.nii.gz"))
        print(f"Post-processed segmentation saved in: {self.output_dir+'_post'}")

    def predict_segmentation(self, output_dir):
        print("Running inference...")
        model = self.load_model()
        self.output_dir = output_dir
        test_loader = DataLoader(self.dataset, batch_size=1, num_workers=8)

        os.makedirs(output_dir, exist_ok=True)
        os.makedirs(output_dir + "_post", exist_ok=True)

        from monai.inferers import sliding_window_inference

        for batch in test_loader:
            patient_id = batch["patient_id"][0]
            image_path = batch["image"].meta["filename_or_obj"][0]
            print(f'image path: {image_path}')

            pred_path = os.path.join(output_dir, f"{patient_id}.nii.gz")
            post_path = os.path.join(output_dir + "_post", f"{patient_id}.nii.gz")

            if os.path.exists(pred_path):
                print(f"Prediction already exists for {patient_id}")
                if os.path.exists(post_path):
                    print(f"Post-processed file already exists for {patient_id}, skipping...")
                    continue
                else:
                    pred_seg = nib.load(pred_path).get_fdata().astype(np.uint8)
                    affine = nib.load(pred_path).affine
                    self.post_process_segmentation(pred_seg, affine, patient_id)
                    continue

            nib_image = nib.load(image_path)
            affine = nib_image.affine
            image = batch["image"].to(self.device)


            with torch.no_grad():
                if self.model_type == "swinunetr":
                    output = sliding_window_inference(
                        inputs=image,
                        roi_size=self.target_shape,
                        sw_batch_size=1,
                        predictor=model,
                        overlap=0.25
                    )
                else:
                    output = model(image)
            batch["pred"] = output
            batch["pred_meta_dict"] = batch["image_meta_dict"]
            batch = decollate_batch(batch)
            post_batch = [self.post_transforms(b) for b in batch]
    
            pred_seg = np.squeeze(post_batch[0]["pred"].cpu().numpy().astype(np.uint8))

            # Save using original affine 
            nib_seg = nib.Nifti1Image(pred_seg, affine)
            nib.save(nib_seg, pred_path)
            print(f"Segmentation saved in: {output_dir}")

            self.post_process_segmentation(pred_seg, affine, patient_id)
            

        return
