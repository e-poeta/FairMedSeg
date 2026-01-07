import os
import pandas as pd
from torch.utils.data import Dataset
from dataloader.preprocess import preprocess_dce_mri

class MamaMiaSegmentationDataset(Dataset):
    def __init__(self, image_root_dir, seg_root_dir, patient_ids,excel_metadata_path, columns_of_interest, phase_slice=3,preprocessing_config=None,):
        self.image_root_dir = image_root_dir
        self.seg_root_dir = seg_root_dir
        self.patient_ids = patient_ids
        self.phase_slice = phase_slice  
        df = pd.read_excel(excel_metadata_path)
        df = df[df['patient_id'].isin(patient_ids)]
        self.metadata_df = df.set_index("patient_id")
        self.columns_of_interest = columns_of_interest
        self.preprocessing_config = preprocessing_config or {
            "apply_bias_correction": False,
            "apply_denoising": False,
            "apply_intensity_clipping": True,
            "apply_intensity_normalization": True,
            "resample_spacing": (1.0, 1.0, 1.0),
            "target_shape": (384, 384, 160)
        }

        # Load metadata CSV once
        df = pd.read_csv(csv_metadata_path)
        df = df[df['patient_id'].isin(patient_ids)]  # filter only relevant
        self.metadata_df = df.set_index("patient_id")
        self.columns_of_interest = columns_of_interest

    def __len__(self):
        return len(self.patient_ids)

    def __getitem__(self, idx):
        patient_id = self.patient_ids[idx]
        patient_path = os.path.join(self.image_root_dir, patient_id)
        image_files = sorted([f for f in os.listdir(patient_path) if f.endswith('.nii.gz')])[:self.phase_slice]
        image_paths = [os.path.join(patient_path, f) for f in image_files]

        seg_path = os.path.join(self.seg_root_dir, f"{patient_id}.nii.gz")
        seg_path = seg_path if os.path.exists(seg_path) else None

        subject = preprocess_dce_mri(
            image_paths=image_paths,
            label_path=seg_path,
            **self.preprocessing_config  
        )
        metadata_row = self.metadata_df.loc[patient_id]
        metadata = {col: metadata_row[col] for col in self.columns_of_interest}                     
        subject['patient_id'] = patient_id
        subject['metadata'] = metadata
        return subject
