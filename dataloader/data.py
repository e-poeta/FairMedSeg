import time
import pandas as pd
import torchio as tio
from dataloader.monai_loader import MamaMiaSegmentationDataset



def get_dataloaders(
    image_root_dir,
    seg_root_dir,
    excel_metadata_path,
    split_csv_path,
    columns_of_interest=['age', 'ethnicity', 'menopause'],
    phase_slice=3,
    batch_size=1,
    num_workers=4,
    preprocessing_config=None
):
    """
    Uses a predefined split CSV file to load train/val sets for MamaMiaSegmentationDataset.
    """
    start_time = time.time()
    
    split_df = pd.read_csv(split_csv_path)
    train_ids = split_df["train_split"].dropna().tolist()
    val_ids = split_df["test_split"].dropna().tolist()

    print(f"Train patients: {len(train_ids)}, Val patients: {len(val_ids)}")

    print("Initializing dataset objects...")
    train_dataset = MamaMiaSegmentationDataset(
        image_root_dir=image_root_dir,
        seg_root_dir=seg_root_dir,
        patient_ids=train_ids,
        excel_metadata_path=excel_metadata_path,
        columns_of_interest=columns_of_interest,
        phase_slice=phase_slice,
        preprocessing_config=preprocessing_config
    )
    val_dataset = MamaMiaSegmentationDataset(
        image_root_dir=image_root_dir,
        seg_root_dir=seg_root_dir,
        patient_ids=val_ids,
        excel_metadata_path=excel_metadata_path,
        columns_of_interest=columns_of_interest,
        phase_slice=phase_slice,
        preprocessing_config=preprocessing_config
    )

    train_loader =  tio.SubjectsLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val_loader =  tio.SubjectsLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    elapsed = time.time() - start_time
    print(f"DataLoaders ready in {elapsed:.2f}s")
    return train_loader, val_loader