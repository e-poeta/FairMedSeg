import os
import pandas as pd


def load_train_val_metadata(
    image_root_dir,
    seg_root_dir,
    excel_metadata_path,
    split_csv_path,
    columns_of_interest,
    phase_slice=3,
):
    """
    Returns MONAI-style dictionaries for training and validation.
    """
    split_df = pd.read_csv(split_csv_path)
    train_ids = split_df["train_split"].dropna().tolist()
    val_ids = split_df["test_split"].dropna().tolist()  

    meta_df = pd.read_excel(excel_metadata_path).set_index("patient_id")
    meta_df = preprocess_metadata(meta_df, columns_of_interest)


    def build_subject_dict(patient_ids):
        subjects = []
        for pid in patient_ids:
            img_dir = os.path.join(image_root_dir, pid)
            image_files = sorted([f for f in os.listdir(img_dir) if f.endswith('.nii.gz')])[:phase_slice]

            if len(image_files) != phase_slice:
                print(f"Patient {pid} has {len(image_files)} phases (expected {phase_slice}). Skipping.")
                continue

            subject_dict = {f"phase_{i}": os.path.join(img_dir, f) for i, f in enumerate(image_files)}
            label_path = os.path.join(seg_root_dir, f"{pid}.nii.gz")
            if os.path.exists(label_path):
                subject_dict["label"] = label_path
            subject_dict["patient_id"] = pid
     
            if pid in meta_df.index:
                for col in columns_of_interest:
                    subject_dict[col] = meta_df.loc[pid, col]

            subjects.append(subject_dict)
        return subjects

    train_files = build_subject_dict(train_ids)
    val_files = build_subject_dict(val_ids)

    return train_files, val_files



def preprocess_metadata(metadata_df: pd.DataFrame, columns_of_interest: list) -> pd.DataFrame:
    """
    Preprocess metadata to retain relevant columns and clean values.
    
    Steps:
    - Keep only specified columns
    - Fill NaNs with 'unknown' (except 'age')
    - Discretize 'age' into bins
    - Normalize 'menopause' categories
    - Convert all columns to string type
    
    Returns:
        pd.DataFrame: Cleaned metadata.
    """
    # Work on a copy to avoid modifying original data and avoid SettingWithCopyWarning
    metadata_df = metadata_df[columns_of_interest].copy()

    # Fill NaNs with 'unknown' except for 'age'
    for col in columns_of_interest:
        if col != 'age' and metadata_df[col].isnull().any():
            metadata_df.loc[:, col] = metadata_df[col].fillna('unknown')

    if 'age' in columns_of_interest:
        # Convert to numeric and handle invalid entries
        metadata_df.loc[:, 'age'] = pd.to_numeric(metadata_df['age'], errors='coerce').fillna(-1)
        metadata_df.loc[:, 'age'] = metadata_df['age'].apply(lambda x: x if x >= 0 else None)

        # Discretize into bins
        age_bins = [0, 40, 50, 60, 70, 100]
        age_labels = ['0-40', '41-50', '51-60', '61-70', '71+']
        metadata_df['age'] = metadata_df['age'].astype("object")
        metadata_df.loc[:, 'age'] = pd.cut(metadata_df['age'], bins=age_bins, labels=age_labels).astype(str)

        # Drop rows with unknown age
        metadata_df = metadata_df.dropna(subset=['age'])

    if 'menopause' in columns_of_interest:
        # Normalize menopause values
        def normalize_menopause(val):
            val = val.lower()
            if 'post' in val:
                return 'post'
            elif 'peri' in val:
                return 'pre'
            elif 'pre' in val:
                return 'pre'
            return 'unknown'
        
        metadata_df.loc[:, 'menopause'] = metadata_df['menopause'].astype(str).apply(normalize_menopause)

    # Convert all columns to string
    metadata_df = metadata_df.astype(str)

    return metadata_df