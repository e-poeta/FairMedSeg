from monai.transforms import (
    LoadImaged,
    Compose,
    NormalizeIntensityd,
    EnsureTyped,
    Orientationd,
    Spacingd,
    RandScaleIntensityd,
    RandShiftIntensityd,
    ResizeWithPadOrCropd,
    RandFlipd,
    EnsureChannelFirstd,
    CropForegroundd,
    DeleteItemsd,
    Activationsd,
    AsDiscreted,
    Compose,
    Invertd,
    CopyItemsd
)
from monai.transforms.utility.dictionary import ConcatItemsd


def get_transforms(config):
    """
    Returns the training and validation transforms for the MAMMA-MIA dataset.

    Args:
        config (dict): Configuration dictionary containing dataset parameters.

    Returns:
        tuple: Training and validation transforms.
    """
    # Define the keys for the phases
    phase_keys = [f"phase_{i}" for i in range(3)]

    # Define training and validation transforms
    phase_keys = [f"phase_{i}" for i in range(3)]
    train_transform = Compose(
        [
            LoadImaged(keys=phase_keys + ["label"], allow_missing_keys=True),
            EnsureChannelFirstd(phase_keys + ["label"], channel_dim="no_channel"),
            Orientationd(keys=phase_keys + ["label"], axcodes="RAS"),
            Spacingd(
                keys=phase_keys + ["label"],
                pixdim=(1.0, 1.0, 1.0),
                mode=("bilinear", "bilinear", "bilinear", "nearest"),
            ),
            ResizeWithPadOrCropd(
                keys=phase_keys + ["label"],
                spatial_size=[
                    config["target_shape"][0],
                    config["target_shape"][1],
                    config["target_shape"][2],
                ],
            ),
            ConcatItemsd(keys=phase_keys, name="image", dim=0),
            DeleteItemsd(keys=phase_keys),
            EnsureTyped(keys=["image", "label"]),
            CropForegroundd(
                keys=["image", "label"],
                source_key="image",
                k_divisible=[16, 16, 16],
                allow_smaller=False,
            ),
            ResizeWithPadOrCropd(
                keys=["image", "label"],
                spatial_size=[
                    config["target_shape"][0],
                    config["target_shape"][1],
                    config["target_shape"][2],
                ],
            ),
            RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=0),
            RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=1),
            RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=2),
            NormalizeIntensityd(keys="image", nonzero=True, channel_wise=False),
            RandScaleIntensityd(keys="image", factors=0.1, prob=1.0),
            RandShiftIntensityd(keys="image", offsets=0.1, prob=1.0),
        ]
    )
    val_transform = Compose(
        [
            LoadImaged(keys=phase_keys + ["label"], allow_missing_keys=True),
            EnsureChannelFirstd(phase_keys + ["label"], channel_dim="no_channel"),
            Orientationd(keys=phase_keys + ["label"], axcodes="RAS"),
            Spacingd(
                keys=phase_keys + ["label"],
                pixdim=(1.0, 1.0, 1.0),
                mode=("bilinear", "bilinear", "bilinear", "nearest"),
            ),
            ResizeWithPadOrCropd(
                keys=phase_keys + ["label"],
                spatial_size=[
                    config["target_shape"][0],
                    config["target_shape"][1],
                    config["target_shape"][2],
                ],
            ),
            ConcatItemsd(keys=phase_keys, name="image", dim=0),
            DeleteItemsd(keys=phase_keys),
            EnsureTyped(keys=["image", "label"]),
            CropForegroundd(
                keys=["image", "label"],
                source_key="image",
                k_divisible=[16, 16, 16],
                allow_smaller=False,
            ),
            ResizeWithPadOrCropd(
                keys=["image", "label"],
                spatial_size=[
                    config["target_shape"][0],
                    config["target_shape"][1],
                    config["target_shape"][2],
                ],
            ),
            NormalizeIntensityd(keys="image", nonzero=True, channel_wise=False),
        ]
    )
    
    test_transform =  Compose([
        LoadImaged(keys=phase_keys + ["label"], allow_missing_keys=True),
        EnsureChannelFirstd(phase_keys + ["label"]),
        Orientationd(keys=phase_keys + ["label"], axcodes="RAS"),
        CopyItemsd(keys="phase_0", times=1, names="image_meta_dict"),
        # Spacingd(keys=phase_keys + ["label"], pixdim=(1.0, 1.0, 1.0), mode=("bilinear", "bilinear", "bilinear", "nearest"),),
        ConcatItemsd(keys=phase_keys, name="image", dim=0),
        DeleteItemsd(keys=phase_keys),
        Spacingd(
                keys=["image", "label"],
                pixdim=(1.0, 1.0, 1.0),
                mode=('bilinear','nearest')),
        EnsureTyped(keys=["image", "label"]),
        CropForegroundd(
                keys=["image", "label"],
                source_key="image",
                k_divisible=[32, 32, 32],
                allow_smaller=False),
        NormalizeIntensityd(keys="image", nonzero=True, channel_wise=True),
    ])

    return train_transform, val_transform, test_transform


def get_postransforms(test_transform,config):
    post = Compose([
            Activationsd(keys="pred", softmax=True),
            Invertd(
                keys="pred",
                transform=test_transform,
                orig_keys="label",
                nearest_interp=False,
                to_tensor=True,
                device=config['device'] if 'device' in config else 'cuda',
            ),

            AsDiscreted(keys="pred", argmax=True),
        ])
    return post