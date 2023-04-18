import torch


def get_num_images(split, num_images):
    if split in ["train", "val", "test"]:
        return num_images

    if split == "smalltrain":
        return 100000

    if split == "tinytrain":
        return 12000

    if split == "minitrain":
        return 300

    if split == "smallval":
        return 12000

    if split == "tinyval":
        return 500

    if split == "minival":
        return 80

    if split == "smalltest":
        return 12000

    if split == "tinytest":
        return 6000

    if split == "minitest":
        return 200

    assert False, f"Invalid split {split}"


def subtract_root_batch(joints: torch.Tensor, root_idx: int):
    assert len(joints.shape) == 3
    assert joints.shape[2] == 3
    joints_ra = joints.clone()
    root = joints_ra[:, root_idx : root_idx + 1].clone()
    joints_ra = joints_ra - root
    return joints_ra


def downsample_index(indices, split):
    if "small" not in split and "mini" not in split and "tiny" not in split:
        return indices
    import random

    random.seed(1)
    assert (
        random.randint(0, 100) == 17
    ), "Same seed but different results; Subsampling might be different."
    num_samples = get_num_images(split, len(indices))
    curr_indices = random.sample(indices, num_samples)
    return curr_indices
