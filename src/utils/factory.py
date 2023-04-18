import torch
from torch.utils.data import DataLoader

from src.datasets.dataset import HO3DDataset
from src.datasets.dataset_infer import HO3DDatasetInfer
from mp_lib.torch_utils import reset_all_seeds
from mp_lib.torch_utils import np2torch


def fetch_dataset(args, is_train):
    split = args.trainsplit if is_train else args.valsplit

    DATASET = HO3DDataset
    if "test" in split:
        DATASET = HO3DDatasetInfer
    ds = DATASET(args=args, split=split)
    return ds


def collate_custom_fn(data_list):
    data = data_list[0]
    _inputs, _targets, _meta_info = data
    out_inputs = {}
    out_targets = {}
    out_meta_info = {}

    for key in _inputs.keys():
        out_inputs[key] = []

    for key in _targets.keys():
        out_targets[key] = []

    for key in _meta_info.keys():
        out_meta_info[key] = []

    for data in data_list:
        inputs, targets, meta_info = data
        for key, val in inputs.items():
            out_inputs[key].append(val)

        for key, val in targets.items():
            out_targets[key].append(val)

        for key, val in meta_info.items():
            out_meta_info[key].append(val)

    for key in _inputs.keys():
        out_inputs[key] = torch.stack(out_inputs[key], dim=0)

    for key in _targets.keys():
        out_targets[key] = torch.stack(out_targets[key], dim=0)

    for key in _meta_info.keys():
        if key not in ["imgname", "query_names"]:
            out_meta_info[key] = [np2torch(ten) for ten in out_meta_info[key]]
            out_meta_info[key] = torch.stack(out_meta_info[key], dim=0)
    return out_inputs, out_targets, out_meta_info


def fetch_dataloader(args, mode):
    if mode == "train":
        reset_all_seeds(args.seed)
        dataset = fetch_dataset(args, is_train=True)
        collate_fn = collate_custom_fn
        return DataLoader(
            dataset=dataset,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            pin_memory=args.pin_memory,
            shuffle=args.shuffle_train,
            collate_fn=collate_fn,
        )

    elif mode in ["val", "test"]:
        dataset = fetch_dataset(args, is_train=False)
        collate_fn = collate_custom_fn
        return DataLoader(
            dataset=dataset,
            batch_size=args.test_batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            collate_fn=collate_fn,
        )
    else:
        assert False


def fetch_model(args):
    from src.models.hmr.wrapper import Wrapper as Wrapper

    model = Wrapper(args)
    return model
