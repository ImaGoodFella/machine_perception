import torch
import mp_lib.data_utils as data_utils
import mp_lib.sys_utils as sys_utils
from src.datasets.dataset import HO3DDataset
import numpy as np


class HO3DDatasetInfer(HO3DDataset):
    """
    Dataset for testing and to produce the results for submission server evaluation.
    See src/datasets/dataset.py for detailed documentation.
    """

    def __getitem__(self, index):
        data = self.getitem(index)
        return data

    def getitem(self, index, load_rgb=True):
        data_params = self.data_params
        idx = self.indices[index]

        args = self.args
        # LOADING START
        intrx = data_params["K"][idx]
        imgname = self.imgnames[idx]
        bbox = self.bbox[idx].copy()

        cv_img = sys_utils.read_lmdb_image(self.txn, imgname)

        center = [bbox[0], bbox[1]]
        scale = bbox[2]

        # no augmentation
        augm_dict = {}
        augm_dict["flip"] = 0
        augm_dict["pn"] = np.ones(3)
        augm_dict["rot"] = 0
        augm_dict["sc"] = 1.0

        # data augmentation: image
        if load_rgb:
            img = data_utils.rgb_processing(
                self.aug_data,
                cv_img,
                center,
                scale,
                augm_dict,
                img_res=args.img_res,
            )
            img = torch.from_numpy(img).float()
            norm_img = self.normalize_img(img)

        # exporting starts
        inputs = {}
        targets = {}
        meta_info = {}
        inputs["img"] = norm_img
        meta_info["imgname"] = imgname
        meta_info["query_names"] = self.query_names[idx]

        intrx = data_utils.get_aug_intrix(
            intrx,
            args.focal_length,
            args.img_res,
        )
        meta_info["intrinsics"] = intrx
        return inputs, targets, meta_info
