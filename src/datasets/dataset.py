import torch
import numpy as np
import os.path as op
from loguru import logger


from torch.utils.data import Dataset
import mp_lib.geometry as geometry
from mp_lib.geometry import batch_rot2aa
import src.datasets.dataset_utils as dataset_utils
from torchvision.transforms import Normalize
import mp_lib.data_utils as data_utils
import mp_lib.sys_utils as sys_utils


def pad_jts2d(jts):
    num_jts = jts.shape[0]
    jts_pad = np.ones((num_jts, 3))
    jts_pad[:, :2] = jts
    return jts_pad


class HO3DDataset(Dataset):
    """
    Dataset for training and validation of the model on {train, val, fulltrain} sets.
    HO3D only consists of right-handed data.
    """

    def __getitem__(self, index):
        data = self.getitem(index)
        return data

    def getitem(self, index, load_rgb=True):
        # data in camera coordinate (full image resolution space)
        data_cam = self.data_cam

        # data in 2d image space (full image resolution space)
        data_2d = self.data_2d

        # parameters for hands and objects
        data_params = self.data_params

        # other meta data
        data_meta = self.data_meta

        idx = self.indices[index]

        args = self.args
        # LOADING START
        intrx = data_params["K"][idx]  # camera intrinsics

        # hands
        joints2d_r = pad_jts2d(
            data_2d["joints.right"][idx].copy()
        )  # 2d joints of right hand
        joints3d_r = data_cam["joints.right"][idx].copy()  # 3d joints of right hand
        rot_r = data_params["rot_r"][idx].copy()  # global rotation of right hand
        pose_r = data_params["pose_r"][idx].copy()  # MANO pose of right hand
        betas_r = data_params["shape_r"][idx].copy()  # MANO shape of right hand
        right_valid = data_cam["right_valid"][idx]  # whether right hand is valid

        # object
        bbox2d_b = pad_jts2d(
            data_2d["bbox.bottom"][idx].copy()
        )  # 2d reprojection of 3d object bounding box
        bbox3d_b = data_cam["bbox.bottom"][idx].copy()

        kp2d_b = pad_jts2d(
            data_2d["kps.bottom"][idx].copy()
        )  # 2d reprojection of 3d object keypoints
        kp3d_b = data_cam["kps.bottom"][idx].copy()  # 3d object keypoints

        imgname = self.imgnames[idx]
        bbox = self.bbox[idx].copy()  # bounding box to crop image

        cv_img = sys_utils.read_lmdb_image(self.txn, imgname)

        # bbox = [x, y, scale] for image cropping
        # to convert bbox to [x1, y1, x2, y2] see mp_utils/data_utils.py
        center = [bbox[0], bbox[1]]
        scale = bbox[2]

        # get data augmentation parameters
        augm_dict = data_utils.augm_params(
            self.aug_data,
            0.0,
            args.noise_factor,
            args.rot_factor,
            args.scale_factor,
        )

        # augment and process 2d points
        # they will be in normalized form
        joints2d_r = data_utils.j2d_processing(
            joints2d_r, center, scale, augm_dict, args.img_res
        )
        kp2d_b = data_utils.j2d_processing(
            kp2d_b, center, scale, augm_dict, args.img_res
        )
        bbox2d_b = data_utils.j2d_processing(
            bbox2d_b, center, scale, augm_dict, args.img_res
        )
        bbox2d = bbox2d_b
        kp2d = kp2d_b

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

        pose_r = np.concatenate((rot_r, pose_r), axis=0)

        # hands
        # MANO pose and shape
        targets["mano.pose.r"] = torch.from_numpy(
            data_utils.pose_processing(pose_r, augm_dict)
        ).float()
        targets["mano.beta.r"] = torch.from_numpy(betas_r).float()

        # MANO joints in 2d reprojection (normalized)
        targets["mano.j2d.norm.r"] = torch.from_numpy(joints2d_r[:, :2]).float()

        # object 3d keypoints in camera coordinate (full resolution image space)
        targets["object.kp3d.full.b"] = torch.from_numpy(kp3d_b[:, :3]).float()

        # 2d reprojection of object 3d keypoints image pixel space (full resolution image space)
        # normalized
        targets["object.kp2d.norm.b"] = torch.from_numpy(kp2d_b[:, :2]).float()

        # 3d bounding box
        targets["object.bbox3d.full.b"] = torch.from_numpy(bbox3d_b[:, :3]).float()
        targets["object.bbox2d.norm.b"] = torch.from_numpy(bbox2d_b[:, :2]).float()

        targets["object.kp2d.norm"] = torch.from_numpy(kp2d[:, :2]).float()
        targets["object.bbox2d.norm"] = torch.from_numpy(bbox2d[:, :2]).float()

        # object canonical 3d keypoints
        meta_info["kp3d.cano"] = data_meta["kp3d.cano"][idx]
        kp3d_cano = meta_info["kp3d.cano"]  # .numpy()
        kp3d_target = targets["object.kp3d.full.b"][:, :3].numpy()

        # rotate canonical kp3d to match original image
        R, _ = geometry.solve_rigid_tf_np(kp3d_cano, kp3d_target)
        obj_rot = (
            batch_rot2aa(torch.from_numpy(R).float().view(1, 3, 3)).view(3).numpy()
        )

        # multiply rotation from data augmentation for object global orientation
        obj_rot_aug = geometry.rot_aa(obj_rot, augm_dict["rot"])
        targets["object.rot"] = torch.FloatTensor(obj_rot_aug).view(1, 3)

        # MANO hand 3d joints in camera coordinate (full resolution image space)
        targets["mano.j3d.full.r"] = torch.FloatTensor(joints3d_r[:, :3])
        # object 3d keypoints in camera coordinate (full resolution image space)
        targets["object.kp3d.full.b"] = torch.FloatTensor(kp3d_b[:, :3])
        # object name for this image
        meta_info["query_names"] = self.query_names[idx]

        # camera intrinsics from focal length based on patch size
        intrx = data_utils.get_aug_intrix(
            intrx,
            args.focal_length,
            args.img_res,
        )

        meta_info["intrinsics"] = torch.FloatTensor(intrx)
        meta_info["center"] = np.array(center, dtype=np.float32)
        meta_info["is_flipped"] = np.array(augm_dict["flip"], dtype=np.float32).reshape(
            -1
        )
        meta_info["rot_angle"] = np.float32(augm_dict["rot"]).reshape(-1)

        # image is valid or not
        # right hand is valid or not
        # each joint is valid or not
        targets["is_valid"] = torch.FloatTensor(np.array([float(right_valid)]))
        targets["right_valid"] = torch.FloatTensor(np.array([float(right_valid)]))
        targets["joints_valid_r"] = torch.ones(21).float() * targets["right_valid"]

        return inputs, targets, meta_info

    def _load_data(self, args, split):
        self.txn = sys_utils.fetch_lmdb_reader("./data/images.lmdb")
        self.args = args
        self.split = split
        self.aug_data = split.endswith("train")
        self.normalize_img = Normalize(mean=args.img_norm_mean, std=args.img_norm_std)

        if "train" in split:
            mode = "train"
        elif "val" in split:
            mode = "val"
        elif "test" in split:
            mode = "test"
        if "fulltrain" in split:
            mode = "fulltrain"

        data_p = f"./data/{mode}.npy"
        logger.info(f"Loading {data_p}")
        data = np.load(data_p, allow_pickle=True).item()

        self.imgnames = data["imgnames"]
        self.data_cam = data["cam_coord"]
        self.data_params = data["params"]
        self.data_2d = data["2d"]
        self.bbox = data["bbox"]
        self.data_meta = data["meta"]
        self.query_names = data["query_name"]

        self.indices = list(range(len(self.imgnames)))

    def __init__(self, args, split):
        self._load_data(args, split)
        # downsample data for development
        self.indices = dataset_utils.downsample_index(self.indices, split)

        logger.info(
            f"ImageDataset Loaded {self.split} split, num samples {len(self.imgnames)}"
        )

    def __len__(self):
        return len(self.indices)
