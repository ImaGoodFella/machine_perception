# rigid object; no articulation

import torch
import sys
import trimesh
import torch.nn as nn
import numpy as np
import os.path as op
from easydict import EasyDict

sys.path = [".."] + sys.path
import mp_lib.torch_utils as torch_utils
from mp_lib.geometry import axis_angle_to_quaternion, quaternion_apply
from mp_lib.torch_utils import pad_tensor_list


def construct_obj(object_model_p, cano_kps):
    # load vtemplate
    mesh_p = op.abspath(op.join(object_model_p, "simple_4000.obj"))

    assert op.exists(mesh_p), f"Not found: {mesh_p}"

    mesh = trimesh.exchange.load.load_mesh(mesh_p, process=False)
    mesh_v = mesh.vertices

    import numpy as np

    np.random.seed(1)
    rand_idx = np.random.permutation(mesh_v.shape[0])[:600]
    sub_v = mesh_v[rand_idx].copy()

    mesh_f = torch.LongTensor(mesh.faces)

    vsk = object_model_p.split("/")[-1]

    np.random.seed(1)
    obj = EasyDict()
    obj.name = vsk
    obj.obj_name = "".join([i for i in vsk if not i.isdigit()])
    obj.v = torch.FloatTensor(mesh_v)
    obj.f = torch.LongTensor(mesh_f)
    obj.v_sub = torch.FloatTensor(sub_v)
    obj.bbox_bottom = torch.FloatTensor(cano_kps[vsk])
    obj.kp_bottom = torch.FloatTensor(cano_kps[vsk])
    return obj


def construct_obj_tensors():
    cano_kps = np.load("./data/cano_kps.npy", allow_pickle=True).item()
    object_names = list(cano_kps.keys())

    obj_list = []
    for k in object_names:
        object_model_p = f"./data/object_models/{k}"
        obj = construct_obj(object_model_p, cano_kps)
        obj_list.append(obj)

    bbox_bottom_list = []
    kp_bottom_list = []
    v_list = []
    v_sub_list = []
    f_list = []
    for obj in obj_list:
        v_list.append(obj.v)
        v_sub_list.append(obj.v_sub)
        f_list.append(obj.f)

        bbox_bottom_list.append(obj.bbox_bottom)
        kp_bottom_list.append(obj.kp_bottom)

    v_list, v_len_list = pad_tensor_list(v_list)
    v_sub_list = torch.stack(v_sub_list, dim=0)

    max_len = v_len_list.max()
    mask = torch.zeros(len(obj_list), max_len)
    for idx, vlen in enumerate(v_len_list):
        mask[idx, :vlen] = 1.0

    f_list, f_len_list = pad_tensor_list(f_list)

    bbox_bottom_list = torch.stack(bbox_bottom_list, dim=0)
    kp_bottom_list = torch.stack(kp_bottom_list, dim=0)

    obj_tensors = {}
    obj_tensors["names"] = object_names

    obj_tensors["v"] = v_list.float()
    obj_tensors["v_sub"] = v_sub_list.float()
    obj_tensors["v_len"] = v_len_list
    obj_tensors["f"] = f_list
    obj_tensors["f_len"] = f_len_list

    obj_tensors["mask"] = mask
    obj_tensors["bbox_bottom"] = bbox_bottom_list.float()
    obj_tensors["kp_bottom"] = kp_bottom_list.float()
    return obj_tensors


class RigidTensors(nn.Module):
    def __init__(self):
        super(RigidTensors, self).__init__()
        obj_tensors = construct_obj_tensors()
        self.obj_tensors = torch_utils.dict2dev(obj_tensors, "cpu")
        self.dev = None

    def forward(self, global_orient, query_names, fwd_meshes=False):
        out = self.forward_batch(global_orient, query_names, fwd_meshes)
        return out

    def to(self, dev):
        self.obj_tensors = torch_utils.dict2dev(self.obj_tensors, dev)
        self.dev = dev

    def forward_batch(
        self, global_orient: (None, torch.Tensor), query_names: list, fwd_meshes: bool
    ):

        batch_size = global_orient.shape[0]
        assert global_orient.shape == (batch_size, 3)
        assert len(query_names) == batch_size

        obj_idx = np.array(
            [self.obj_tensors["names"].index(name) for name in query_names]
        )

        # canonical points -- start
        if fwd_meshes:
            f_tensor = self.obj_tensors["f"][obj_idx]
            f_len_tensor = self.obj_tensors["f_len"][obj_idx]
            v_len_tensor = self.obj_tensors["v_len"][obj_idx]
            max_len = v_len_tensor.max()
            mask = self.obj_tensors["mask"][obj_idx][:, :max_len]

            v_bottom_tensor = self.obj_tensors["v"][obj_idx][:, :max_len].clone()
            v_sub_tensor = self.obj_tensors["v_sub"][obj_idx]

        # m
        bbox_bottom_tensor = self.obj_tensors["bbox_bottom"][obj_idx]
        kp_bottom_tensor = self.obj_tensors["kp_bottom"][obj_idx]
        # canonical points -- end

        # global orientation
        quat_global = axis_angle_to_quaternion(global_orient.view(-1, 3))

        # apply transform
        if fwd_meshes:
            v_rot_tensor = quaternion_apply(quat_global[:, None, :], v_bottom_tensor)
            v_sub_rot_tensor = quaternion_apply(quat_global[:, None, :], v_sub_tensor)

        bbox_rot_tensor = quaternion_apply(quat_global[:, None, :], bbox_bottom_tensor)
        kp_rot_tensor = quaternion_apply(quat_global[:, None, :], kp_bottom_tensor)

        out = {}
        # all output should be in meter
        if fwd_meshes:
            out["v"] = v_rot_tensor  # vertices
            out["v_sub"] = v_sub_rot_tensor  # sub vertices
            out["v_len"] = v_len_tensor  # num vertices per example in the batch

            out["f"] = f_tensor  # faces
            out["f_len"] = f_len_tensor  # num faces per example in the batch
            out["mask"] = mask  # 1 or 0 indicating whether the vertex is valid

        out["bbox3d"] = bbox_rot_tensor
        out["kp3d"] = kp_rot_tensor
        return out
