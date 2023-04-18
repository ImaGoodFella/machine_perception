import torch.nn as nn

from src.utils.rigid_tensors import RigidTensors
import mp_lib.data_utils as data_utils
from mp_lib.unidict import xdict
import mp_lib.geometry as geometry


class RigidHead(nn.Module):
    def __init__(self, focal_length, img_res):
        super().__init__()
        self.object_tensors = RigidTensors()
        self.focal_length = focal_length
        self.img_res = img_res

    def forward(
        self,
        rot,
        query_names,
        cam,
        K,
        transl=None,
        fwd_meshes=False,
    ):
        if self.object_tensors.dev != rot.device:
            self.object_tensors.to(rot.device)

        out = self.object_tensors.forward(rot, transl, query_names, fwd_meshes)

        # after adding relative transl
        bbox3d = out["bbox3d"]
        kp3d = out["kp3d"]

        # right hand translation
        avg_focal_length = (K[:, 0, 0] + K[:, 1, 1]) / 2.0
        cam_t = geometry.weak_perspective_to_perspective_torch(
            cam, focal_length=avg_focal_length, img_res=self.img_res, min_s=0.1
        )

        # camera coord
        bbox3d_cam = bbox3d + cam_t[:, None, :]
        kp3d_cam = kp3d + cam_t[:, None, :]

        # 2d keypoints
        kp2d = geometry.project2d_batch(K, kp3d_cam)
        bbox2d = geometry.project2d_batch(K, bbox3d_cam)

        kp2d = data_utils.normalize_kp2d(kp2d, self.img_res)
        bbox2d = data_utils.normalize_kp2d(bbox2d, self.img_res)
        num_kps = kp2d.shape[1] // 2

        output = xdict()
        output.register("rot", rot)
        if transl is not None:
            # relative transl
            output.register("transl", transl)  # meter

        output.register("cam_t.wp", cam)
        output.register("cam_t", cam_t)
        output.register("kp3d", kp3d)
        output.register("bbox3d", bbox3d)
        output.register("bbox3d.cam", bbox3d_cam)
        output.register("kp3d.cam", kp3d_cam)
        output.register("kp2d.norm", kp2d)
        output.register("kp2d.norm.t", kp2d[:, :num_kps])
        output.register("kp2d.norm.b", kp2d[:, num_kps:])
        output.register("bbox2d.norm.t", bbox2d[:, :8])
        output.register("bbox2d.norm.b", bbox2d[:, 8:])

        if fwd_meshes:
            output.register("v.cam", out["v"] + cam_t[:, None, :])
            output.register("v_len", out["v_len"])
            output.register("f", out["f"])
            output.register("f_len", out["f_len"])
        return output
