import torch
import torch.nn as nn
from src.datasets.dataset_utils import subtract_root_batch


mse_loss = nn.MSELoss(reduction="none")


def keypoint_3d_loss(pred_keypoints_3d, gt_keypoints_3d, criterion, jts_valid):
    """
    Compute 3D keypoint loss for the examples that 3D keypoint annotations are available.
    The loss is weighted by the confidence.
    """

    gt_root = gt_keypoints_3d[:, :1, :]
    gt_keypoints_3d = gt_keypoints_3d - gt_root
    pred_root = pred_keypoints_3d[:, :1, :]
    pred_keypoints_3d = pred_keypoints_3d - pred_root

    return joints_loss(pred_keypoints_3d, gt_keypoints_3d, criterion, jts_valid)


def hand_kp3d_loss(pred_3d, gt_3d, criterion, jts_valid):
    pred_3d_ra = subtract_root_batch(pred_3d, root_idx=0)
    gt_3d_ra = subtract_root_batch(gt_3d, root_idx=0)
    loss_kp = keypoint_3d_loss(
        pred_3d_ra, gt_3d_ra, criterion=criterion, jts_valid=jts_valid
    )
    return loss_kp


def vector_loss(pred_vector, gt_vector, criterion, is_valid=None):
    dist = criterion(pred_vector, gt_vector)
    if is_valid.sum() == 0:
        return torch.zeros((1)).to(gt_vector.device)
    if is_valid is not None:
        valid_idx = is_valid.long().bool().view(-1)
        dist = dist[valid_idx]
    loss = dist.mean().view(-1)
    return loss


def joints_loss(pred_vector, gt_vector, criterion, jts_valid):
    dist = criterion(pred_vector, gt_vector)
    if jts_valid is not None:
        dist = dist * jts_valid[:, :, None]

    # return 0 if all invalid
    if jts_valid.sum() == 0:
        return torch.zeros((1)).to(gt_vector.device)
    else:
        loss = dist.mean().view(-1)
    return loss


def mano_loss(pred_rotmat, pred_betas, gt_rotmat, gt_betas, criterion, is_valid=None):
    loss_regr_pose = vector_loss(pred_rotmat, gt_rotmat, criterion, is_valid)
    loss_regr_betas = vector_loss(pred_betas, gt_betas, criterion, is_valid)
    return loss_regr_pose, loss_regr_betas


def loss_obj_rot(pred, gt):
    criterion = mse_loss
    pred_rot = pred["object.rot"].view(-1, 3).float()
    gt_rot = gt["object.rot"].view(-1, 3).float()
    is_valid = gt["is_valid"]
    loss_rot = vector_loss(pred_rot, gt_rot, criterion, is_valid)
    loss_dict = {
        "loss/object/rot": loss_rot,
    }
    return loss_dict


loss_fn_dict = {
    "loss.obj.rot": loss_obj_rot,
}
