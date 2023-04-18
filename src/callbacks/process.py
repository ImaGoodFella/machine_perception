from mp_lib.geometry import estimate_translation_k
import mp_lib.data_utils as data_utils
import mp_lib.geometry as geometry


def process_data(models, inputs, targets, meta_info, mode, args):
    img_res = 224
    K = meta_info["intrinsics"]
    gt_pose_r = targets["mano.pose.r"]  # MANO pose parameters
    gt_betas_r = targets["mano.beta.r"]  # MANO beta parameters

    gt_kp2d_b = targets["object.kp2d.norm.b"]  # 2D keypoints of object
    gt_object_rot = targets["object.rot"].view(-1, 3)  # global orientation of object

    # pose the object in canonical space (object rotated but not translated)
    out = models["object_head"].object_tensors.forward(
        global_orient=gt_object_rot,
        query_names=meta_info["query_names"],
        fwd_meshes=True,
    )

    # map hand and object keypoints from camera coord to object canonical space
    # since R, T is used, relative distance is preserved

    kp3d_b_cano = out["kp3d"]  # 3d keypoints of object in canonical space
    # find the rigid transformation from object camera space (full image resolution) to canonical space
    R0, T0 = geometry.batch_solve_rigid_tf(targets["object.kp3d.full.b"], kp3d_b_cano)
    # map 3d joints of hand in camera space (full image resolution) to canonical space
    joints3d_r0 = geometry.rigid_tf_torch_batch(targets["mano.j3d.full.r"], R0, T0)

    # pose MANO in MANO canonical space (hand posed but not translated)
    gt_out_r = models["mano_r"](
        betas=gt_betas_r,
        hand_pose=gt_pose_r[:, 3:],
        global_orient=gt_pose_r[:, :3],
        transl=None,
    )
    gt_model_joints_r = gt_out_r.joints
    gt_vertices_r = gt_out_r.vertices
    gt_root_cano_r = gt_out_r.joints[:, 0]

    # map MANO mesh to object canonical space
    Tr0 = (joints3d_r0 - gt_model_joints_r).mean(dim=1)
    gt_model_joints_r = joints3d_r0
    gt_vertices_r += Tr0[:, None, :]

    # denormalize 2d keypoints for object
    gt_kp2d_b_cano = data_utils.unormalize_kp2d(gt_kp2d_b, img_res)

    # estimate translation from object canonical space to camera space (patch resolution)
    gt_transl = estimate_translation_k(
        kp3d_b_cano,
        gt_kp2d_b_cano,
        meta_info["intrinsics"].cpu().numpy(),
        use_all_joints=True,
        pad_2d=True,
    )

    # move to camera coord (patch resolution)
    gt_vertices_r = gt_vertices_r + gt_transl[:, None, :]
    gt_model_joints_r = gt_model_joints_r + gt_transl[:, None, :]

    gt_kp3d_o = out["kp3d"] + gt_transl[:, None, :]
    gt_bbox3d_o = out["bbox3d"] + gt_transl[:, None, :]

    # roots
    gt_root_cam_patch_r = gt_model_joints_r[:, 0]
    gt_cam_t_r = gt_root_cam_patch_r - gt_root_cano_r
    gt_cam_t_o = gt_transl

    targets.register("mano.cam_t.r", gt_cam_t_r)
    targets.register("object.cam_t", gt_cam_t_o)

    avg_focal_length = (K[:, 0, 0] + K[:, 1, 1]) / 2.0

    # Convert root of camera from xyz to [s, tx, ty] (scale, and 2d translation) based on the weak perspective model
    gt_cam_t_wp_r = geometry.perspective_to_weak_perspective_torch(
        gt_cam_t_r, avg_focal_length, img_res
    )

    gt_cam_t_wp_o = geometry.perspective_to_weak_perspective_torch(
        gt_cam_t_o, avg_focal_length, img_res
    )

    # cameras in weak perspective form (denoted by wp)
    targets.register("mano.cam_t.wp.r", gt_cam_t_wp_r)
    targets.register("object.cam_t.wp", gt_cam_t_wp_o)

    targets.register("object.cam_t.kp3d.b", gt_transl)

    # vertices, joints, keypoints, 3D bounding boxes in camera space (patch resolution)
    targets.register("mano.v3d.cam.r", gt_vertices_r)
    targets.register("mano.j3d.cam.r", gt_model_joints_r)
    targets.register("object.kp3d.cam", gt_kp3d_o)
    targets.register("object.bbox3d.cam", gt_bbox3d_o)

    out = models["object_head"].object_tensors.forward(
        global_orient=gt_object_rot,
        query_names=meta_info["query_names"],
        fwd_meshes=True,
    )

    # Object vertices in camera coord (patch resolution)
    targets.register("object.v.cam", out["v"] + gt_transl[:, None, :])

    # number of vertices per example in a batch
    targets.register("object.v_len", out["v_len"])

    # faces and number of faces per example in a batch
    targets.register("object.f", out["f"])
    targets.register("object.f_len", out["f_len"])

    return inputs, targets, meta_info
