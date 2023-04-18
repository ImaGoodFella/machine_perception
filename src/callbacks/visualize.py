import trimesh
import numpy as np

from mp_lib.data_utils import denormalize_images
from mp_lib.rend_utils import color2material
import mp_lib.torch_utils as torch_utils
from mp_lib.torch_utils import unpad_vtensor
import mp_lib.vis_utils as vis_utils


mesh_color_dict = {
    "right": [200, 200, 250],
    "object": [144, 250, 100],
}


def visualize_rend(
    renderer,
    vertices_r,
    mano_faces_r,
    vertices_o,
    faces_o,
    r_valid,
    K,
    img,
):
    mesh_r = trimesh.Trimesh(vertices_r, mano_faces_r)
    mesh_o = trimesh.Trimesh(
        vertices=torch_utils.tensor2np(vertices_o), faces=torch_utils.tensor2np(faces_o)
    )

    # render only valid meshes
    meshes = []
    mesh_names = []
    if r_valid:
        meshes.append(mesh_r)
        mesh_names.append("right")

    meshes = meshes + [mesh_o]
    mesh_names = mesh_names + ["object"]
    materials = [color2material(mesh_color_dict[name]) for name in mesh_names]

    render_img_img = renderer.render_meshes_pose(
        cam_transl=None,
        meshes=meshes,
        image=img,
        materials=materials,
        sideview_angle=None,
        K=K,
    )
    render_img_list = [render_img_img]
    for angle in list(np.linspace(45, 300, 3)):
        render_img_angle = renderer.render_meshes_pose(
            cam_transl=None,
            meshes=meshes,
            image=None,
            materials=materials,
            sideview_angle=angle,
            K=K,
        )
        render_img_list.append(render_img_angle)
    render_img = np.concatenate(render_img_list, axis=0)
    return render_img


def visualize_rends(renderer, vis_dict, max_examples):
    image_ids = vis_dict["vis.image_ids"]
    right_valid = vis_dict["targets.right_valid"].bool()
    images = vis_dict["vis.images"].permute(0, 2, 3, 1).numpy()
    gt_vertices_r_cam = vis_dict["targets.mano.v3d.cam.r"]
    mano_faces_r = vis_dict["meta_info.mano.faces.r"]
    pred_vertices_r_cam = vis_dict["pred.mano.v3d.cam.r"]

    # unpack object
    gt_obj_v_cam = unpad_vtensor(
        vis_dict["targets.object.v.cam"], vis_dict["targets.object.v_len"]
    )  # meter

    # object uses GT for visualization
    # This is because we don't need to estimate object.
    # However, if one wants to use estimate objects to improve hands, this can be modified and to visualize the object prediction.
    pred_obj_v_cam = unpad_vtensor(
        vis_dict["targets.object.v.cam"], vis_dict["targets.object.v_len"]
    )
    pred_obj_f = unpad_vtensor(
        vis_dict["targets.object.f"], vis_dict["targets.object.f_len"]
    )
    K = vis_dict["meta_info.intrinsics"]

    im_list = []
    # rendering
    for idx in range(min(len(image_ids), max_examples)):
        # right hand is valid or not
        r_valid = right_valid[idx]
        # intrinsics
        K_i = K[idx]
        # image name
        image_id = image_ids[idx]

        # meshes
        # render
        image_list = []
        image_list.append(images[idx])
        image_gt = visualize_rend(
            renderer,
            gt_vertices_r_cam[idx],
            mano_faces_r,
            gt_obj_v_cam[idx],
            pred_obj_f[idx],
            r_valid,
            K_i,
            images[idx],
        )
        image_list.append(image_gt)

        # prediction
        image_pred = visualize_rend(
            renderer,
            pred_vertices_r_cam[idx],
            mano_faces_r,
            pred_obj_v_cam[idx],
            pred_obj_f[idx],
            r_valid,
            K_i,
            images[idx],
        )
        image_list.append(image_pred)
        image_pred = vis_utils.im_list_to_plt(
            image_list,
            figsize=(15, 8),
            title_list=["input image", "GT", "pred"],
        )
        im_list.append(
            {
                "fig_name": f"{image_id}",
                "im": image_pred,
            }
        )
    return im_list


def visualize_all(vis_dict, max_examples, renderer, postfix):
    image_ids = [
        "/".join(key.split("/")[-5:]).replace(".jpg", "")
        for key in vis_dict["meta_info.imgname"]
    ]
    images = denormalize_images(vis_dict["inputs.img"])
    vis_dict.remove("inputs.img")
    vis_dict.register("vis.images", images)
    vis_dict.register("vis.image_ids", image_ids)

    im_list = []
    im_list += visualize_rends(renderer, vis_dict, max_examples)

    # post fix image list
    im_list_postfix = []
    for im in im_list:
        im["fig_name"] += postfix
        im_list_postfix.append(im)
    return im_list
