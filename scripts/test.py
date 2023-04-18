import comet_ml
import torch
import json
from loguru import logger
import os.path as op
from tqdm import tqdm
import sys

sys.path.append(".")
import mp_lib.torch_utils as torch_utils
import src.utils.factory as factory


def main(args):
    # set flags
    args.valsplit = "test"
    args.exp_key = args.load_ckpt.split("/")[-3]
    out_dir = args.load_ckpt.split("checkpoints")[0]
    device = "cuda:0"

    # load model
    wrapper = factory.fetch_model(args).to(device)
    assert args.load_ckpt != ""
    ckpt = torch.load(args.load_ckpt)
    wrapper.load_state_dict(ckpt["state_dict"])
    logger.info(f"Loaded weights from {args.load_ckpt}")
    wrapper.eval()
    wrapper.to(device)
    wrapper.model.object_head.object_tensors.to(device)

    # load data
    out_list = []
    test_loader = factory.fetch_dataloader(args, "test")
    with torch.no_grad():
        for idx, batch in tqdm(enumerate(test_loader), total=len(test_loader)):
            batch = torch_utils.thing2dev(batch, device)
            inputs, targets, meta_info = batch
            out_dict = wrapper.model.inference(inputs["img"], meta_info["intrinsics"])
            out_dict["imgnames"] = meta_info["imgname"]
            out_list.append(out_dict)
    jts3d_pred = torch.cat([out["j3d.cam.r"] for out in out_list], dim=0).numpy()
    imgnames = sum([out["imgnames"] for out in out_list], [])

    # save predictions
    pred_dict = {}
    for idx in range(len(imgnames)):
        key = imgnames[idx]
        val = jts3d_pred[idx].tolist()
        pred_dict[key] = val
    out_p = op.join(out_dir, "pred.json")
    with open(out_p, "w") as f:
        json.dump(pred_dict, f, indent=4)
    logger.info("Saved predictions to {}".format(out_p))
    logger.info("Done")


if __name__ == "__main__":
    from src.utils.const import args

    main(args)
