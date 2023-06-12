import comet_ml
import sys
import torch
import pytorch_lightning as pl
from loguru import logger
from pytorch_lightning.callbacks import ModelCheckpoint, ModelSummary
from pprint import pformat

sys.path.append(".")

from src.utils.const import args
import src.utils.factory as factory
from mp_lib.torch_utils import reset_all_seeds
import mp_lib.comet_utils as comet_utils


def main(args):

    args.batch_size=8
    args.num_workers=128
    #just checks for some input in utils.const
    if args.experiment is not None:
        comet_utils.log_exp_meta(args)
    #setting random seeds for comparision
    reset_all_seeds(args.seed)
    #try to use GPU
    device = "cuda" if torch.cuda.is_available() else "cpu"
    #gets the model - from several files back
    wrapper = factory.fetch_model(args).to(device)
    #load and save checkpoints
    if args.ckpt_p != "":
        ckpt_p = args.ckpt_p
        ckpt = torch.load(ckpt_p)
        wrapper.load_state_dict(ckpt["state_dict"])
        logger.info(f"Loaded weights from {ckpt_p}")

    #config model
    wrapper.model.object_head.object_tensors.to(device)

    #fitting config
    ckpt_callback = ModelCheckpoint(
        monitor="loss__val",
        verbose=True,
        save_top_k=5,
        mode="min",
        every_n_epochs=args.eval_every_epoch,
        save_last=True,
    )

    pbar_cb = pl.callbacks.progress.TQDMProgressBar(refresh_rate=1)
    model_summary_cb = ModelSummary(max_depth=3)
    #visualising stuff
    callbacks = [ckpt_callback, pbar_cb, model_summary_cb]
    #setting up the trainer
    trainer = pl.Trainer(
        gradient_clip_val=args.grad_clip,
        gradient_clip_algorithm="norm",
        accumulate_grad_batches=args.acc_grad,
        gpus=1,
        logger=None,
        min_epochs=args.num_epoch,
        max_epochs=args.num_epoch,
        callbacks=callbacks,
        log_every_n_steps=args.log_every,
        default_root_dir=args.log_dir,
        check_val_every_n_epoch=args.eval_every_epoch,
        num_sanity_val_steps=0,
        enable_model_summary=False,
    )

    #resetting some parameters again
    reset_all_seeds(args.seed)
    train_loader = factory.fetch_dataloader(args, "train")
    if args.valsplit == "none":
        val_loader = []
    else:
        val_loader = [factory.fetch_dataloader(args, "val")]

    logger.info(f"Hyperparameters: \n {pformat(args)}")
    logger.info("*** Started training ***")
    reset_all_seeds(args.seed)
    #finally fitting
    trainer.fit(wrapper, train_loader, val_loader, ckpt_path=args.ckpt_p)


if __name__ == "__main__":
    main(args)
