# Reproducing the results

# Training
CUDA_VISIBLE_DEVICES=0 python scripts/train.py --trainsplit train --valsplit val

# Evaluation
CUDA_VISIBLE_DEVICES=0 python scripts/test.py --load_ckpt logs/a806fa191/checkpoints/last.ckpt
