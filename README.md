# Reproducing the results

## Training
`python scripts/train.py --trainsplit train --valsplit minival`

## Evaluation
`python scripts/test.py --load_ckpt logs/5542634e4/checkpoints/last.ckpt`
