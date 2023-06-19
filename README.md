# Machine Perception Project

This project endeavors to reconstruct the intricate surface of a human hand engaged in interaction, solely from a single RGB image. To capture the three-dimensional form of the hand, we employ the widely recognized MANO model. Our approach employs a comprehensive framework that initially extracts image features via a pre-trained backbone model. Subsequently, a Multilayer Perceptron (MLP) is employed to predict the camera translation, shape, and pose parameters of the MANO model.

Final Report: [Click here](/document.pdf)

# Reproducing the results

## Training
`python scripts/train.py --trainsplit train --valsplit minival`

## Evaluation
`python scripts/test.py --load_ckpt logs/5542634e4/checkpoints/last.ckpt`
