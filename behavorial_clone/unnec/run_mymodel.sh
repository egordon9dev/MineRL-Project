#!/bin/sh
#############################################################################
# TODO: Modify the hyperparameters such as hidden layer dimensionality, 
#       number of epochs, weigh decay factor, momentum, batch size, learning 
#       rate mentioned here to achieve good performance
#############################################################################
python -u train.py \
    --model mymodel \
    --kernel-size 3 \
    --hidden-dim 64 \
    --epochs 5 \
    --weight-decay 0.5 \
    --momentum 0.9 \
    --batch-size 512 \
    --lr 0.001 | tee mymodel.log
#############################################################################
#                             END OF YOUR CODE                              #
#############################################################################

# python -u train.py \
#     --model mymodel \
#     --kernel-size 5 \
#     --hidden-dim 32 \
#     --epochs 4 \
#     --weight-decay 0.5 \
#     --momentum 0.6 \
#     --batch-size 256 \
#     --lr 0.001 | tee mymodel.log
