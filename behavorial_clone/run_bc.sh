#!/bin/sh
#############################################################################
# TODO: Modify the hyperparameters such as hidden layer dimensionality, 
#       number of epochs, weigh decay factor, momentum, batch size, learning 
#       rate mentioned here to achieve good performance
#############################################################################
python -u train_bc.py \
    --model mineNet \
    --kernel-size 5 \
    --hidden-dim 64 \
    --epochs 2 \
    --weight-decay 0.5 \
    --momentum 0.5 \
    --batch-size 128 \
    --lr 0.0001 | tee convnet.log
#############################################################################
#                             END OF YOUR CODE                              #
#############################################################################
