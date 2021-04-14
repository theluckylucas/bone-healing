# bone-healing
Prediction of the spatial transformation of T2 hyper-intense tissue surrounding a biodegradable implant during bone healing phases.

This repository includes the source files used for MIDL short paper submission 2021.

## Visual Comparison

The following visualises the center slice of the segmentations for each of the 8 subjects (rows) and timepoints (colums) for the setting of t=1 (two observation, predicting the following five):

![alt text](https://github.com/theluckylucas/bone-healing/blob/main/comparison.png?raw=true)

## Config used for RNN-full

```
cuda="cuda:0"
loss="0.0357 0.0714 0.1071 0.1429 0.1786 0.2143 0.25"
resampling="0.375 0.375 0.4"
channels="64"
celllen="3"
lossgamma="0"
lossdelta="0.001"
epochs="51"
batchsize="3"
accumulate="1"
loocv=1

output="~/exps/midl${jobid}/midl${jobid}"
train=( "100_2 101_1 101_2 102_1 102_2 103_1 103_2" "100_1 101_1 101_2 102_1 102_2 103_1 103_2" "100_1 100_2 101_2 102_1 102_2 103_1 103_2" "100_1 100_2 101_1 102_1 102_2 103_1 103_2" "100_1 100_2 101_1 101_2 102_2 103_1 103_2" "100_1 100_2 101_1 101_2 102_1 103_1 103_2" "100_1 100_2 101_1 101_2 102_1 102_2 103_2" "100_1 100_2 101_1 101_2 102_1 102_2 103_1" )
valid=( "100_1" "100_2" "101_1" "101_2" "102_1" "102_2" "103_1" "103_2" )

python ~/Train_ConvGRUCellSequence.py --pin_memory --device $cuda --loss_alpha 0 --loss_beta 0 --output_basename $output --n_channels_input 18 --loss_weight $loss --batchsize $batchsize --accumulate $accumulate --cell_len $celllen --loss_gamma $lossgamma --loss_delta $lossdelta --resampling $resampling --n_channels_hidden $channels --n_epochs $epochs --subjects_train ${train[loocv]} --subject_valid ${valid[loocv]}
```

## Layers

|# Layer | # Ch in | # Ch out | Kernel | Stride | Padding |
|---|---|---|---|---|---|
| 01 | Conv | 18 | 64 | 3 | 1 | 0 |
| 02 | ReLU ||||||
| 03 | Conv | 64 | 96 | 3 | 1 | 0 |
| 04 | Norm ||||||
| 05 | ReLU ||||||
| 06 | Pool | | | 2 | 2 | 0 |
| 07 | Conv | 96 | 128 | 3 | 1 | 0 |
| 08 | Norm ||||||
| 09 | ReLU ||||||
| 10 | Conv | 128 | 160 | 3 | 1 | 0 |
| 11 | Norm ||||||
| 12 | ReLU ||||||
| 13 | Pool | | | 2 | 2 | 0 |
| 14 | Conv | 160 | 64 | 3 | 1 | 0 |
| 15 | Norm ||||||
| 16 | ReLU ||||||
| 17 | ConvGRU | 146 | 64 | 3 | 1 | 1 |
| 18 | ConvGRU | 128 | 64 | 3 | 1 | 1 |
| 19 | ConvGRU | 128 | 64 | 3 | 1 | 1 |
| 20 | Conv | 64 | 3 | 1 | 1 | 0 |
| 21 | Tanh ||||||

## References:

https://openreview.net/pdf?id=t55nGIO-hl (MIDL 2021 short paper submission)
