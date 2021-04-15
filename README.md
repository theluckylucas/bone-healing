# bone-healing
Prediction of the spatial transformation of T2 hyper-intense tissue surrounding a biodegradable implant during bone healing phases.

This repository includes the source files used for MIDL short paper submission 2021.

## Visual Comparison of early observations' prediction

The following image visualises the axial center slice of the segmentations for each of the 8 rats (rows) and timepoints (colums) for the setting of t=1 (two observations, predicting the following five) from the LOOCV. It demonstrates the fact that the inflammation process peaks up to day 3 before it recedes, leading to a false prediction when linearly extrapolating only from the first two observations.

![Comparison of SDM extrapolation with the ground truth and the RNN predictions](https://github.com/theluckylucas/bone-healing/blob/main/comparison.png?raw=true)

## Setup

The source code has been run on Python 3.6 with the pip environment of https://github.com/theluckylucas/bone-healing/blob/main/requirements.txt and a closed dataset from [MHH Institute for Laboratory Animal Science](https://www.mhh.de/tierlabor).

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

for i in "${!valid[@]}"; do
    echo
    echo "Run LOOCV for case ${valid[i]}"
    python /home/lucaschr/TemporalStorage/learn-tvvf/Train_ConvGRUCellSequence.py --pin_memory --device $cuda --loss_alpha 0 --loss_beta 0 --output_basename $output --n_channels_input 18 --loss_weight $loss --batchsize $batchsize --accumulate $accumulate --cell_len $celllen --loss_gamma $lossgamma --loss_delta $lossdelta --resampling $resampling --n_channels_hidden $channels --n_epochs $epochs --subjects_train ${train[i]} --subject_valid ${valid[i]}
    echo
done
```

The training run black-outs (= sets to zero) the T2 images and SDMs in each batch's input tensor at a random timepoint between 1 to 6 (the following timepoints) to simulate only a limited number of observations, i.e. it trains the setting of a single observation and predict the remaining six, two observations and predict the remaining five, and so on... - up to six observations but the last.

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

## Pre-processing of dataset

A semi-automated region-growing segmentation (Mevislab [3]) has been conducted to label the inflammation-associated hyper-intense tissue at the fracture site in the T2 image in each of the 7 timepoints. The images of the time points have been registered with AirLab [2] using a diffeomorphic B-spline transformation in a bootstrap manner, i.e. first register d3 onto d0, then d7 on a template of d0 and d3, and so on. For training, these images have been augmented by random flips as well as random local deformations.

## References:

[1] https://openreview.net/pdf?id=t55nGIO-hl (MIDL 2021 short paper submission)

[2] https://github.com/airlab-unibas/airlab (Airlab autograd registration toolbox)

[3] https://www.mevislab.de/ (Medical image processing framework)
