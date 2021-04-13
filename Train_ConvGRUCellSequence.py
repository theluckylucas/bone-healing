#!/usr/bin/env python
# coding: utf-8

# In[10]:


import scipy
import argparse
import numpy as np
import nibabel as nib
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import data
import metrics
import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt
import warnings
warnings.filterwarnings('ignore')

from GRU import ConvGRUCellSequenceImplant
from train_rnn import get_title, iterate

from dreg import IsotropicTVRegulariser




parser = argparse.ArgumentParser(description='Config', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--num_workers', type=int, default=4)
parser.add_argument('--device', type=str, default="cuda:0")
parser.add_argument('--convgru_kernel', type=int, default=3)
parser.add_argument('--pin_memory', action="store_true")
parser.add_argument('--num_clinical_input', type=int, default=1)
parser.add_argument('--cell_len', type=int, default=1, help="GRU cell length")
parser.add_argument('--batchsize', type=int, default=2)
parser.add_argument('--n_epochs', type=int, default=51)
parser.add_argument('--accumulate', type=int, default=1, help="accumulate grads")
parser.add_argument('--loss_alpha', type=float, default=.01, help="aux loss factor")
parser.add_argument('--loss_beta', type=float, default=.0, help="regulariser loss factor")
parser.add_argument('--loss_gamma', type=float, default=.00000001, help="volume loss factor")
parser.add_argument('--loss_delta', type=float, default=.001, help="softdice loss factor")
parser.add_argument('--vmax_imgs', type=int, default=15, help="max value for image visualisation")
parser.add_argument('--deform_segs', action="store_true", help="deform segs")
parser.add_argument('--loss_weight', type=float, nargs='+', default=None)
parser.add_argument('--path_file', type=str, default="~/MR/{}_MetBioMat3_20??????_{}_{}/{}_MetBioMat3_20??????_{}_{}????????-??????.nii.gz")
parser.add_argument('--timepoints', type=str, nargs='+', default=["d0", "d3", "d7", "d14", "d21", "d28", "d56"])
parser.add_argument('--subjects_train', type=str, nargs='+', default=["100_1", "101_1", "101_2", "102_1", "102_2", "103_1", "103_2"])
parser.add_argument('--subject_valid', type=str, default="100_2")
parser.add_argument('--modalities', type=str, nargs='+', default=[('TurboRARE-T2', 'TurboRARE-T2_bootstrap4d_')])
parser.add_argument('--labels', type=str, nargs='+', default=[('TurboRARE-T2', 'T2_srg_label_bootstrap4d_')])
parser.add_argument('--common_size', type=int, nargs='+', default=[256, 256, 145])
parser.add_argument('--resampling', type=float, nargs='+', default=[.25, .25, .2])
parser.add_argument('--deform_alpha', type=int, default=100)
parser.add_argument('--deform_sigma', type=int, default=4)
parser.add_argument('--deform_random', type=float, default=0.95)
parser.add_argument('--labels_thresh', type=float, default=0.5)
parser.add_argument('--output_basename', type=str, default='rnn_')
parser.add_argument('--n_channels_input', type=int, default=3)
parser.add_argument('--n_channels_hidden', type=int, default=64)
parser.add_argument('--n_channels_output', type=int, default=3)
parser.add_argument('--bspline_kernel', type=int, default=15, help="k=15 for 128x128 images")
parser.add_argument('--tv_weights', type=int, nargs='+', default=[1,1])
parser.add_argument('--blackout', type=int, default=-1)
parser.add_argument('--lr', type=float, default=0.0001)


args = parser.parse_args()
print(args)


device = torch.device(args.device)
zsize=int(args.resampling[2]*args.common_size[2])
assert int(args.resampling[0]*args.common_size[0]) == int(args.resampling[1]*args.common_size[1])
xylength=int(args.resampling[0]*args.common_size[0])
input2d = (zsize == 1)
if input2d:
    convgru_kernel = (1, args.convgru_kernel, args.convgru_kernel)
convgru_kernel = (args.convgru_kernel, args.convgru_kernel, args.convgru_kernel)
zslice = zsize // 2
n_visual_samples = min(4, args.batchsize)
sequence_length = len(args.timepoints)
if 1 > args.blackout or args.blackout > sequence_length:
    blackout = [2, sequence_length]
else:
    blackout = args.blackout


print("device =", device)
print("zsize =", zsize)
print("xylength =", xylength)
print("input2d =", input2d)
print("convgru_kernel =", convgru_kernel)
print("zslice =", zslice)
print("n_visual_samples =", n_visual_samples)
print("sequence_length =", sequence_length)
print("blackout =", blackout)


# In[4]:


train_trafo = [data.HemisphericFlip(),
               data.Resample(scale_factor=args.resampling, mode='linear'),
               data.ElasticDeform3D(alpha=args.deform_alpha, sigma=args.deform_sigma, apply_to_images=True, random=args.deform_random, seed=0),
               data.LabelsToDistanceMaps(threshold=args.labels_thresh),
               data.ToTensor()]
valid_trafo = [data.Resample(scale_factor=args.resampling, mode='linear'),
               data.LabelsToDistanceMaps(threshold=args.labels_thresh),
               data.ToTensor()]

dataset_train = data.ImplantRatDataset3DRegistered(path_pattern=args.path_file,
                                                     timepoints=args.timepoints,
                                                     modalities=args.modalities,
                                                     labels=args.labels,
                                                     subjects=args.subjects_train,
                                                     common_size=args.common_size,
                                                     transform=transforms.Compose(train_trafo))
dataset_valid = data.ImplantRatDataset3DRegistered(path_pattern=args.path_file,
                                                     timepoints=args.timepoints,
                                                     modalities=args.modalities,
                                                     labels=args.labels,
                                                     subjects=[args.subject_valid],
                                                     common_size=args.common_size,
                                                     transform=transforms.Compose(valid_trafo))

ds_train = DataLoader(
    dataset_train,
    batch_size=args.batchsize,
    shuffle=True,
    num_workers=args.num_workers,
    pin_memory=args.pin_memory,
    drop_last=(args.batchsize!=1)
)

ds_valid = DataLoader(
    dataset_valid,
    batch_size=1,
    shuffle=False,
    num_workers=args.num_workers,
    pin_memory=args.pin_memory,
    drop_last=(args.batchsize!=1)
)


rnn = ConvGRUCellSequenceImplant(sequence_length,
                                 depth = zsize,  # z-dim after processing cell hierarchy 
                                 length = xylength, # x/y-dim after processing cell hierarchy (depending on kernel_size, dilation, cell_len)
                                 n_channels_input = args.n_channels_input,
                                 n_channels_hidden = args.n_channels_hidden,
                                 n_channels_output = args.n_channels_output,
                                 n_channels_clinical = args.num_clinical_input,
                                 cell_len = args.cell_len,
                                 kernel_size = convgru_kernel,
                                 dilation=1,
                                 bspline_kernel = args.bspline_kernel,     # 15 for 128 side length
                                 smooth_offsets = True    
                                ).to(device)

print(rnn)

dreg = IsotropicTVRegulariser(args.tv_weights)

def weighted_mse_loss(inputs, target, weight=1):
    return torch.mean(weight * (inputs - target) ** 2)
criterion = weighted_mse_loss  #nn.MSELoss()
softdice = metrics.BatchDiceLoss(args.loss_weight) 


params = [p for p in rnn.parameters() if p.requires_grad]
print('# optimizing params', sum([p.nelement() * p.requires_grad for p in params]), '/ total: RNN', sum([p.nelement() for p in rnn.parameters()]))
validation_metric = scipy.spatial.distance.dice
optimizer = torch.optim.Adam(params, lr=args.lr)


# In[ ]:


loss_train = []
loss_valid = [[], [], [], [], [], [], []]
loss_reg_list= []
loss_train_seg = []
loss_train_aux = []
loss_train_vol = []
loss_train_dcm = []
loss_valid_vol = [[], [], [], [], [], [], []]
loss_valid_aux = [[], [], [], [], [], [], []]
loss_valid_seg = [[], [], [], [], [], [], []]
loss_valid_dcm = [[], [], [], [], [], [], []]
dice_valid_seg = [[], [], [], [], [], [], []]
dice_valid_int = [[], [], [], [], [], [], []]
f1_valid_seg = [[], [], [], [], [], [], []]
f1_valid_int = [[], [], [], [], [], [], []]

for epoch in range(0, args.n_epochs):
    optimizer.zero_grad()
    loss_min = np.inf    

    #scheduler.step()
    f1, axarr1 = plt.subplots(n_visual_samples + 1, 14)
    f2, axarr2 = plt.subplots(n_visual_samples + 1, 14)
    f3, axarr3 = plt.subplots(n_visual_samples + 1, 14)
    f4, axarr4 = plt.subplots(n_visual_samples + 1, 14)
    f5, axarr5 = plt.subplots(n_visual_samples + 1, 14)

    ### Train ###
    # grunet, loss_train, optimizer, axarr
    _,_,_,_,_,blackout_idx = iterate(True,
                        args.accumulate,
                        ds_train,
                        args.batchsize,
                        zsize,
                        zslice,
                        sequence_length,
                        rnn,
                        device,
                        criterion,
                        softdice,
                        optimizer,
                        loss_train,
                        loss_train_seg,
                        loss_train_aux,
                        loss_reg_list,
                        loss_train_vol,
                        loss_train_dcm,
                        axarr1,
                        axarr2,
                        axarr3,
                        axarr4,
                        axarr5,
                        n_visual_samples,
                        0,
                        validation_metric,
                        [],
                        [],
                        [],
                        [],
                        args.loss_alpha,
                        args.loss_beta,
                        args.loss_gamma,
                        args.loss_delta,
                        dreg,
                        args.loss_weight,
                        args.vmax_imgs,
                        args.deform_segs,
                        blackout)


    print('Epoch {:03d}\ttraining loss: {:1.5f} (TV reg loss: {:1.5f})'.format(epoch, loss_train[-1], loss_reg_list[-1]))

    ### Validation monitoring ###

    if epoch % 5 == 0:

        for i in range(1, sequence_length):
    
            ys, ls, ints, gts, imgs, blackout_idx = iterate(False,
                            args.accumulate,
                            ds_valid,
                            1,
                            zsize,
                            zslice,
                            sequence_length,
                            rnn,
                            device,
                            criterion,
                            softdice,
                            optimizer,
                            loss_valid[i],
                            loss_valid_seg[i],
                            loss_valid_aux[i],
                            [],
                            loss_valid_vol[i],
                            loss_valid_dcm[i],
                            axarr1,
                            axarr2,
                            axarr3,
                            axarr4,
                            axarr5,				
                            1,
                            n_visual_samples,
                            validation_metric,
                            dice_valid_seg[i],
                            dice_valid_int[i],
                            f1_valid_seg[i],
                            f1_valid_int[i],
                            args.loss_alpha,
                            args.loss_beta,
                            args.loss_gamma,
                            args.loss_delta,
                            dreg,
                            args.loss_weight,
                            args.vmax_imgs,
                            args.deform_segs,
                            i)
    
            print("    {} \tvalidation batch loss: {:1.5f} = {:1.5f} + {:1.4f}*{:1.4f} + {:1.4f}*{:1.4f} + {:1.8f}*{:1.1f} + {:1.4f}*{:1.2f}".format(
              i,
              loss_valid[i][-1],
              loss_valid_seg[i][-1],
              args.loss_alpha,
              loss_valid_aux[i][-1],
              args.loss_beta,
              0,
              args.loss_gamma,
              loss_valid_vol[i][-1],
              args.loss_delta,
              loss_valid_dcm[i][-1])
            )
            print("       \tvalidation dice preds: predicted sdms = {:0.2f}, interpolated sdms = {:0.2f} (blackout: {})".format(dice_valid_seg[i][-1], dice_valid_int[i][-1], blackout_idx))
            print("       \tvalidation  F1  preds: predicted sdms = {:0.2f}, interpolated sdms = {:0.2f} (blackout: {})".format(f1_valid_seg[i][-1], f1_valid_int[i][-1], blackout_idx))
         
            assert gts.shape[4] == 1
    
            if epoch == 0:
                assert gts.shape[4] == 1
                if i == 2:
                    nib.save(nib.Nifti1Image(gts[:, :, :, :, 0], np.eye(4)), '{}-{}-gt.nii.gz'.format(args.output_basename, args.subject_valid))
                    nib.save(nib.Nifti1Image(imgs[:, :, :, :, 0], np.eye(4)), '{}-{}-im.nii.gz'.format(args.output_basename, args.subject_valid))

                nib.save(nib.Nifti1Image(ints[:, :, :, :, 0], np.eye(4)), '{}-{}-ed_bo{}-{}.nii.gz'.format(args.output_basename, args.subject_valid, i, sequence_length))

                for arr, fig, pth in zip([axarr3, axarr5], [f3, f5], ['{}-{}-LIN_pm_bo{}-{}.png', '{}-{}-LIN_ps_bo{}-{}.png']):
                    for ax in arr.flatten():
                        ax.title.set_fontsize(3)
                        ax.xaxis.set_visible(False)
                        ax.yaxis.set_visible(False)
                    fig.subplots_adjust(hspace=0.05)
                    fig.savefig(pth.format(args.output_basename, args.subject_valid, i, sequence_length), bbox_inches='tight', dpi=600)
         
            if epoch % 50 == 0:
                torch.save(rnn, '{}-{}-NN_e{}.model'.format(args.output_basename, args.subject_valid, epoch))
         
            if epoch % 5 == 0:
         
                assert ys.shape[4] == 1
                nib.save(nib.Nifti1Image(ys[:, :, :, :, 0], np.eye(4)), '{}-{}-pm_bo{}-{}_e{}.nii.gz'.format(args.output_basename, args.subject_valid, i, sequence_length, epoch))
                nib.save(nib.Nifti1Image(ls[:, :, :, :, 0], np.eye(4)), '{}-{}-ps_bo{}-{}_e{}.nii.gz'.format(args.output_basename, args.subject_valid, i, sequence_length, epoch))
         
                for arr, fig, pth in zip([axarr1, axarr2, axarr4], [f1, f2, f4], ['{}-{}-DEB_pm_bo{}-{}_e{}.png', '{}-{}-RNN_pm_bo{}-{}_e{}.png', '{}-{}-RNN_ps_bo{}-{}_e{}.png']):
                    for ax in arr.flatten():
                        ax.title.set_fontsize(3)
                        ax.xaxis.set_visible(False)
                        ax.yaxis.set_visible(False)
                    fig.subplots_adjust(hspace=0.05)
                    fig.savefig(pth.format(args.output_basename, args.subject_valid, i, sequence_length, epoch), bbox_inches='tight', dpi=600)
    
        del f1
        del axarr1
        del f2
        del axarr2
        del f3
        del axarr3
        del f4
        del axarr4
        del f5
        del axarr5
        del fig
        del arr
        del pth

    else:
        for i in range(2, sequence_length):
            loss_valid[i].append(loss_valid[i][-1])
            loss_valid_seg[i].append(loss_valid_seg[i][-1])
            loss_valid_aux[i].append(loss_valid_aux[i][-1])
            loss_valid_vol[i].append(loss_valid_vol[i][-1])
            loss_valid_dcm[i].append(loss_valid_dcm[i][-1])


    if epoch > 0:
        for i in range(2, sequence_length):
            fig, ax1 = plt.subplots()
            epochs = range(1, epoch + 2)
            ax1.plot(epochs, loss_train, 'r-')
            ax1.plot(epochs, loss_train_seg, 'r--')
            ax1.plot(epochs, [v * args.loss_alpha for v in loss_train_aux], 'r:')
            ax1.plot(epochs, [v * args.loss_gamma for v in loss_train_vol], 'r.')
            ax1.plot(epochs, [v * args.loss_delta for v in loss_train_dcm], 'r+')
            ax1.plot(epochs, loss_valid[i], 'b-')
            ax1.plot(epochs, loss_valid_seg[i], 'b--')
            ax1.plot(epochs, [v * args.loss_alpha for v in loss_valid_aux[i]], 'b:')
            ax1.plot(epochs, [v * args.loss_gamma for v in loss_valid_vol[i]], 'b.')
            ax1.plot(epochs, [v * args.loss_delta for v in loss_valid_dcm[i]], 'b+')
            ax2 = ax1.twinx()
            ax2.plot(epochs, loss_reg_list, 'g--')
            ax1.set_ylabel('Loss Training (r), Validation (b) | Seg-- Aux: Vol. SD+')
            ax2.set_ylabel('Regularisation (g--)')
            fig.savefig('{}-{}_plots_bo{}-{}.png'.format(args.output_basename, args.subject_valid, i, sequence_length), bbox_inches='tight', dpi=200)
            del ax1
            del ax2
            del fig


# #### 

# In[ ]:




