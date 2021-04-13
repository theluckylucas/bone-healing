import torch
import torch.nn as nn
import data
import numpy
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.colors import ListedColormap
from sklearn.metrics import f1_score


def get_title(prefix, idx, batch, seq_len, row):
    w_core = float(batch[data.KEY_W_CORE][row, idx])
    w_fuct = float(batch[data.KEY_W_FUCT][row, idx])
    if w_core > 0 or w_fuct > 0:
        return '{:0.2f}/{:0.2f}'.format(w_core, w_fuct)
    return prefix + str(idx)


def iterate(is_train,
            accumulate,
            dataset,
            batchsize,
            zsize,
            zslice,
            sequence_length,
            rnn,
            device,
            criterion,
            softdice,
            optimizer,
            loss_list,
            loss_seg_list,
            loss_aux_list,
            loss_reg_list,
            loss_vol_list,
            loss_dcm_list,
            axarr1,
            axarr2,
            axarr3,
            axarr4,
            axarr5,
            n_visual_samples,
            row_offset,
            validation_metric,
            dice_seg_list,
            dice_int_list,
            f1_seg_list,
            f1_int_list,
            loss_alpha,
            loss_beta,
            loss_gamma,
            loss_delta,
            dreg,
            loss_weight,
            vmax_imgs,
            deform_segs,
	    blackout_arg):
    loss_mean = 0
    loss_aux_mean = 0
    loss_seg_mean = 0
    loss_vol_mean = 0
    loss_reg_mean = 0
    loss_dcm_mean = 0
    dice_seg_mean = 0
    dice_int_mean = 0
    f1_seg_mean = 0
    f1_int_mean = 0
    inc = 0
    rnn.train(is_train)
    with torch.set_grad_enabled(is_train):        
        for bidx, batch in enumerate(dataset):
            sdms = batch[data.KEY_SDMAPS].to(device)
            imgs = batch[data.KEY_IMAGES].to(device)
            segs = batch[data.KEY_LABELS].to(device)
            
            if not isinstance(blackout_arg, int):
                assert len(blackout_arg) == 2 and blackout_arg[0]>1 and blackout_arg[1]<=sequence_length
                blackout = (numpy.ones(batchsize) * numpy.random.randint(blackout_arg[0], blackout_arg[1])).astype(numpy.int)
            else:
                blackout = (numpy.ones(batchsize) * blackout_arg).astype(numpy.int)

            if loss_weight is None:
                spatial_loss_weight = torch.FloatTensor([0 if i < blackout[0] else 1/(sequence_length-blackout[0]) for i in range(sequence_length)]).view(1, sequence_length, 1, 1, 1).to(device)
            else:
                spatial_loss_weight = torch.FloatTensor(loss_weight).view(1, len(loss_weight), 1, 1, 1).to(device)
            if inc == 0:
                print("        Loss weight of time steps:", list(spatial_loss_weight.cpu().numpy().flatten()))

            clinical_data = batch[data.KEY_GLOBAL].to(device)
            
            if batch[data.KEY_W_CORE]:  # stroke data

                w_core = batch[data.KEY_W_CORE].to(device)
                w_fuct = batch[data.KEY_W_FUCT].to(device)

                sdm_core = sdms[:, 0, :, :, :].unsqueeze(1)
                sdm_fuct = sdms[:, 1, :, :, :].unsqueeze(1)
                sdm_penu = sdms[:, 2, :, :, :].unsqueeze(1)

                img_core = imgs[:, 0, :, :, :].unsqueeze(1)
                img_penu = imgs[:, 1, :, :, :].unsqueeze(1)

                ys, gs, aux, mul = rnn(
                    sdm_core,
                    sdm_penu,
                    img_core,
                    img_penu,
                    clinical_data
                )

                # Spatial predictions
                out_core = (w_core * ys).sum(dim=1).unsqueeze(1)
                out_fuct = (w_fuct * ys).sum(dim=1).unsqueeze(1)
                out_penu = ys[:, -1, :, :, :].unsqueeze(1)

                pr_spatial = torch.cat((out_core, out_fuct, sdm_penu), dim=1)  # network core and F lesion prediction
                gt_spatial = torch.cat((sdm_core, sdm_fuct, sdm_penu), dim=1)  # ground truth of core and F lesion
                
            elif deform_segs:  # tumor data

                mod1 = batch[data.KEY_MODAL1].to(device)
                mod2 = batch[data.KEY_MODAL2].to(device)
                mod3 = batch[data.KEY_MODAL3].to(device)
                
                ys, gs, aux, ls = rnn(sdms, imgs, segs, mod1, mod2, mod3, None, clinical_data, blackout)
                
                pr_spatial = ls    # network pred
                gt_spatial = segs  # ground truth
            
                pr_volumes = (ls > 0.5).float().view(batchsize, sequence_length, -1).sum(2)
                gt_volumes = (segs > 0.5).float().view(batchsize, sequence_length, -1).sum(2)
                
            else:  # implant data

                ys, gs, aux, ls, sdms_check = rnn(sdms, imgs, segs, None, None, None, None, clinical_data, blackout)
                
                pr_spatial = ys    # network pred
                gt_spatial = sdms  # ground truth

                pr_volumes = ls.float().view(batchsize, sequence_length, -1).sum(2)
                gt_volumes = segs.float().view(batchsize, sequence_length, -1).sum(2)

            # Scalar predictions
            pr_scalars = aux
            gt_scalars = torch.linspace(0, sequence_length, steps=sequence_length) / sequence_length
            gt_scalars = gt_scalars.repeat(batchsize, 1).to(clinical_data.device)
            
            # Evaluation / Comparison with simple linear interpolation of SDMs
            interpolates = []
            for b in range(sdms.size(0)):
                t_observe = blackout[b]-1
                sdm_range = sdms[b, t_observe] - sdms[b, 0]
                interpolates_t = []
                for t in range(sequence_length):
                    interpolates_t.append(sdms[b, 0] + t * sdm_range / t_observe)
                interpolates.append(torch.stack(interpolates_t))
            interpolates = torch.stack(interpolates)

            # Dice eval
            dice_gt = (sdms < 0).detach().cpu().numpy()
            dice_lab = 0
            dice_seg = 0
            dice_int = 0
            f1_lab = 0
            f1_seg = 0
            f1_int = 0
            if not is_train:
                dice_inc = 0
                for b in range(dice_gt.shape[0]):
                    print('           DICE valid batch', b, end=' => ')
                    for t in range(blackout[b], sequence_length):
                        dseg = validation_metric((ys < 0).detach().cpu().numpy()[b, t].flatten(), dice_gt[b, t].flatten())
                        dlab = validation_metric((ls > 0.5).detach().cpu().numpy()[b, t].flatten(), dice_gt[b, t].flatten())
                        dint = validation_metric((interpolates < 0).detach().cpu().numpy()[b, t].flatten(), dice_gt[b, t].flatten())
                        if not is_train:
                            print('{}:[{:0.2f}|{:0.2f}|{:0.2f}]'.format(t, dseg, dlab, dint), end='  ')
                        dice_seg += dseg
                        dice_lab += dlab
                        dice_int += dint
                        dice_inc += 1
                    print()

                    print('             F1 valid batch', b, end=' => ')
                    for t in range(blackout[b], sequence_length):
                        dseg = f1_score(dice_gt[b, t].flatten(), (ys < 0).detach().cpu().numpy()[b, t].flatten())
                        dlab = f1_score(dice_gt[b, t].flatten(), (ls > 0.5).detach().cpu().numpy()[b, t].flatten())
                        dint = f1_score(dice_gt[b, t].flatten(), (interpolates < 0).detach().cpu().numpy()[b, t].flatten())
                        if not is_train:
                            print('{}:[{:0.2f}|{:0.2f}|{:0.2f}]'.format(t, dseg, dlab, dint), end='  ')
                        f1_seg += dseg
                        f1_lab += dlab
                        f1_int += dint
                    print()
                dice_lab /= dice_inc
                dice_seg /= dice_inc
                dice_int /= dice_inc
                f1_lab /= dice_inc
                f1_seg /= dice_inc
                f1_int /= dice_inc


            
            # Losses
            loss_seg = criterion(pr_spatial, gt_spatial, weight=spatial_loss_weight)
            loss_aux = criterion(pr_scalars, gt_scalars, weight=spatial_loss_weight)
            loss_vol = criterion(pr_volumes, gt_volumes, weight=spatial_loss_weight)
            loss_dcm = softdice(ls, segs)

            # Regularise displacements
            loss_reg = 0
            for g in gs:
                loss_reg += dreg(g)
            loss_reg /= len(gs)

            # Combined loss
            loss = 0
            if loss_seg > 0:
                loss += loss_seg
            if loss_alpha > 0:
                loss += loss_alpha * loss_aux
            if loss_beta > 0:
                loss += loss_beta * loss_reg
            if loss_gamma > 0:
                loss += loss_gamma * loss_vol
            if loss_delta > 0:
                loss += loss_delta * loss_dcm
            loss_seg_mean += loss_seg.item()
            loss_aux_mean += loss_aux.item()
            loss_reg_mean += loss_reg.item()
            loss_vol_mean += loss_vol.item()
            loss_dcm_mean += loss_dcm.item()
            dice_seg_mean += dice_seg
            dice_int_mean += dice_int
            f1_seg_mean += f1_seg
            f1_int_mean += f1_int
            loss_mean += loss.item()
            
            
            inc += 1
            
            if is_train:
                loss.backward()
                if inc % accumulate == 0:
                    optimizer.step()
                    optimizer.zero_grad()

        loss_list.append(loss_mean/inc)
        loss_seg_list.append(loss_seg_mean/inc)
        loss_aux_list.append(loss_aux_mean/inc)
        loss_reg_list.append(loss_reg_mean/inc)
        loss_vol_list.append(loss_vol_mean/inc)
        loss_dcm_list.append(loss_dcm_mean/inc)
        dice_seg_list.append(dice_seg_mean/inc)
        dice_int_list.append(dice_int_mean/inc)
        f1_seg_list.append(f1_seg_mean/inc)
        f1_int_list.append(f1_int_mean/inc)
        
        for ai, axarr in enumerate([axarr1, axarr2, axarr3, axarr4, axarr5]):
            for row in range(n_visual_samples):
                titles = []

                col = 0
                cols = range(sequence_length)

                if ai == 0:

                    for c in cols:
                        titles.append('IMG ' + str(c))
                        axarr[row + row_offset, col].imshow(imgs.detach().cpu().numpy()[row, c, zslice, :, :], cmap='jet', vmin=0, vmax=vmax_imgs)
                        col += 1

                    for c in cols:
                        titles.append('Grid ' + str(c))
                        img = gs[c].detach().cpu().permute(0, 4, 3, 2, 1).numpy()[row, :, :, zslice, :]
                        vmax = numpy.max(img)
                        vmin = numpy.min(img)
                        img -= vmin
                        img /= (vmax-vmin)
                        axarr[row + row_offset, col].imshow(numpy.swapaxes(img, 0, 1), vmin=0, vmax=1)
                        col += 1

                elif ai < 3:

                    for c in cols:
                        titles.append('SDM ' + str(c))
                        axarr[row + row_offset, col].imshow(sdms_check.detach().cpu().numpy()[row, c, zslice, :, :], vmin=-1, vmax=1, cmap='seismic')
                        col += 1

                    for c in cols:       
                        titles.append("Pr {:0.2f}".format(float(aux[row, c])))
                        if ai == 1:
                            axarr[row + row_offset, col].imshow(ys.detach().cpu().numpy()[row, c, zslice, :, :], vmin=-1, vmax=1, cmap='seismic')
                        elif ai == 2:
                            axarr[row + row_offset, col].imshow(interpolates.detach().cpu().numpy()[row, c, zslice, :, :], vmin=-1, vmax=1, cmap='seismic')
                        col += 1

                else:
                    for c in cols:
                        titles.append('LABEL ' + str(c))
                        axarr[row + row_offset, col].imshow(segs.detach().cpu().numpy()[row, c, zslice, :, :], vmin=0, vmax=1, cmap='gray')
                        col += 1

                    for c in cols:
                        titles.append("{:0.1f}/{:0.1f}".format(pr_volumes[row, c]/100, gt_volumes[row, c]/100))
                        if ai == 3:
                            axarr[row + row_offset, col].imshow(ls.detach().cpu().numpy()[row, c, zslice, :, :], vmin=0, vmax=1, cmap='gray')
                        elif ai == 4:
                            axarr[row + row_offset, col].imshow(interpolates.detach().cpu().numpy()[row, c, zslice, :, :] < 0, vmin=0, vmax=1, cmap='gray')
                        col += 1

                for ax, title in zip(axarr[row + row_offset], titles):
                    ax.set_title(title)
        
        del batch

    del gs
    del pr_scalars
    del gt_scalars
    del pr_spatial
    del gt_spatial
    del aux
    del loss

    result_ys = (ys < 0).detach().cpu().permute(4, 3, 2, 1, 0).float().numpy()
    result_ls = ls.detach().cpu().permute(4, 3, 2, 1, 0).float().numpy()
    result_ints = (interpolates < 0).detach().cpu().permute(4, 3, 2, 1, 0).float().numpy()
    result_sdms = (sdms < 0).detach().cpu().permute(4, 3, 2, 1, 0).float().numpy()
    result_imgs = imgs.detach().cpu().permute(4, 3, 2, 1, 0).float().numpy()
    
    return result_ys, result_ls, result_ints, result_sdms, result_imgs, blackout
