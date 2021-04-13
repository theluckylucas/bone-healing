"""
Based on:
https://github.com/jacobkimmel/pytorch_convgru
"""

import torch
import torch.nn as nn
from torch.nn import functional as F


class ConvGRU(nn.Module):
    def __init__(self,
                 input_size,
                 hidden_size,
                 kernel_size,
                 dilation,
                 padding,
                 max_pool):
        super().__init__()
        
        self.input_size = input_size
        if max_pool:
            if isinstance(max_pool, int):
                self.max_pool = (1, max_pool, max_pool)
            else:
                self.max_pool = self.max_pool
        else:
            self.max_pool = False

        # GRU convolution with incorporation of hidden state
        p = [k//2 if k > 1 else 0 for k in kernel_size]  # TODO
        self.hidden_size = hidden_size
        self.reset_gate = nn.Conv3d(input_size + hidden_size, hidden_size, kernel_size, dilation=dilation, padding=p)
        self.update_gate = nn.Conv3d(input_size + hidden_size, hidden_size, kernel_size, dilation=dilation, padding=p)
        self.out_gate = nn.Conv3d(input_size + hidden_size, hidden_size, kernel_size, dilation=dilation, padding=p)

        # Appropriate initialization
        nn.init.orthogonal_(self.reset_gate.weight)
        nn.init.orthogonal_(self.update_gate.weight)
        nn.init.orthogonal_(self.out_gate.weight)

        torch.nn.init.normal(self.reset_gate.bias, 0, 0.0001)
        torch.nn.init.normal(self.update_gate.bias, 0, 0.0001)
        torch.nn.init.normal(self.out_gate.bias, 0, 0.0001)

    def forward(self, x, h):
        # Get batch and spatial sizes
        if self.max_pool:
            x = F.max_pool3d(x, self.max_pool)
        
        batch_size = x.data.size()[0]
        spatial_size = x.data.size()[2:]

        # Generate empty prev_state, if None is provided
        if h is None:
            state_size = [batch_size, self.hidden_size] + list(spatial_size)
            h = torch.zeros(state_size, device=x.device)
        
        # Data size: [batch, channel, depth, height, width]
        stacked_inputs = torch.cat((x, h), dim=1)
        update = torch.sigmoid(self.update_gate(stacked_inputs))
        reset = torch.sigmoid(self.reset_gate(stacked_inputs))
        output = torch.tanh(self.out_gate(torch.cat((x, h * reset), dim=1)))
        h_new = h * (1 - update) + output * update
        
        return h_new


class ConvGRUCell(nn.Module):
    def __init__(self,
                 input_size,
                 hidden_size,
                 kernel_size,
                 dilation=1,
                 padding=0,
                 n_gru=1,
                 max_pool=0
                ):
        super().__init__()
        assert n_gru > 0, "A cell must contain at least one GRU"

        # non-lin deformation Cell
        gru = ConvGRU(input_size, hidden_size, kernel_size, dilation, padding, max_pool=False)
        self.add_module("convGru0", gru)
        self.gru_list = [gru]
        for i in range(1, n_gru):
            gru = ConvGRU(hidden_size, hidden_size, kernel_size, dilation, padding, max_pool=max_pool)
            self.add_module("convGru" + str(i), gru)
            self.gru_list.append(gru)
        
        
    def forward(self, x, hs):
        if not hs:
            hs = [None] * len(self.gru_list)
        for i in range(len(hs)):
            x = self.gru_list[i](x, hs[i])
            hs[i] = x
        return hs


class ConvGRUCellSequenceImplant(nn.Module):
    def grid_identity_def(self, batch_size, output_size, depth, length):
        result = torch.tensor([1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0], dtype=torch.float)
        result = result.view(-1, 3, 4).expand(batch_size, 3, 4)
        return nn.functional.affine_grid(result, (batch_size, output_size, depth, length, length)).permute(0, 4, 1, 2, 3)
    
    def __init__(self,
                 seq_len,
                 depth,
                 length,
                 n_channels_input,
                 n_channels_hidden,
                 n_channels_output,
                 n_channels_clinical=3,
                 cell_len=1,
                 kernel_size=3,
                 dilation=1,
                 bspline_kernel=13,
                 smooth_offsets=True,
                ):
        super().__init__()
        assert seq_len > 0
        self.len = seq_len
        self.n_channels_output = n_channels_output
        self.depth = depth
        self.length = length
        
        self.smooth_offsets = smooth_offsets
        self.bspline_kernel = bspline_kernel        
        
        padding = 0
        base = n_channels_hidden//4
        self.trunk_cnn = nn.Sequential(
            nn.Conv3d(n_channels_input, base*4, kernel_size, padding=padding),
            nn.LeakyReLU(),
            nn.Conv3d(base*4, base*6, kernel_size, padding=padding),
            nn.InstanceNorm3d(base*6),
            nn.LeakyReLU(),
            nn.MaxPool3d((1, 2, 2)),
            nn.Conv3d(base*6, base*8, kernel_size, padding=padding),
            nn.InstanceNorm3d(base*8),
            nn.LeakyReLU(),
            nn.Conv3d(base*8, base*10, kernel_size, padding=padding),
            nn.InstanceNorm3d(base*10),
            nn.LeakyReLU(),
            nn.MaxPool3d((1, 2, 2)),
            nn.Conv3d(base*10, base*7, kernel_size, padding=padding),
            nn.InstanceNorm3d(base*7),
            nn.LeakyReLU(),
            nn.Conv3d(base*7, n_channels_hidden, kernel_size, padding=padding),
            nn.InstanceNorm3d(n_channels_hidden),
            nn.LeakyReLU()
        )
        
        self.trunk_gru = None
        if cell_len > 0:
            self.trunk_gru = ConvGRUCell(
                 n_channels_hidden+n_channels_input,
                 n_channels_hidden,
                 kernel_size,
                 dilation=1,
                 padding=1,
                 n_gru=cell_len,
                 max_pool=0
            )
        
        self.head_cnn = nn.Sequential(
            nn.Conv3d(n_channels_hidden, n_channels_output, 1),
            nn.Tanh()
        )

        torch.nn.init.normal(self.head_cnn[-2].weight, 0, 0.0001)
        torch.nn.init.normal(self.head_cnn[-2].bias, 0, 0.0001)
        
        self.head_aux = None
        
    def _bspline_controlpoint_smoothing(self, grid):
        k = (1, self.bspline_kernel, self.bspline_kernel)
        p = (0, self.bspline_kernel//2, self.bspline_kernel//2)
        for _ in range(3):
            grid = F.avg_pool3d(grid, kernel_size=k, stride=(1, 1, 1), padding=p)
        return grid
        
    def forward(self, sdms, imgs, segs, mods1, mods2, mods3, mods4, clinical_data, blackout=None):
        input_size = sdms[0, 0].unsqueeze(0).unsqueeze(0).size()
        
        ys = [sdms[:, 0].unsqueeze(1)]
        ls = [segs[:, 0].unsqueeze(1)]
        aux = [torch.ones(sdms.size(0)).view(-1, 1).to(sdms.device) * -1] # aux not used
        grd_size = list(ys[0].size())
        grd_size[1] = 3
        gs = [torch.zeros(grd_size).to(ys[0].device)]  # grids (for debug/visualisation purposes)

        hs = []  # recurrent hidden states
        
        if blackout is not None:  # blackout future input during training
            bo_imgs = []
            bo_segs = []
            bo_sdms = []
            bo_mods1 = []
            bo_mods2 = []
            bo_mods3 = []
            bo_mods4 = []
            for b in range(segs.size(0)):
                assert 0 < blackout[b] < self.len
                ones = torch.ones(imgs[b, blackout[b]:].size()).to(imgs.device)
                bo_imgs.append(torch.cat((imgs[b, :blackout[b]], ones * 0), dim=0).unsqueeze(0))
                bo_segs.append(torch.cat((segs[b, :blackout[b]], ones * 0), dim=0).unsqueeze(0))
                if sdms is not None:
                    bo_sdms.append(torch.cat((sdms[b, :blackout[b]], ones), dim=0).unsqueeze(0))
                if mods1 is not None:
                    bo_mods1.append(torch.cat((mods1[b, :blackout[b]], ones * 0), dim=0).unsqueeze(0))
                if mods2 is not None:
                    bo_mods2.append(torch.cat((mods2[b, :blackout[b]], ones * 0), dim=0).unsqueeze(0))
                if mods3 is not None:
                    bo_mods3.append(torch.cat((mods3[b, :blackout[b]], ones * 0), dim=0).unsqueeze(0))
                if mods4 is not None:
                    bo_mods4.append(torch.cat((mods4[b, :blackout[b]], ones * 0), dim=0).unsqueeze(0))
            imgs = torch.cat(bo_imgs, dim=0)
            segs = torch.cat(bo_segs, dim=0)
            if bo_sdms:
                sdms = torch.cat(bo_sdms, dim=0)
            if bo_mods1:
                mods1 = torch.cat(bo_mods1, dim=0)
            if bo_mods2:
                mods2 = torch.cat(bo_mods2, dim=0)
            if bo_mods3:
                mods3 = torch.cat(bo_mods3, dim=0)
            if bo_mods4:
                mods4 = torch.cat(bo_mods4, dim=0)
                                    
        for i in range(1, self.len):
            img_ = []
            seg_ = []
            for b in range(sdms.size(0)):
                if i < blackout[b]:
                    img_.append(sdms[b, i].unsqueeze(0).unsqueeze(0))
                    seg_.append(segs[b, i].unsqueeze(0).unsqueeze(0))
                else:
                    img_.append(ys[-1][b].unsqueeze(0))
                    seg_.append(ls[-1][b].unsqueeze(0))
            img = torch.cat(img_, dim=0)
            seg = torch.cat(seg_, dim=0)
            grd = gs[-1]
            
            # Compose input for current time step i
            inputs = [img, grd, sdms, imgs]
            if mods1 is not None:
                inputs.append(mods1)
            if mods2 is not None:
                inputs.append(mods2)
            if mods3 is not None:
                inputs.append(mods3)
            if mods4 is not None:
                inputs.append(mods4)

            x_spatial = torch.cat(inputs, dim=1)

            # Predict spatial offsets
            trunk = self.trunk_cnn(x_spatial)

            gru_inputs = torch.cat((trunk, F.interpolate(x_spatial, size=trunk.size()[2:])), dim=1)

            if self.trunk_gru is not None:
                hs = self.trunk_gru(gru_inputs, hs)
                trunk = hs[-1]

            offset = self.head_cnn(trunk)
            offset = F.interpolate(offset, size=input_size[2:], mode='trilinear')
            if self.smooth_offsets:
                offset = self._bspline_controlpoint_smoothing(offset)

            gs.append(offset)
            
            # Predict auxiliary head
            if not self.head_aux is None:
                head = self.head_aux(trunk)
                aux.append(head.view(-1, 1))
            else:
                aux.append(torch.Tensor([-1] * sdms.size(0)).view(-1, 1).to(sdms.device))
            
            # Compute resampling grid from offsets
            grid_identity = self.grid_identity_def(sdms.size(0), self.n_channels_output, self.depth, self.length).to(offset.device)
            assert grid_identity.size() == offset.size()
            grid = grid_identity + offset
                
            if not self.smooth_offsets:
                grid = self._bspline_controlpoint_smoothing(grid)
                
            # Resample last time step's prediction
            ys.append(F.grid_sample(ys[-1], grid.permute(0, 2, 3, 4, 1), padding_mode='border')) # grid.clamp_(-1, +1) if no padding)
            ls.append(F.grid_sample(ls[-1], grid.permute(0, 2, 3, 4, 1), padding_mode='border'))
            
        # Starting from penumbra backwards, so reverse to get predictions in forward time direction order
        
        #ys.reverse()
        ys = torch.cat(ys, dim=1)

        ls = torch.cat(ls, dim=1)
        
        #gs.reverse()
        
        #aux.reverse()
        
        aux = torch.cat(aux, dim=1)
               
        return ys, gs, aux, ls, sdms
