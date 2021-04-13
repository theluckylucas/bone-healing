import os
import csv
import glob
import nibabel as nib
import random
import datetime
import pprint

import torch
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.sampler import SubsetRandomSampler, SequentialSampler

import numpy as np
import scipy.ndimage as ndi
from scipy.ndimage.interpolation import map_coordinates, zoom
from scipy.ndimage.filters import gaussian_filter

from matplotlib import pyplot as plt


KEY_CASEID = 'case_id'
KEY_CLINICAL_IDX = 'clinical_idx'
KEY_IMAGES = 'images'
KEY_MODAL1 = 'modal1'
KEY_MODAL2 = 'modal2'
KEY_MODAL3 = 'modal3'
KEY_MODAL4 = 'modal4'
KEY_SDMAPS = 'sdmaps'
KEY_LABELS = 'labels'
KEY_GLOBAL = 'clinical'
KEY_W_CORE = 't_core_seq_weights'
KEY_W_FUCT = 't_fuct_seq_weights'

DIM_HORIZONTAL_NUMPY_3D = 0
DIM_DEPTH_NUMPY_3D = 2
DIM_CHANNEL_NUMPY_3D = 3
DIM_CHANNEL_TORCH3D_5 = 1


def get_distance_map(seg, invert=False, signed=True, threshold=0.5):
    seg_bin = seg > threshold
    
    if invert:
        sdm = ndi.distance_transform_edt(seg_bin)
        if signed:
            return sdm - ndi.distance_transform_edt(1 - seg_bin)
        return sdm
    
    sdm = ndi.distance_transform_edt(1 - seg_bin)
    if signed:
        return sdm - ndi.distance_transform_edt(seg_bin)
    return sdm


def sdm_interpolate_numpy(sdm0, sdm1, t):
    return sdm1 * t - sdm0 * (1 - t)


def sdm_interpolate(sdm0, sdm1, t0, t1, t):
    sdm_range = sdm1 - sdm0
    t_range = t1 - t0
    factor = (t - t0)/t_range
    return sdm0 + factor * sdm_range


def normalize_sdm(sdm_a, steps=50):
    border = np.zeros(shape=sdm_a.shape)
    if border.shape[0]>2:
        border[0, :, :] = 1
        border[-1, :, :] = 1
    if border.shape[1]>2:
        border[:, 0, :] = 1
        border[:, -1, :] = 1
    if border.shape[2]>2:
        border[:, :, 0] = 1
        border[:, :, -1] = 1
    sdm_b = get_distance_map(border, invert=False)
    out = np.ones(shape=sdm_a.shape)
    for j in range(0,steps+1,1):
        sdm_t = sdm_interpolate_numpy(sdm_b, sdm_a, time_func(j/steps, 'lin'))
        out[sdm_t < 0] = (steps-j)/steps
    out[sdm_a < 0] = sdm_a[sdm_a < 0] / abs(np.min(sdm_a))
    
    if np.min(out) > 0:
        out = 1
        
    return out


def time_func(t, func='lin'):
    assert 0 <= t <= 1

    if func == 'lin':
        return t

    if func == 'slow':
        return pow(t, 2)
    if func == 'fast':
        return pow(t, 0.5)

    if func == 'log':  # logistic: slow-fast-slow
        return 1 / (1 + 1000 * pow(0.000001, t))

    return t

    
class ImplantRatDataset3DRegistered(Dataset):
    PATH_FILE = 'D:/Christian/MetBioMat2_MRI/{}_MetBioMat3_20{}__{}_P1/{}_MetBioMat3_20{}_{}.nii.gz'
    TIMEPOINTS = ["d0"]
    SUBJECTS = ["100_0"]
    def __init__(self,
                 path_pattern=PATH_FILE,
                 modalities=[('E0', 'MR_SEQUENCE')],
                 labels=[('E0', 'MR_LABEL')],
                 timepoints=TIMEPOINTS,
                 subjects=SUBJECTS,
                 common_size=[128, 128, 128],
                 transform=None):
        self._path_pattern = path_pattern
        self._modalities = modalities
        self._labels = labels
        self._timepoints = timepoints
        self._transform = transform
        self._subjects = subjects
        self._common_size = common_size

    def _resample_array(self, image, target_size):
        if len(image.shape) == 4:
            image = image.squeeze(3)
        im_shape = image.shape
        assert len(im_shape) == len(target_size), ("Image: " + str(im_shape) + "vs. Target:" + str(target_size))
        if any([im_shape[i]!=target_size[i] for i in range(len(target_size))]):
            scale = [b/a for a,b in zip(im_shape, target_size)]
            return zoom(image, scale)
        return image
        
    def _load_image_data_from_nifti(self, subject, modality, timepoint):
        fpattern = self._path_pattern.format(subject, timepoint, modality[0], subject, timepoint, modality[1])
        fmatches = glob.glob(fpattern)
        assert len(fmatches) == 1, "File name pattern {} did not yield a single result: {}".format(fpattern, fmatches)
        img_data = nib.load(fmatches[0]).get_data()
        result = []
        if len(img_data.shape)>4:
            img_data = img_data.squeeze()
        for c in range(img_data.shape[3]):
            result.append(self._resample_array(img_data[:, :, :, c], self._common_size)[:, :, :, np.newaxis])
        return np.concatenate(result, axis=3), img_data.shape
    
    def __len__(self):
        return len(self._subjects)

    def __getitem__(self, item):
        
        subject = self._subjects[item]
        
        result = {
            KEY_CASEID: subject,
            "origsize": [],
            KEY_IMAGES: [],
            KEY_LABELS: [],
            KEY_GLOBAL: [],
            KEY_SDMAPS: [],
            KEY_W_CORE: [],
            KEY_W_FUCT: [],
            KEY_MODAL1: [],
            KEY_MODAL2: [],
            KEY_MODAL3: [],
            KEY_MODAL4: []
        }
        
        timepoint_pos = 0
        timepoint = self._timepoints[timepoint_pos]
        
        for i in range(len(self._timepoints)):
            result[KEY_GLOBAL].append(i)
        result[KEY_GLOBAL] = np.array(result[KEY_GLOBAL]).reshape((1, 1, 1, -1))
            
        assert len(self._labels) < 2, "can only handle single label channel per each time point"
        if self._labels:
            label = self._labels[0]
            label, orig_shape = self._load_image_data_from_nifti(subject, label, timepoint)
            result[KEY_LABELS] = label[:, :, :, :len(self._timepoints)]
            result["origsize"] = orig_shape 
        
        if self._modalities:
            for m, modality in enumerate(self._modalities):
                if m < 4:
                    mod, _ = self._load_image_data_from_nifti(subject, modality, timepoint)
                    result["modal" + str(m+1)] = mod[:, :, :, :len(self._timepoints)]
            result[KEY_IMAGES] = result[KEY_MODAL1].copy()
        
        if self._transform:
            result = self._transform(result)
        
        return result


def emptyCopyFromSample(sample):
    result = {KEY_CASEID: int(sample[KEY_CASEID]),
              "origsize": sample["origsize"],
              KEY_IMAGES:  [],
              KEY_LABELS:  [],
              KEY_SDMAPS:  [],
              KEY_GLOBAL:  [],
              KEY_W_CORE:  [],
              KEY_W_FUCT:  [],
              KEY_MODAL1:  [],
              KEY_MODAL2:  [],
              KEY_MODAL3:  [],
              KEY_MODAL4:  []
             }
    return result


class HemisphericFlip(object):
    """Flip numpy images along X-axis."""
    def __call__(self, sample):
        if random.random() > 0.5:
            result = emptyCopyFromSample(sample)
            if sample[KEY_IMAGES] != []:
                result[KEY_IMAGES] = np.flip(sample[KEY_IMAGES], DIM_HORIZONTAL_NUMPY_3D).copy()
            if sample[KEY_MODAL1] != []:
                result[KEY_MODAL1] = np.flip(sample[KEY_MODAL1], DIM_HORIZONTAL_NUMPY_3D).copy()
            if sample[KEY_MODAL2] != []:
                result[KEY_MODAL2] = np.flip(sample[KEY_MODAL2], DIM_HORIZONTAL_NUMPY_3D).copy()
            if sample[KEY_MODAL3] != []:
                result[KEY_MODAL3] = np.flip(sample[KEY_MODAL3], DIM_HORIZONTAL_NUMPY_3D).copy()
            if sample[KEY_MODAL4] != []:
                result[KEY_MODAL4] = np.flip(sample[KEY_MODAL4], DIM_HORIZONTAL_NUMPY_3D).copy()
            if sample[KEY_LABELS] != []:
                result[KEY_LABELS] = np.flip(sample[KEY_LABELS], DIM_HORIZONTAL_NUMPY_3D).copy()
            if sample[KEY_SDMAPS] != []:
                result[KEY_SDMAPS] = np.flip(sample[KEY_SDMAPS], DIM_HORIZONTAL_NUMPY_3D).copy()
            if sample[KEY_GLOBAL] != []:
                result[KEY_GLOBAL] = np.flip(sample[KEY_GLOBAL], DIM_HORIZONTAL_NUMPY_3D).copy()
            if sample[KEY_W_CORE] != []:
                result[KEY_W_CORE] = np.flip(sample[KEY_W_CORE], DIM_HORIZONTAL_NUMPY_3D).copy()
            if sample[KEY_W_FUCT] != []:
                result[KEY_W_FUCT] = np.flip(sample[KEY_W_FUCT], DIM_HORIZONTAL_NUMPY_3D).copy()
            return result
        return sample


class ToTensor(object):
    """Convert ndarrays in sample to Tensors."""
    def __init__(self, time_dim=None):
        self.time_dim = time_dim

    def __call__(self, sample):
        result = emptyCopyFromSample(sample)
        
        for key in [KEY_IMAGES, KEY_MODAL1, KEY_MODAL2, KEY_MODAL3, KEY_MODAL4, KEY_LABELS, KEY_SDMAPS]:
            if sample[key] != []:
                result[key] = torch.from_numpy(sample[key]).permute(3, 2, 1, 0).float()
                if self.time_dim is not None:
                    result[key] = result[key].unsqueeze(self.time_dim)
                    
        if sample[KEY_GLOBAL] != []:
            result[KEY_GLOBAL] = torch.from_numpy(np.array(sample[KEY_GLOBAL]))
            result[KEY_GLOBAL] = result[KEY_GLOBAL].permute(3, 2, 1, 0).float()
            if self.time_dim is not None:
                result[KEY_GLOBAL] = result[KEY_GLOBAL].unsqueeze(self.time_dim)
             
        if sample[KEY_W_CORE] != []:     
            result[KEY_W_CORE] = torch.from_numpy(np.array(sample[KEY_W_CORE]))              
            result[KEY_W_CORE] = result[KEY_W_CORE].permute(3, 2, 1, 0).float()              
            if self.time_dim is not None:
                result[KEY_W_CORE] = result[KEY_W_CORE].unsqueeze(self.time_dim)

        if sample[KEY_W_FUCT] != []:
            result[KEY_W_FUCT] = torch.from_numpy(np.array(sample[KEY_W_FUCT])).permute(3, 2, 1, 0).float()
            if self.time_dim is not None:
                result[KEY_W_FUCT] = result[KEY_W_FUCT].unsqueeze(self.time_dim)
            
        return result
    
    
class LabelsToDistanceMaps(object):
    def __init__(self, threshold=0.5, signed=True):
        self.thresh = threshold
        self.signed = signed
    
    def __call__(self, sample):        
        sample[KEY_SDMAPS] = sample[KEY_LABELS].copy().astype(np.float)
        for i in range(sample[KEY_SDMAPS].shape[3]):
            inputs = sample[KEY_SDMAPS][:, :, :, i]
            sdm = get_distance_map(inputs, invert=False, signed=self.signed, threshold=self.thresh)
            sample[KEY_SDMAPS][:, :, :, i] = normalize_sdm(sdm)
        return sample



class ElasticDeform3D(object):
    """Elastic deformation of images as described in [Simard2003]
       Simard, Steinkraus and Platt, "Best Practices for Convolutional
       Neural Networks applied to Visual Document Analysis", in Proc.
       of the International Conference on Document Analysis and
       Recognition, 2003.
    """

    def __init__(self, alpha=100, sigma=4, apply_to_images=False, random=1, seed=None):
        self._alpha = alpha
        self._sigma = sigma
        self._apply_to_images = apply_to_images
        self._random = random
        self._seed = None
        if seed is not None:
            self.seed = np.random.RandomState(seed)

    def elastic_transform(self, image, alpha=100, sigma=4, random_state=None):
        new_seed = datetime.datetime.now().second + datetime.datetime.now().microsecond
        if random_state is None:
            random_state = np.random.RandomState(new_seed)

        shape = image.shape
        dx = gaussian_filter((random_state.rand(*shape) * 2 - 1), sigma, mode="constant", cval=0) * alpha
        dy = gaussian_filter((random_state.rand(*shape) * 2 - 1), sigma, mode="constant", cval=0) * alpha
        dz = gaussian_filter((random_state.rand(*shape) * 2 - 1), sigma, mode="constant", cval=0) * alpha * 28/128  # TODO: correct according to voxel spacing

        x, y, z = np.meshgrid(np.arange(shape[0]), np.arange(shape[1]), np.arange(shape[2]))
        indices = np.reshape(y + dy, (-1, 1)), np.reshape(x + dx, (-1, 1)), np.reshape(z + dz, (-1, 1))

        return map_coordinates(image, indices, order=1).reshape(shape), random_state

    def __call__(self, sample):
        if random.random() < self._random and sample[KEY_LABELS] != []:
            sample[KEY_LABELS][:, :, :, 0], random_state = self.elastic_transform(sample[KEY_LABELS][:, :, :, 0], self._alpha, self._sigma, self._seed)
            for c in range(1, sample[KEY_LABELS].shape[3]):
                sample[KEY_LABELS][:, :, :, c], _ = self.elastic_transform(sample[KEY_LABELS][:, :, :, c], self._alpha, self._sigma, random_state=random_state)
                if self._apply_to_images:
                    if sample[KEY_IMAGES] != []:
                        sample[KEY_IMAGES][:, :, :, c], _ = self.elastic_transform(sample[KEY_IMAGES][:, :, :, c], self._alpha, self._sigma, random_state=random_state)
                if sample[KEY_MODAL1] != []:
                    sample[KEY_MODAL1][:, :, :, c], _ = self.elastic_transform(sample[KEY_MODAL1][:, :, :, c], self._alpha, self._sigma, random_state=random_state)
                if sample[KEY_MODAL2] != []:
                    sample[KEY_MODAL2][:, :, :, c], _ = self.elastic_transform(sample[KEY_MODAL2][:, :, :, c], self._alpha, self._sigma, random_state=random_state)
                if sample[KEY_MODAL3] != []:
                    sample[KEY_MODAL3][:, :, :, c], _ = self.elastic_transform(sample[KEY_MODAL3][:, :, :, c], self._alpha, self._sigma, random_state=random_state)
                if sample[KEY_MODAL4] != []:
                    sample[KEY_MODAL4][:, :, :, c], _ = self.elastic_transform(sample[KEY_MODAL4][:, :, :, c], self._alpha, self._sigma, random_state=random_state)
        return sample
    

class Resample(object):
    """Down- or upsample images."""
    def __init__(self, scale_factor=(1, 1, 1), mode='linear'):
        self._scale_factor = scale_factor
        if mode == 'linear':
            self._order = 1
        else:
            self._order = 0
            
    def _resample_channels(self, inputs):
        if inputs != []:
            assert len(inputs.shape) == len(self._scale_factor)+1
            results = []
            for c in range(inputs.shape[DIM_CHANNEL_NUMPY_3D]):
                results.append(ndi.zoom(inputs[:, :, :, c], self._scale_factor, order=self._order))
            return np.stack(results, axis=-1)
        return inputs

    def __call__(self, sample):
        result = emptyCopyFromSample(sample)
        result[KEY_GLOBAL] = sample[KEY_GLOBAL]
        result[KEY_W_CORE] = sample[KEY_W_CORE]
        result[KEY_W_FUCT] = sample[KEY_W_FUCT]
        
        result[KEY_IMAGES] = self._resample_channels(sample[KEY_IMAGES])
        result[KEY_MODAL1] = self._resample_channels(sample[KEY_MODAL1])
        result[KEY_MODAL2] = self._resample_channels(sample[KEY_MODAL2])
        result[KEY_MODAL3] = self._resample_channels(sample[KEY_MODAL3])
        result[KEY_MODAL4] = self._resample_channels(sample[KEY_MODAL4])
        result[KEY_LABELS] = self._resample_channels(sample[KEY_LABELS])
        result[KEY_SDMAPS] = self._resample_channels(sample[KEY_SDMAPS])
        
        return result
