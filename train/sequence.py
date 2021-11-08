## sequence.py

## the dicom reference coordinates system is used:
#   x direction: increases from the right side of the patient to the left
#   y direction: increases from the front of the patient to the back
#   z direction: increases from the feet to the head
# https://dicomiseasy.blogspot.com/2013/06/getting-oriented-using-image-plane.html

from model import make_surv_array

from pathlib import Path

from elasticdeform import deform_random_grid
import h5py
import numpy as np
import pandas as pd
from scipy.ndimage.interpolation import zoom
from sklearn.preprocessing import MinMaxScaler
import torch
from torch.utils.data import Dataset

class CustomDataset(Dataset):
    '''Custom Dataset that read the data and applies augmentations '''

    def __init__(self, args, with_label=True):
        self.args = args
        self.shape = args['shape']
        if args['split_color']:
            self.out_shape = [args['shape'][-1]//3, *args['shape'][:-1], 3] 
        else:
            self.out_shape = args['shape']

        label_df = pd.read_csv(args['label_file'])

        patch_file = Path(args['patch_file'])
        disk_df = pd.read_csv(patch_file.parent / (patch_file.stem + '.csv'))
        disk_ids = disk_df['identifier'].values

        label_df = label_df[label_df['identifier'].isin(disk_ids)]
        label_df = label_df.reset_index(drop=True)

        # normalize clinical features between 0 and 10 
        mm_scaler = MinMaxScaler()
        for col in args['cat_input']:
            label_df[col] = mm_scaler.fit_transform(label_df[col].values.reshape(-1, 1)) * 10

        breaks = self.get_breaks()

        self.cat_dict = dict(zip(
            label_df['identifier'],
            label_df[args['cat_input']].values
        ))

        if with_label:
            label = make_surv_array(
                label_df['time'],
                label_df['event'],
                breaks
            )

            self.label_dict = dict(zip(label_df['identifier'], label))
            self.raw_label_dict = dict(zip(
                label_df['identifier'],
                [[time, event] for time, event in zip(
                    label_df['time'],
                    label_df['event']
                )]  
            ))  

        self.identifiers = label_df['identifier'].values
        if self.args['training']:
            np.random.shuffle(self.identifiers)
        else:
            self.identifiers.sort()

        self.total_count = len(self.identifiers)
        if not self.total_count:
            raise ValueError('no cases were found: check label naming')
        print(f'len {self.total_count}')

    def get_breaks(self):
        breaks = self.args['breaks']
        if type(breaks) == list:
            return np.array(breaks) 
        if type(breaks) != dict:
            raise ValueError(f'breaks data type not known {type(breaks)}')
        if breaks['type'] == 'linear':
            init = breaks['min']
            fin = breaks['max']
            step = breaks['step']
            print(init, fin, step)
            return np.arange(init, fin, step)
        raise ValueError('breaks dict data type not known')
        
    def add_gaussian_noise(self, data, noise_var):
        variance = np.random.uniform(noise_var[0], noise_var[1])
        data = data + np.random.normal(0.0, variance, size=data.shape) 
    
        return data

    def add_offset(self, data, var):
        off = np.random.uniform(var[0], var[1])
        data = data + off

        return data

    def deform_image(self, data, sigma, points):
        data = deform_random_grid(
            data,
            sigma,
            points
        )
        return data

    def resample(self, data, xy_zoom, z_zoom):

        fac = [xy_zoom, xy_zoom, z_zoom]

        return zoom(data, zoom=fac, order=1)

    def get_aug_func(self):
        ''' returns a list with all the augmentations '''
        aug = []
        if self.args['augment']['flip']:
            # flip on sagittal plane
            if np.random.rand() < 0.5:
                aug.append(
                    lambda data: np.flip(data, axis=0)
                )
            # flip on coronal plane
            if np.random.rand() < 0.5:
                aug.append(
                    lambda data: np.flip(data, axis=1)
                )

        if self.args['augment']['rot']:
            # rotate k*90 degree
            rot_k = np.random.randint(0, 4)
            if rot_k:
                aug.append(
                    lambda data: np.rot90(
                        data, axes=(0, 1), k=rot_k
                    )
                )

        if self.args['augment']['zoom']:
            xy_zoom = np.random.uniform(
                self.args['augment']['zoom'][0],
                self.args['augment']['zoom'][1],
            )
            z_zoom = np.random.uniform(
                1.,
                self.args['augment']['zoom'][1],
            )
            aug.append(
                lambda data: self.resample(
                    data,
                    xy_zoom,
                    z_zoom,
                )
            )

        if self.args['augment']['deform']['sigma']:
            sigma = np.random.uniform(
                self.args['augment']['deform']['sigma'][0],
                self.args['augment']['deform']['sigma'][1],
            )
            aug.append(
                lambda data: self.deform_image(
                    data,
                    sigma,
                    self.args['augment']['deform']['points']
                )
            )

        if self.args['augment']['offset']:
            aug.append(
                lambda data: self.add_offset(
                    data,
                    self.args['augment']['offset']
                )
            ) 

        if self.args['augment']['noise']['type'] != 'None':
            if self.args['augment']['noise']['type'] == 'gaussian':
                aug.append(
                    lambda data: self.add_gaussian_noise(
                        data,
                        self.args['augment']['noise']['variance']
                    )
                )
            else:
                raise ValueError('noise not registered')

        return aug

    def augmentation(self, data, aug_func):
        for func in aug_func:
            data = func(data)
        return data

    def __len__(self):
        ''' cases involved '''
        return self.total_count

    def get_index(self, patch_shape):
        '''
        returns the indices in order
        to crop patches to smaller size
        '''
        init = np.zeros(len(patch_shape))
        fin = np.zeros(len(patch_shape))

        # use random indices for training
        if self.args['training']:
            for ii in range(0, len(patch_shape)):
                init[ii] = np.random.randint(
                    low=0,
                    high=patch_shape[ii] - self.shape[ii] + 1
                )

        # center for validation
        else:
            for ii in range(0, len(patch_shape)):
                init[ii] = (patch_shape[ii] - self.shape[ii]) // 2 

        fin = init + np.array(self.shape)

        init = init.astype(int)
        fin = fin.astype(int)

        out = np.empty(len(init) + len(fin))
        out[::2] = init
        out[1::2] = fin

        return out.astype(int)

        
    def crop_patch(self, data, idx):
        if len(idx) == 6:
            return data[idx[0]:idx[1], idx[2]:idx[3], idx[4]:idx[5]]
        elif len(idx) == 4:
            return data[idx[0]:idx[1], idx[2]:idx[3]]
        raise ValueError('patch_shape not implemented')

    def get_data(self, idx):
        data = {}

        if torch.is_tensor(idx):
            idx = idx.tolist()

        pid = self.identifiers[idx]

        with h5py.File(self.args['patch_file'], 'r') as ff:
            ct_patch = ff[pid]['ct'][:]
            pt_patch = ff[pid]['pt'][:]

        if self.args['training']:
            aug_func = self.get_aug_func()
            ct_patch = self.augmentation(ct_patch, aug_func)
            pt_patch = self.augmentation(pt_patch, aug_func)

        index = self.get_index(ct_patch.shape)
        ct_patch = self.crop_patch(ct_patch, index)
        pt_patch = self.crop_patch(pt_patch, index)

        cat_input = self.cat_dict[pid]

        if self.args['split_color']:
            ct_patch_ = np.empty((self.shape[-1]//3, *self.shape[:2], 3))
            pt_patch_ = np.empty((self.shape[-1]//3, *self.shape[:2], 3))
        for jj in range(0, ct_patch_.shape[0]):
            ct_patch_[jj] = ct_patch[:, :, jj*3:(jj+1)*3]
            pt_patch_[jj] = pt_patch[:, :, jj*3:(jj+1)*3]
        ct_patch = ct_patch_
        pt_patch = pt_patch_

        if self.args['channel_first']:
            ct_patch = np.moveaxis(ct_patch, -1, 0)
            pt_patch = np.moveaxis(pt_patch, -1, 0)

        data['ct'] = ct_patch.astype(np.float32)
        data['pt'] = pt_patch.astype(np.float32)
        data['cat'] = cat_input.astype(np.float32)

        return data

    def get_label(self, idx):
        data = {}
        
        if torch.is_tensor(idx):
            idx = idx.tolist()

        pid = self.identifiers[idx]
        data['label'] = np.array(
            self.label_dict[pid],
            dtype=np.float32
        )
        data['raw_label'] = np.array(
            self.raw_label_dict[pid],
            dtype=np.float32
        )

        return data

    def __getitem__(self, idx):
        inputs = self.get_data(idx)
        label = self.get_label(idx)

        return {**inputs, **label}
    

    def getitem(self, idx):
        ''' idx \epsilon [0, self.__len__()[ '''
        return_dict = {}

        if torch.is_tensor(idx):
            idx = idx.tolist()

        pid = self.identifiers[idx]

        with h5py.File(self.args['patch_file'], 'r') as ff:
            ct_patch = ff[pid]['ct'][:]
            pt_patch = ff[pid]['pt'][:]

        if self.args['training']:
            aug_func = self.get_aug_func()
            ct_patch = self.augmentation(ct_patch, aug_func)
            pt_patch = self.augmentation(pt_patch, aug_func)

        index = self.get_index(ct_patch.shape)
        ct_patch = self.crop_patch(ct_patch, index)
        pt_patch = self.crop_patch(pt_patch, index)

        label = self.label_dict[pid]
        cat_input = self.cat_dict[pid]
        raw_label = self.raw_label_dict[pid]

        if self.args['split_color']:
            ct_patch_ = np.empty((self.shape[-1]//3, *self.shape[:2], 3))
            pt_patch_ = np.empty((self.shape[-1]//3, *self.shape[:2], 3))
        for jj in range(0, ct_patch_.shape[0]):
            ct_patch_[jj] = ct_patch[:, :, jj*3:(jj+1)*3]
            pt_patch_[jj] = pt_patch[:, :, jj*3:(jj+1)*3]
        ct_patch = ct_patch_
        pt_patch = pt_patch_

        if self.args['channel_first']:
            ct_patch = np.moveaxis(ct_patch, -1, 0)
            pt_patch = np.moveaxis(pt_patch, -1, 0)

        ct_patch = ct_patch.astype(np.float32)
        pt_patch = pt_patch.astype(np.float32)
        label = np.array(label, dtype=np.float32)
        raw_label = np.array(raw_label, dtype=np.float32)
        cat_input = cat_input.astype(np.float32)

        return_dict['ct'] = ct_patch
        return_dict['pt'] = pt_patch
        return_dict['label'] = label
        return_dict['raw_label'] = raw_label
        return_dict['cat'] = cat_input

        return return_dict
