import torch
from torch.utils.data import Dataset
import numpy as np

import help_functions


class BaseDataset(Dataset):
    """
    Define length and sampling method
    """
    def __init__(self, root_dir,  split='train', downsample=1.0):
        self.root_dir = root_dir
        self.split = split
        self.downsample = downsample


    def read_intrinsics(self):
        raise NotImplementedError

    def __len__(self):
        if self.split.startswith('train'):
            return 1000
        return len(self.poses)

    def __getitem__(self, idx):
        if self.split.startswith('train'):



            # training pose is retrieved in train.py

            img_idxs = np.random.choice(len(self.poses), self.batch_size)
            # randomly select pixels
            pix_idxs = np.random.choice(self.img_wh[0]*self.img_wh[1], self.batch_size)
            rays = self.rays[img_idxs, pix_idxs]
            poses_w2c = help_functions.get_train_pose_w2c(self.poses)


            # randomly select depth
            i = idx % 6
            index = torch.randint(0,self.depth_rays_list[i].shape[0],(img_idxs.shape[0],))

            rays_o_depth = self.depth_rays_list[i][index, 0, :]
            rays_d_depth = self.depth_rays_list[i][index, 1, :]
            depth_s = self.depth_value_list[i][index]
            sample = {'img_idxs':img_idxs, 'rgb': rays[:, :3],'pix_idxs': pix_idxs,
                      'rays_o_depth': rays_o_depth, 'rays_d_depth':rays_d_depth, 'depth_value': depth_s, 'rgb_len':len(img_idxs),
                      'K':self.K, 'h_w_size':self.h_w_size, 'poses_w2c':poses_w2c,
                     }

        else:
            sample = {'pose': self.poses[idx], 'img_idxs': idx}
            if len(self.rays)>0: # if ground truth available
                rays = self.rays[idx]
                sample['rgb'] = rays[:, :3]


        return sample