
import os

from tqdm import tqdm
import help_functions
from .ray_utils import *
from .color_utils import read_image
from .colmap_utils import \
    read_cameras_binary, read_images_binary, read_points3d_binary

from .base import BaseDataset



train_image = [0,4, 8 ,12,16]
test_image = [x for x in range(19) if x not in train_image]

class ColmapDataset(BaseDataset):
    def __init__(self, root_dir, split='train',cameras_num =21000, downsample=1.0, **kwargs):
        super().__init__(root_dir, split, downsample)

        self.read_intrinsics()
        self.cameras_num = cameras_num
        if kwargs.get('read_meta', True):
            self.read_meta(split, **kwargs)

    def read_intrinsics(self):
        # Step 1: read and scale intrinsics (same for all images)
        camdata = read_cameras_binary(os.path.join(self.root_dir, 'sparse/0/cameras.bin'))
        h = int(camdata[1].height*self.downsample)
        w = int(camdata[1].width*self.downsample)
        self.img_wh = (w, h)

        if camdata[1].model == 'SIMPLE_RADIAL':
            fx = fy = camdata[1].params[0]*self.downsample
            cx = camdata[1].params[1]*self.downsample
            cy = camdata[1].params[2]*self.downsample
        elif camdata[1].model in ['PINHOLE', 'OPENCV']:
            fx = camdata[1].params[0]*self.downsample
            fy = camdata[1].params[1]*self.downsample
            cx = camdata[1].params[2]*self.downsample
            cy = camdata[1].params[3]*self.downsample
        else:
            raise ValueError(f"Please parse the intrinsics for camera model {camdata[1].model}!")
        self.K = torch.FloatTensor([[fx, 0, cx],
                                    [0, fy, cy],
                                    [0,  0,  1]])

        self.directions = get_ray_directions(h, w, self.K)

    def read_meta(self, split, **kwargs):
        # Step 2: correct poses
        # read extrinsics (of successfully reconstructed images)
        imdata = read_images_binary(os.path.join(self.root_dir, 'sparse/0/images.bin'))
        img_names = [imdata[k].name for k in imdata]
        perm = np.argsort(img_names)
        if '360_v2' in self.root_dir and self.downsample<1: # mipnerf360 data
            folder = f'images_{int(1/self.downsample)}'
        else:
            folder = 'images'
        # read successfully reconstructed images and ignore others
        img_paths = [os.path.join(self.root_dir, folder, name)
                     for name in sorted(img_names)]
        w2c_mats = []
        bottom = np.array([[0, 0, 0, 1.]])
        for k in imdata:
            im = imdata[k]
            R = im.qvec2rotmat(); t = im.tvec.reshape(3, 1)
            w2c_mats += [np.concatenate([np.concatenate([R, t], 1), bottom], 0)]
        w2c_mats = np.stack(w2c_mats, 0)
        poses = np.linalg.inv(w2c_mats)[perm, :3] # (N_images, 3, 4) cam2world matrices


        pts3d = read_points3d_binary(os.path.join(self.root_dir, 'sparse/0/points3D.bin'))

        #data_list [depth,cood,err]
        point_loss_data = help_functions.get_data_list(poses,imdata,pts3d,self.downsample)
        pts3d_point = np.concatenate(([point_loss_data[i]['pts'] for i in train_image]))
        self.poses, self.pts3d = center_poses(poses,point_loss_data, pts3d_point)
        scale = np.linalg.norm(self.poses[..., 3], axis=-1).min()


        self.poses[...,3] /= scale

        self.pts3d[:, 0:3] /= scale


        for i in range(len(point_loss_data)):
            point_loss_data[i]['pts'] /= scale
        self.point_loss_data = [x['pts'] for i, x in enumerate(point_loss_data) if i in train_image]

        self.train_poses = np.array([x for i, x in enumerate(self.poses) if i in train_image])
        if split == 'train':
            self.poses_interpolation, self.pts_interpolation = help_functions.poses_interpolation(self.train_poses, self.pts3d, self.point_loss_data, num=self.cameras_num)

        # point_depth_loss
        self.depth_rays, self.depth_value, self.depth_error = help_functions.get_depth_rays(point_loss_data, self.poses, self.K, scale)

        self.rays = []

        if split == 'test_traj': # use precomputed test poses
            self.poses = create_spheric_poses(1.2, self.poses[:, 1, 3].mean())
            self.poses = torch.FloatTensor(self.poses)
            return


        # use every 8th image as test set
        if split=='train':
            img_paths = [x for i, x in enumerate(img_paths) if i in train_image]
            self.poses = np.array([x for i, x in enumerate(self.poses) if i in train_image])
            self.depth_rays = torch.cat([x for i, x in enumerate(self.depth_rays) if i in train_image])
            self.depth_value = torch.cat([x for i, x in enumerate(self.depth_value) if i in train_image])

            # update
            self.depth_rays_update, self.depth_value_update = help_functions.get_depth_update(self.pts_interpolation, self.poses_interpolation, self.K)
            self.depth_rays_list = []
            self.depth_value_list = []
            for i in range(6):
                index = torch.randperm(self.depth_rays_update.shape[0])[0:len(self.depth_rays_update)//6]
                depth_rays = torch.cat((self.depth_rays,self.depth_rays_update[index]))
                depth_value = torch.cat((self.depth_value,self.depth_value_update[index]))
                self.depth_rays_list.append(depth_rays)
                self.depth_value_list.append(depth_value)
            del self.depth_rays_update
            del self.depth_value_update
        elif split == 'test':

            img_paths = [x for i, x in enumerate(img_paths) if i in test_image]
            self.poses = np.array([x for i, x in enumerate(self.poses) if i in test_image])
        print(f'Loading {len(img_paths)} {split} images ...')
        for img_path in tqdm(img_paths):
            buf = [] # buffer for ray attributes: rgb, etc
            img = read_image(img_path, self.img_wh, blend_a = False)
            img = torch.FloatTensor(img)
            buf += [img]
            self.rays += [torch.cat(buf, 1)]
        self.rays = torch.stack(self.rays) # (N_images, hw, ?)

        self.poses = torch.FloatTensor(self.poses) # (N_images, 3, 4)

        self.h_w_size = torch.ones((3)).float()
        self.h_w_size[0] = self.img_wh[1]
        self.h_w_size[1] = self.img_wh[0]
        self.h_w_size[2] = 3






