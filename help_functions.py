
import numpy as np
import torch
from datasets.ray_utils import  get_rays


def get_train_pose_w2c(train_poses_c2w):
    train_pose_list = []
    for i in range(len(train_poses_c2w)):
        train_tem = torch.cat((train_poses_c2w[i], torch.tensor([[0, 0, 0, 1]])), dim=0)
        train_pose_list.append(torch.inverse(train_tem)[None, ...])
    train_poses_w2c = torch.cat(train_pose_list, dim=0)
    return train_poses_w2c




def compute(H,W,K,pts,w2c):

    one_fill = torch.ones((pts.shape[0],1,1))
    pts = torch.cat((pts,one_fill),dim=-1).permute(0,2,1)
    c_pts = (w2c@pts).squeeze(-1)[:,0:3]
    depths = c_pts[:, 2:3].float()
    image_index_x, image_index_y = (K[0][0] * c_pts[:, 0:1] / c_pts[:, 2:3]) + W / 2,\
                                   (K[0][0] * c_pts[:, 1:2] / c_pts[:, 2:3]) + H / 2
    pixel_index = torch.cat((image_index_x,image_index_y),dim=1)

    valid_mask = (pixel_index[:,0] < W) *  (pixel_index[:,1] < H) * (pixel_index[:,0] > 0)* (pixel_index[:,1] > 0)
    depths = depths[valid_mask]
    pixel_index = pixel_index[valid_mask]
    index_depth = torch.cat((pixel_index,depths),dim=-1)


    return index_depth, valid_mask
def get_depth_update( pts3d,poses_interpolation,K):
    W = K[0,2]*2
    H = K[1,2]*2
    K_np = np.array(K)
    poses_interpolation_c2w = torch.tensor(poses_interpolation).float()
    poses_interpolation_w2c = get_train_pose_w2c(poses_interpolation_c2w)
    depth_rays = []
    depth_value = []

    for i in range(len(poses_interpolation_w2c)):
        index_depth, valid_mask = compute(H,W,K,pts3d[i][:,None,:],poses_interpolation_w2c[i])

        index_depth_t = index_depth[:,2]
        index_depth, valid_mask = np.array(index_depth),np.array(valid_mask)

        directions = np.concatenate(
            ([(index_depth[:, 0] - K_np [0, 2]) / K_np [0, 0]], [(index_depth[:, 1] - K_np [1, 2]) / K_np [1, 1]],
             [np.ones_like(index_depth[:, 0])]), axis=0).T
        rays_o, rays_d = get_rays(torch.tensor(directions).float(), poses_interpolation_c2w[i])

        rays_od = torch.cat((rays_o[:, None, ...], rays_d[:, None, ...]), dim=1)
        depth_rays.append(rays_od)
        depth_value.append(index_depth_t)

    depth_rays = torch.cat(depth_rays)
    depth_value = torch.cat(depth_value)

    return depth_rays,depth_value




# inverse weight
def poses_interpolation(train_poses,pts_3d,trains_pts,num=0):
    poses_i_list = []
    t_poses = train_poses[..., -1]
    train_poses_zmean = np.mean(train_poses[:,2,3])
    pts_center = np.mean(pts_3d[:,0:3],axis=0,keepdims=False)
    row_max = np.max(np.abs(pts_center[0:1] - t_poses[:,0:1]))
    clomn_max = np.max(np.abs(pts_center[1:2] - t_poses[:, 1:2]))
    k = np.min(pts_3d[:,2:3])
    p_i_t = pts_center[None,0:2] + np.random.uniform(-row_max*5, clomn_max*5,(num,2))
    p_i_t = np.concatenate((p_i_t,np.random.uniform(k, 5*train_poses_zmean,(num,1))),axis=1)
    pts_r = []
    for i in range(p_i_t.shape[0]):
        z = pts_center - p_i_t[i]
        z = z / np.linalg.norm(z)
        pts = []
        distance = np.zeros((len(train_poses)))
        for j in range(len(train_poses)):
            normal1 = z
            normal2 = train_poses[j, 0:3, 2]
            data_M = np.sqrt(np.sum(normal1 * normal1))
            data_N = np.sqrt(np.sum(normal2 * normal2))
            cos_theta = np.sum(normal1 * normal2) / (data_M * data_N)
            theta = np.arccos(cos_theta)
            distance[j] = theta
        w = 1/distance
        w = w/w.sum()
        for j in range(len(train_poses)):
            index_len = int(len(trains_pts[j]) * w[j])
            index =  np.random.choice(len(trains_pts[j]), index_len)
            pts.append(torch.tensor(trains_pts[j][index]).float())
        pts_r.append(torch.cat(pts))

        x = np.cross([0,1,0],z)
        x /= np.linalg.norm(x)

        y = np.cross(z,x)
        y /= np.linalg.norm(y)


        rotation_matrix = np.array([x,y,z]).T
        pose_i = np.concatenate((rotation_matrix,p_i_t[i,:,None]),axis=-1)
        poses_i_list.append(pose_i[None,...])
    r = np.concatenate(poses_i_list)
    return r,pts_r

# get depth rays
def get_depth_rays(data_list,poses,K,scale):
    depth_rays = []
    depth_value = []
    depth_weight = []
    pts = []
    K = np.array(K)
    for i in range(len(data_list)):
        data_list[i]['depth'] /= scale
        directions = np.concatenate(([(data_list[i]['coord'][:,0] - K[0, 2]) / K[0, 0]], [(data_list[i]['coord'][:,1] - K[1, 2]) / K[1, 1]],
                                  [np.ones_like(data_list[i]['coord'][:,0])]),axis=0).T
        rays_o, rays_d = get_rays(torch.tensor(directions).float(), torch.tensor(poses[i]).float())
        pts_tem = rays_o + torch.tensor(data_list[i]['depth'][...,None]).float() * rays_d
        pts.append(pts_tem)
        rays_od = torch.cat((rays_o[:,None,...],rays_d[:,None,...]),dim=1)
        depth_rays.append(rays_od)
        depth_value.append(torch.tensor(data_list[i]['depth']).float())

        depth_weight.append(torch.tensor(data_list[i]['error']).float())

    return depth_rays, depth_value, depth_weight
def get_data_list(poses,images,points,factor):

    Errs = np.array([point3D.error for point3D in points.values()])
    Err_mean = np.sum(Errs)

    data_list = []
    for id_im in range(1, len(images) + 1):
        depth_list = []
        coord_list = []
        weight_list = []
        pts = []
        for i in range(len(images[id_im].xys)):
            point2D = images[id_im].xys[i]
            id_3D = images[id_im].point3D_ids[i]
            if id_3D == -1:
                continue
            point3D = points[id_3D].xyz
            pts.append(point3D)
            depth = (poses[id_im-1,:3,2].T @ (point3D - poses[id_im-1,:3,3]))
            err = points[id_3D].error
            weight = 2 * np.exp(-(err / Err_mean) ** 2)

            depth_list.append(depth)
            coord_list.append(point2D*factor)
            weight_list.append(weight)
        if len(depth_list) > 0:
            data_list.append(
                {"depth": np.array(depth_list), "coord": np.array(coord_list), "error": np.array(weight_list),'pts':np.array(pts)})
        else:
            print(id_im, len(depth_list))
    return data_list

def get_ray_directions_train(W, H, pix, K,test=False):
    fx, fy, cx, cy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]

    v = (pix // W).float()
    u = (pix - (v * W)).float()
    if test:
        rand_uv=0.5
    else:
        rand_uv = torch.rand_like(u) * 0.6 + 0.2

    directions = \
        torch.stack([(u - cx + rand_uv) / fx, (v - cy + rand_uv) / fy, torch.ones_like(u)], -1)

    return directions









