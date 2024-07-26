## PIDSNeRF: Pose Interpolation Depth Supervision Neural Radiance Fields for View Synthesis from Challenging Input
![Caption for the picture.](/figure/f1.png)

## Hardware

* OS: Ubuntu 20.04
* NVIDIA GPU with Compute Compatibility >= 75 and memory > 6GB (Tested with RTX 3060), CUDA 11.3 (might work with older version)
* 32GB RAM (in order to load full size images)

## Installing
* Python>=3.8 (installation via [anaconda](https://www.anaconda.com/distribution/) is recommended, use `conda create -n ngp_pl python=3.8` to create a conda environment and activate it by `conda activate ngp_pl`)
* Python libraries
    * Install pytorch by `pip install torch==1.11.0 --extra-index-url https://download.pytorch.org/whl/cu113`
    * Install `torch-scatter` following their [instruction](https://github.com/rusty1s/pytorch_scatter#installation)
    * Install `tinycudann` following their [instruction](https://github.com/NVlabs/tiny-cuda-nn#pytorch-extension) (pytorch extension)
    * Install `apex` following their [instruction](https://github.com/NVIDIA/apex#linux)
    * Install core requirements by `pip install -r requirements.txt`

* Cuda extension: Upgrade `pip` to >= 22.1 and run `pip install models/csrc/` (please run this each time you `pull` the code)

## Datasets

  Colmap data

For custom data, run `colmap` and get a folder `sparse/0` under which there are `cameras.bin`, `images.bin` and `points3D.bin`. The following data with colmap format are also supported:

  *  [nerf_llff_data](https://drive.google.com/file/d/16VnMcF1KJYxN9QId6TClMsZRahHNMW5g/view?usp=sharing) 
  *  [mipnerf360 data](http://storage.googleapis.com/gresearch/refraw360/360_v2.zip)
  *  [HDR-NeRF data](https://drive.google.com/drive/folders/1OTDLLH8ydKX1DcaNpbQ46LlP0dKx6E-I). Additionally, download my colmap pose estimation from [here](https://drive.google.com/file/d/1TXxgf_ZxNB4o67FVD_r0aBUIZVRgZYMX/view?usp=sharing) and extract to the same location.


## Training

Quickstart: `python train.py --root_dir <path/to/lego> --exp_name Lego`

## Testing

run `python show_gui.py` followed by the **same** hyperparameters used in training (`dataset_name`, `root_dir`, etc) and **add the checkpoint path** with `--ckpt_path <path/to/.ckpt>`

# Comparison with Instan-NGP and DSNeRF from sparse inputs

![Caption for the picture.](/figure/f2.png)

# Comparison the depth map with DSNeRF
![Caption for the picture.](/figure/f3.png)

## Thanks

Our code follows [kwea123](https://github.com/kwea123/ngp_pl) repositories. We appreciate him for making his codes available to the public.

