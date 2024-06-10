PIDSNeRF: Pose Interpolation Depth Supervision Neural Radiance Fields for View Synthesis from Challenging Input

## Install
* Python>=3.8 (installation via [anaconda](https://www.anaconda.com/distribution/) is recommended, use `conda create -n ngp_pl python=3.8` to create a conda environment and activate it by `conda activate ngp_pl`)
* Python libraries
    * Install pytorch by `pip install torch==1.11.0 --extra-index-url https://download.pytorch.org/whl/cu113`
    * Install `torch-scatter` following their [instruction](https://github.com/rusty1s/pytorch_scatter#installation)
    * Install `tinycudann` following their [instruction](https://github.com/NVlabs/tiny-cuda-nn#pytorch-extension) (pytorch extension)
    * Install `apex` following their [instruction](https://github.com/NVIDIA/apex#linux)
    * Install core requirements by `pip install -r requirements.txt`

* Cuda extension: Upgrade `pip` to >= 22.1 and run `pip install models/csrc/` (please run this each time you `pull` the code)

## Supported Datasets

  Colmap data

For custom data, run `colmap` and get a folder `sparse/0` under which there are `cameras.bin`, `images.bin` and `points3D.bin`. The following data with colmap format are also supported:

  *  [nerf_llff_data](https://drive.google.com/file/d/16VnMcF1KJYxN9QId6TClMsZRahHNMW5g/view?usp=sharing) 
  *  [mipnerf360 data](http://storage.googleapis.com/gresearch/refraw360/360_v2.zip)
  *  [HDR-NeRF data](https://drive.google.com/drive/folders/1OTDLLH8ydKX1DcaNpbQ46LlP0dKx6E-I). Additionally, download my colmap pose estimation from [here](https://drive.google.com/file/d/1TXxgf_ZxNB4o67FVD_r0aBUIZVRgZYMX/view?usp=sharing) and extract to the same location.


## Training

Quickstart: `python train.py --root_dir <path/to/lego> --exp_name Lego`



