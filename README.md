# 2D Gaussian Splatting for Geometrically Accurate Radiance Fields

### Read first
__This is a forked Windows Installation Tutorial and the main codes will not be updated__
This forked GitHub project is intented for folks who have little to know command-line knowledge and want to install, train, and view 2D Gaussian Splats and derivative meshes. If you have used Nerftudio, 3DGS, or other similar command-line based radiance field projects, most likely you have already installed some or all of the depedencies required for this project.

Now let's get 3D Gaussian Splatting!

The section below is from the original GitHub page. Jump down to [Overview](#overview) to get started. <br>
<br>
<br>


# About 2D Gaussian Splatting for Geometrically Accurate Radiance Fields
[Project page](https://surfsplatting.github.io/) | [Paper](https://arxiv.org/pdf/2403.17888) | [Video](https://www.youtube.com/watch?v=oaHCtB6yiKU) | [Surfel Rasterizer (CUDA)](https://github.com/hbb1/diff-surfel-rasterization) | [Surfel Rasterizer (Python)](https://colab.research.google.com/drive/1qoclD7HJ3-o0O1R8cvV3PxLhoDCMsH8W?usp=sharing) | [DTU+COLMAP (3.5GB)](https://drive.google.com/drive/folders/1SJFgt8qhQomHX55Q4xSvYE2C6-8tFll9) | [SIBR Viewer Pre-built for Windows](https://drive.google.com/file/d/1DRFrtFUfz27QvQKOWbYXbRS2o2eSgaUT/view?usp=sharing)<br>

![Teaser image](assets/teaser.jpg)

This repo contains the official implementation for the paper "2D Gaussian Splatting for Geometrically Accurate Radiance Fields". Our work represents a scene with a set of 2D oriented disks (surface elements) and rasterizes the surfels with [perspective correct differentiable raseterization](https://colab.research.google.com/drive/1qoclD7HJ3-o0O1R8cvV3PxLhoDCMsH8W?usp=sharing). Our work also develops regularizations that enhance the reconstruction quality. We also devise meshing approaches for Gaussian splatting.

<br>
<br>



## Overview

2D Gaussian Splatting for Geometrically Accurate Radiance Fields is a project that is built upon the foundational work originally authored by Inria, Université Côte d'Azur, and MPI Informatik. The 2DGS workflow follows the same process as any other 3D gaussian splatting project with the additional layer of 3D Mesh extraction.


The following instructions will walk you through installation and all 4 main components of the 2DGS workflow. If you get stuck or have questions, please check the original project's disucssion page first prior to asking questions here. Most likely someone else has already ran into your issue and has documented how to resolve it!

<br>
<br>

## Installation

Despite my best efforts to compile the original environment provided by the researchers - I ended up just using the environment I set up for the original 3DGS project. The plus side, you will be able to run both projects at the end of this tutorial!

If you have already set up the original gaussian splatting environment, skip down to the diff-surfel-rasterization submodule installation instructions.

### Requirements - Hardware

An NVIDIA 2XXX GPU or better. Preferably an RTX 3XXX or 4XXXX GPU with 12GB VRAM or more. Less VRAM means you will need to use less images, train less iterations, or downsample input images. This will be covered further down in the instructions.

### Requirements - Software

This is the sofware dependencies you will need installed prior to installing the project. Many of these dependencies are shared with other NeRF projects.
- __Git__ - You will need this to pull the code from GitHub. You can download it [here ](https://git-scm.com/downloads). Follow default installation instructions. You can test to see if you have it already installed by typing ```git --version``` into command prompt
- __Conda__ - I recommend using [Anaconda](https://www.anaconda.com/download) because it's easy to install and manage environments in the future. [MiniConda](https://docs.conda.io/en/latest/miniconda.html) is a great lightweight alternative.
- __CUDA Toolkit__ - this was tested with 11.8. Ensure you are not running 11.6 or 12+. You can download CUDA Toolkit [here](https://developer.nvidia.com/cuda-toolkit-archive) You can check which version of CUDA Toolkit you have installed by typing ```nvcc --version``` into command prompt.
- __Visual Studio 2019 or newer__ - I found that the latest version of VS2022 will throw errors at you. I suggest downloading the latest Professional version of VS2019 [here](https://learn.microsoft.com/en-us/visualstudio/releases/2019/history#release-dates-and-build-numbers) and then restarting your machine. You don't need to register it. __note__: make sure you add __Desktop Development with C++__ when installing <br>
- __COLMAP__ - Use the Windows binary, it's easy! You can download it [here](https://github.com/colmap/colmap/releases)
- __ImageMagik__ - This is for preparing your images. Download it [here](https://imagemagick.org/script/download.php)
- __FFMPEG__ - Use this to extract images from video. Download it [here](https://ffmpeg.org/download.html)

### Cloning th 3DGS Repository

You will need to clone Inria's 3DGS repository from GitHub to set up the conda environment that you will run 2DGS (I know, confusing but trust me). Follow these steps to clone the repository: <br>

1. Open Windows Command Prompt by tying "cmd" into your search bar.
2. Copy the below code into command prompt and press enter

```git clone https://github.com/graphdeco-inria/gaussian-splatting --recursive```

The folder will download to the root of our command line prompt with the name "Gaussian-Splatting". Typically in your ```C:User/<username>``` folder. For example, on my PC the folder is now located at C:User/Jonat/Guassian-Splatting

### Setting up the 3DGS environment

Next, you will need to create a Conda environment from the 3DGS code. Open command prompt and enter these lines below one at a time. The second line will compile the code which can take 10 minutes or longer. The last line will "activate" the conda environment. You will need to enter ```conda activate gaussian_splatting``` any time you want to run 2DGS (or 3DGS if you decide to use this project too!).

```shell
cd gaussian-splatting
SET DISTUTILS_USE_SDK=1
conda env create --file environment.yml
conda activate gaussian_splatting
```
Please note that this process assumes that you have CUDA SDK **11** installed.

### Installing 2DGS

After setting up the 3DGS environment, I suggest closing your command prompt window and start a fresh one. This will avoid cloning 2DGS code into the right wont.

```git clone https://github.com/hbb1/2d-gaussian-splatting.git --recursive```

Next, change directories into the newly cloned 2DGS folder:

```cd 2d-gaussian-splatting```

 Activate the Gaussian Splatting conda environment that you set up earlier

 ```conda activate gaussian_splatting```

 Now, install the 2DGS surfel rasterizer:

```pip install submodules/diff-surfel-rasterization```

**Congrats! You have successfully installed 2DGS!!!**

<br>
<br>

## Preparing Images From Your Own Scenes

Our COLMAP loaders expect the following dataset structure in the source path location:

```
<location>
|---images
|   |---<image 0>
|   |---<image 1>
|   |---...
|---sparse
    |---0
        |---cameras.bin
        |---images.bin
        |---points3D.bin
```

For rasterization, the camera models must be either a SIMPLE_PINHOLE or PINHOLE camera. We provide a converter script ```convert.py```, to extract undistorted images and SfM information from input images. Optionally, you can use ImageMagick to resize the undistorted images. This rescaling is similar to MipNeRF360, i.e., it creates images with 1/2, 1/4 and 1/8 the original resolution in corresponding folders. To use them, please first install a recent version of COLMAP (ideally CUDA-powered) and ImageMagick. Put the images you want to use in a directory ```<location>/input```.
```
<location>
|---input
    |---<image 0>
    |---<image 1>
    |---...
```
 If you have COLMAP and ImageMagick on your system path, you can simply run 
```shell
python convert.py -s <location> [--resize] #If not resizing, ImageMagick is not needed
```
Alternatively, you can use the optional parameters ```--colmap_executable``` and ```--magick_executable``` to point to the respective paths. Please note that on Windows, the executable should point to the COLMAP ```.bat``` file that takes care of setting the execution environment. Once done, ```<location>``` will contain the expected COLMAP data set structure with undistorted, resized input images, in addition to your original images and some temporary (distorted) data in the directory ```distorted```.

If you have your own COLMAP dataset without undistortion (e.g., using ```OPENCV``` camera), you can try to just run the last part of the script: Put the images in ```input``` and the COLMAP info in a subdirectory ```distorted```:
```
<location>
|---input
|   |---<image 0>
|   |---<image 1>
|   |---...
|---distorted
    |---database.db
    |---sparse
        |---0
            |---...
```
Then run 
```shell
python convert.py -s <location> --skip_matching [--resize] #If not resizing, ImageMagick is not needed
```

<details>
<summary><span style="font-weight: bold;">Command Line Arguments for convert.py</span></summary>

  #### --no_gpu
  Flag to avoid using GPU in COLMAP.
  #### --skip_matching
  Flag to indicate that COLMAP info is available for images.
  #### --source_path / -s
  Location of the inputs.
  #### --camera 
  Which camera model to use for the early matching steps, ```OPENCV``` by default.
  #### --resize
  Flag for creating resized versions of input images.
  #### --colmap_executable
  Path to the COLMAP executable (```.bat``` on Windows).
  #### --magick_executable
  Path to the ImageMagick executable.
</details>
<br>
<br>

## Training
To train a scene, simply use
```bash
python train.py -s <path to COLMAP or NeRF Synthetic dataset>
```
Commandline arguments for regularizations
```bash
--lambda_normal  # hyperparameter for normal consistency
--lambda_distortion # hyperparameter for depth distortion
--depth_ratio # 0 for mean depth and 1 for median depth, 0 works for most cases
```
**Tips for adjusting the parameters on your own dataset:**
- For unbounded/large scenes, we suggest using mean depth, i.e., ``depth_ratio=0``,  for less "disk-aliasing" artifacts.

<br>
<br>

## Extracting Meshes
### Bounded Mesh Extraction
To export a mesh within a bounded volume, simply use
```bash
python render.py -m <path to pre-trained model> -s <path to COLMAP dataset> 
```
Commandline arguments you should adjust accordingly for meshing for bounded TSDF fusion, use
```bash
--depth_ratio # 0 for mean depth and 1 for median depth
--voxel_size # voxel size
--depth_trunc # depth truncation
```
If these arguments are not specified, the script will automatically estimate them using the camera information.
### Unbounded Mesh Extraction
To export a mesh with an arbitrary size, we devised an unbounded TSDF fusion with space contraction and adaptive truncation.
```bash
python render.py -m <path to pre-trained model> -s <path to COLMAP dataset> --mesh_res 1024
```

### Quick Examples
Assuming you have downloaded [MipNeRF360](https://jonbarron.info/mipnerf360/), simply use
```bash
python train.py -s <path to m360>/<garden> -m output/m360/garden
# use our unbounded mesh extraction!!
python render.py -s <path to m360>/<garden> -m output/m360/garden --unbounded --skip_test --skip_train --mesh_res 1024
# or use the bounded mesh extraction if you focus on foreground
python render.py -s <path to m360>/<garden> -m output/m360/garden --skip_test --skip_train --mesh_res 1024
```
If you have downloaded the [DTU dataset](https://drive.google.com/drive/folders/1SJFgt8qhQomHX55Q4xSvYE2C6-8tFll9), you can use
```bash
python train.py -s <path to dtu>/<scan105> -m output/date/scan105 -r 2 --depth_ratio 1
python render.py -r 2 --depth_ratio 1 --skip_test --skip_train
```
**Custom Dataset**: We use the same COLMAP loader as 3DGS, you can prepare your data following [here](https://github.com/graphdeco-inria/gaussian-splatting?tab=readme-ov-file#processing-your-own-scenes). 


#### Novel View Synthesis
For novel view synthesis on [MipNeRF360](https://jonbarron.info/mipnerf360/) (which also works for other colmap datasets), use
```bash
python scripts/mipnerf_eval.py -m60 <path to the MipNeRF360 dataset>
```
We provide <a> Evaluation Results (Pretrained, Images)</a>. 
<details>
<summary><span style="font-weight: bold;">Table Results</span></summary>

</details>


## Viewing the Results
https://github.com/RongLiu-Leo/2d-gaussian-splatting/assets/102014841/b75dd9a7-e3ee-4666-99ff-8c9121ff66dc

The Pre-built Viewer for Windows can be found [here](https://drive.google.com/file/d/1DRFrtFUfz27QvQKOWbYXbRS2o2eSgaUT/view?usp=sharing).
### How to use
Firstly open the viewer, 
```shell
<path to downloaded/compiled viewer>/bin/SIBR_remoteGaussian_app_rwdi
```
and then
```shell
# Monitor the training process
python train.py -s <path to COLMAP or NeRF Synthetic dataset> 
# View the trained model
python view.py -s <path to COLMAP or NeRF Synthetic dataset> -m <path to trained model> 
```


## FAQ
- **Training does not converge.**  If your camera's principal point does not lie at the image center, you may experience convergence issues. Our code only supports the ideal pinhole camera format, so you may need to make some modifications. Please follow the instructions provided [here](https://github.com/graphdeco-inria/gaussian-splatting/issues/144#issuecomment-1938504456) to make the necessary changes. We have also modified the rasterizer in the latest [commit](https://github.com/hbb1/diff-surfel-rasterization/pull/6) to support data accepted by 3DGS. To avoid further issues, please update to the latest commit.

- **No mesh / Broken mesh.** When using the *Bounded mesh extraction* mode, it is necessary to adjust the `depth_trunc` parameter to perform TSDF fusion to extract meshes. On the other hand, *Unbounded mesh extraction* does not require tuning the parameters but is less efficient.  

- **Can 3DGS's viewer be used to visualize 2DGS?** Technically, you can export 2DGS to 3DGS's ply file by appending an additional zero scale. However, due to the inaccurate affine projection of 3DGS's viewer, you may see some distorted artefacts. We are currently working on a viewer for 2DGS, so stay tuned for updates.

## Acknowledgements
This project is built upon [3DGS](https://github.com/graphdeco-inria/gaussian-splatting). The TSDF fusion for extracting mesh is based on [Open3D](https://github.com/isl-org/Open3D). The rendering script for MipNeRF360 is adopted from [Multinerf](https://github.com/google-research/multinerf/), while the evaluation scripts for DTU and Tanks and Temples dataset are taken from [DTUeval-python](https://github.com/jzhangbs/DTUeval-python) and [TanksAndTemples](https://github.com/isl-org/TanksAndTemples/tree/master/python_toolbox/evaluation), respectively. The fusing operation for accelerating the renderer is inspired by [Han's repodcue](https://github.com/Han230104/2D-Gaussian-Splatting-Reproduce). We thank all the authors for their great repos. 


## Citation
If you find our code or paper helps, please consider citing:
```bibtex
@inproceedings{Huang2DGS2024,
    title={2D Gaussian Splatting for Geometrically Accurate Radiance Fields},
    author={Huang, Binbin and Yu, Zehao and Chen, Anpei and Geiger, Andreas and Gao, Shenghua},
    publisher = {Association for Computing Machinery},
    booktitle = {SIGGRAPH 2024 Conference Papers},
    year      = {2024},
    doi       = {10.1145/3641519.3657428}
}
```
