# The-Long-Range-Depth-Estimation

Installation

Prerequisites
Linux or macOS (Windows is in experimental support)
Python 3.6+
PyTorch 1.3+
CUDA 9.2+ (If you build PyTorch from source, CUDA 9.0 is also compatible)
GCC 5+
MMCV

I ran experiments with PyTorch 1.8.0, CUDA 11.1, Python 3.7, and Ubuntu 20.04. Other settings that satisfact the requirement would work.

Use Anaconda to create a conda environment:
conda create -n MDE python=3.7
conda activate MDE

Install Pytorch:
conda install pytorch==1.8.0 torchvision==0.9.0 torchaudio==0.8.0 cudatoolkit=11.1 -c pytorch -c conda-forge
Then, install MMCV and install our toolbox:

pip install mmcv-full==1.3.13 -f https://download.openmmlab.com/mmcv/dist/cu111/torch1.8.0/index.html

git clone https://github.com/zhyever/Monocular-Depth-Estimation-Toolbox.git
cd Monocular-Depth-Estimation-Toolbox
pip install -e .

If training, you should install the tensorboard:

pip install future tensorboard
dataset
