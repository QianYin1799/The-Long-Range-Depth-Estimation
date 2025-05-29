# The-Long-Range-Depth-Estimation

## 📖 Introduction

**Monocular depth estimation**, a cost-effective solution for reconstructing 3D structures from single 2D images, finds widespread applications in robotic navigation, UAV control, and intelligent manufacturing.

Despite significant advancements in deep neural networks, accurately estimating depth for **distant objects** remains challenging due to insufficient global context modeling.

To address this, we introduce a novel monocular depth estimation framework featuring:

* **CNN-Transformer dual-branch encoder** for local-global feature extraction.
* A **Cross-Dimensional Feature Fusion (CSF) Module** for enhanced global depth understanding.
* In the decoder, a **Depthwise Separable Upsampling Block (DSUB)** and **Multi-Scale Self-Attention Module (MSA)** to refine upsampling and recover detailed spatial information.

**Experimental results** demonstrate superior long-range prediction accuracy on KITTI and SUN RGB-D datasets.

---

## 📊 Benchmark Performance

| Dataset       | Abs-Rel 🔽 | RMSE 🔽   | Notes                                                      |
| :------------ | :--------- | :-------- | :--------------------------------------------------------- |
| **KITTI**     | **0.053**  | **2.128** |                                                            |
| **NYU-Depth** | **0.094**  | **0.329** |                                                            |
| **SUN RGB-D** | **0.140**  | **0.417** | Using NYU pre-trained weights, shows strong generalization |

---

## 📊 Method Overview

### 📌 Architecture Overview

📷 ![Framework Overview](docs/framework_overview.png)

Illustration: Dual-branch Encoder (CNN + Transformer), CSF Fusion Module, DSUB & MSA-based Decoder

---

## 📦 Installation

### 📋 Prerequisites

* **Linux** or **macOS** (Windows experimental support)
* Python **3.6+**
* PyTorch **1.3+**
* CUDA **9.2+** (or CUDA 9.0 if building from source)
* GCC **5+**
* MMCV

### 🛠️ Installation Steps

Tested with:

* PyTorch 1.8.0
* CUDA 11.1
* Python 3.7
* Ubuntu 20.04

If your environment is similar, follow these steps:

```bash
conda create -n MDE python=3.7
conda activate MDE

conda install pytorch==1.8.0 torchvision==0.9.0 torchaudio==0.8.0 cudatoolkit=11.1 -c pytorch -c conda-forge

pip install mmcv-full==1.3.13 -f https://download.openmmlab.com/mmcv/dist/cu111/torch1.8.0/index.html

git clone https://github.com/QianYin/The-Long-Range-Depth-Estimation.git
cd The-Long-Range-Depth-Estimation
pip install -e .

pip install future tensorboard
```

---

## 📂 Dataset Preparation
### 🔹 KITTI

From the official website, download the KITTI dataset. Then unzip the files into `data/kitti`. Finally, copy split files (whose names start with `kitti`) in the `splits` folder into `data/kitti`.

### 🔹 NYU Depth v2

Download from: [Google Drive Link](https://drive.google.com/file/d/1AysroWpfISmm-yRFGBgFTrLy6FjQwvwP/view?usp=sharing)

Then, download the standard test set from the official site:
[NYU Depth v2 Dataset](https://cs.nyu.edu/~fergus/datasets/nyu_depth_v2.html)

> ⚠️ The downloaded file will unzip into `train` and `test` folders.

### 🔹 SUN RGB-D

Download from: [SUNRGBD Official Download](https://rgbd.cs.princeton.edu/)

After downloading, place the dataset under `data/SUNRGBD/`.

### 🔹 Example Folder Structure:

```
monocular-depth-estimation-toolbox
├── depth
├── tools
├── configs
├── splits
├── data
│   ├── kitti
│   │   ├── input
│   │   │   ├── 2011_09_26
│   │   │   ├── ...
│   │   ├── gt_depth
│   │   │   ├── 2011_09_26_drive_0001_sync
│   │   │   ├── ...
│   │   ├── split_file.txt
│   ├── nyu
│   │   ├── basement_0001a
│   │   ├── ...
│   │   ├── split_file.txt
│   ├── SUNRGBD
│   │   ├── SUNRGBD
│   │   │   ├── kv1
│   │   │   ├── kv2
│   │   │   ├── ...
│   │   ├── split_file.txt
```

## 🚀 Usage

### 🏋️ Training

**Single / Multi-GPU Training:**

```bash
bash tools/dist_train.sh ${CONFIG_FILE} ${GPU_NUM} [optional arguments]
```

**Optional Arguments:**

* `--no-validate`
* `--work-dir ${WORK_DIR}`
* `--resume-from ${CHECKPOINT_FILE}`
* `--load-from ${CHECKPOINT_FILE}`
* `--deterministic`

**Example:**

```bash
bash ./tools/dist_train.sh configs/long_rangedepth/swinl_w7_nyu.py 2 --work-dir work_dirs/saves/long_rangedepth/swinl_w7_nyu
```

### 🦪 Inference & Testing

**Single-GPU Testing:**

```bash
python tools/test.py ${CONFIG_FILE} ${CHECKPOINT_FILE} [--out ${RESULT_FILE}] [--eval ${EVAL_METRICS}] [--show]
```

**Example:**

```bash
python tools/test.py configs/long_rangedepth/swinl_22k_w7_nyu.py \
checkpoints/swinl_22k_w7_nyu.pth \
--show-dir swinl_22k_w7_nyu_results
```

---

## 📊 Qualitative Results

📷 ![Qualitative Results](docs/qualitative_results.png)

---

## 📖 Citation

If you find this work useful in your research, please consider citing:

```bibtex
@misc{thelongrangedepth2025,
  title={The Long-Range Depth Estimation Framework},
  author={Your Name},
  year={2025},
  note={https://github.com/your-repo/The-Long-Range-Depth-Estimation}
}
```

---

## 📩 Contact

For questions and issues, feel free to open an [Issue](https://github.com/your-repo/The-Long-Range-Depth-Estimation/issues) or reach out via email: `your_email@example.com`

---

## 📌 To-Do

* [x] Code Refactoring
* [ ] Add Demonstration Videos
* [ ] Improve Documentation
