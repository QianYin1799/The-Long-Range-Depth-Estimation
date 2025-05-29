# The-Long-Range-Depth-Estimation

## 📖 Introduction

**Monocular depth estimation**, a cost-effective solution for reconstructing 3D structures from single 2D images, finds widespread applications in robotic navigation, UAV control, and intelligent manufacturing. 

Despite significant advancements in deep neural networks, accurately estimating depth for **distant objects** remains challenging due to insufficient global context modeling.

To address this, we introduce a novel monocular depth estimation framework featuring:
- **CNN-Transformer dual-branch encoder** for local-global feature extraction.
- A **Cross-Dimensional Feature Fusion (CSF) Module** for enhanced global depth understanding.
- In the decoder, a **Depthwise Separable Upsampling Block (DSUB)** and **Multi-Scale Self-Attention Module (MSA)** to refine upsampling and recover detailed spatial information.

**Experimental results** demonstrate superior long-range prediction accuracy on KITTI and SUN RGB-D datasets.

---

## 📊 Benchmark Performance

| Dataset      | Abs-Rel ↓ | RMSE ↓  | Notes                              |
|:-------------|:-----------|:--------|:-----------------------------------|
| **KITTI**     | **0.053**  | **2.128** | Excellent performance, especially in distant regions |
| **SUN RGB-D** | **0.140**  | **0.417** | Using NYU pre-trained weights, shows strong generalization |

---

## 📊 Method Overview

### 📌 Architecture Overview

> 📷 **[Insert high-level framework diagram here]**

*Illustration: Dual-branch Encoder (CNN + Transformer), CSF Fusion Module, DSUB & MSA-based Decoder*

---

## 📦 Installation

### 📋 Prerequisites

- **Linux** or **macOS** (Windows experimental support)
- Python **3.6+**
- PyTorch **1.3+**
- CUDA **9.2+** (or CUDA 9.0 if building from source)
- GCC **5+**
- MMCV

### 🛠️ Installation Steps

Tested with:
- PyTorch 1.8.0
- CUDA 11.1
- Python 3.7
- Ubuntu 20.04

If your environment is similar, follow these steps:

```bash
conda create -n MDE python=3.7
conda activate MDE

conda install pytorch==1.8.0 torchvision==0.9.0 torchaudio==0.8.0 cudatoolkit=11.1 -c pytorch -c conda-forge

pip install mmcv-full==1.3.13 -f https://download.openmmlab.com/mmcv/dist/cu111/torch1.8.0/index.html

git clone https://github.com/zhyever/Monocular-Depth-Estimation-Toolbox.git
cd Monocular-Depth-Estimation-Toolbox
pip install -e .

pip install future tensorboard
📂 Dataset Preparation
It is recommended to symlink your datasets to $MONOCULAR-DEPTH-ESTIMATION-TOOLBOX/data

Example Folder Structure:

css
复制
编辑
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
│   │   ├── benchmark_test
│   │   ├── benchmark_cam
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
🚀 Usage
🏋️‍♂️ Training
Single / Multi-GPU Training:

bash
复制
编辑
bash tools/dist_train.sh ${CONFIG_FILE} ${GPU_NUM} [optional arguments]
Optional Arguments:

--no-validate

--work-dir ${WORK_DIR}

--resume-from ${CHECKPOINT_FILE}

--load-from ${CHECKPOINT_FILE}

--deterministic

Example:

bash
复制
编辑
bash ./tools/dist_train.sh configs/depthformer/depthformer_swint_w7_nyu.py 2 --work-dir work_dirs/saves/depthformer/depthformer_swint_w7_nyu
🧪 Inference & Testing
Single-GPU Testing:

bash
复制
编辑
python tools/test.py ${CONFIG_FILE} ${CHECKPOINT_FILE} [--out ${RESULT_FILE}] [--eval ${EVAL_METRICS}] [--show]
Example:

bash
复制
编辑
python tools/test.py configs/depthformer/depthformer_swinl_22k_w7_nyu.py \
checkpoints/depthformer_swinl_22k_w7_nyu.pth \
--show-dir depthformer_swinl_22k_w7_nyu_results
📊 Qualitative Results
📷 [Insert qualitative result images here]

📖 Citation
If you find this work useful in your research, please consider citing:

bibtex
复制
编辑
@misc{thelongrangedepth2025,
  title={The Long-Range Depth Estimation Framework},
  author={Your Name},
  year={2025},
  note={https://github.com/your-repo/The-Long-Range-Depth-Estimation}
}
📬 Contact
For questions and issues, feel free to open an Issue or reach out via email: your_email@example.com

📌 To-Do
 Code Refactoring

 Add Demonstration Videos

 Improve Documentation
