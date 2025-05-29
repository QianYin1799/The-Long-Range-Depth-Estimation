# The-Long-Range-Depth-Estimation

## 项目简介

Monocular depth estimation, a cost-effective solution for reconstructing 3D structures from single 2D images, finds widespread applications in robotic navigation, UAV control, and intelligent manufacturing. Despite significant advancements in deep neural networks, accurately estimating depth for distant objects remains challenging due to insufficient global context modeling.

To address this, we introduce a novel depth estimation framework incorporating a **CNN-Transformer dual-branch encoder** to extract local and global features, respectively. A **Cross-Dimensional Feature Fusion Module (CSF)** is designed to enhance global depth understanding through feature interaction. In the decoder, the **Depthwise Separable Upsampling Block (DSUB)** and **Multi-Scale Self-Attention Module (MSA)** refine upsampling and recover detailed spatial information, significantly improving depth prediction accuracy.

**Experimental results:**

* **KITTI dataset:** Abs-Rel 0.053, RMSE 2.128, with notable improvements in distant regions.
* **SUN RGB-D dataset (with NYU pre-trained weights):** Abs-Rel 0.140, RMSE 0.417.

This framework significantly advances long-range depth prediction, holding promising application potential.

---

## 用到的库/框架

* PyTorch 1.8.0
* MMSegmentation v0.16.0
* MMCV 1.3.13
* OpenCV
* Numpy
* Matplotlib
* TensorBoard (for training visualization)

---

## 安装方法

### Prerequisites

* Linux or macOS (Windows experimental support)
* Python 3.6+
* PyTorch 1.3+
* CUDA 9.2+ (or CUDA 9.0 if building PyTorch from source)
* GCC 5+

### Installation

#### 1. 创建 Conda 环境

```bash
conda create -n MDE python=3.7
conda activate MDE
```

#### 2. 安装 PyTorch

```bash
conda install pytorch==1.8.0 torchvision==0.9.0 torchaudio==0.8.0 cudatoolkit=11.1 -c pytorch -c conda-forge
```

#### 3. 安装 MMCV 和工具箱

```bash
pip install mmcv-full==1.3.13 -f https://download.openmmlab.com/mmcv/dist/cu111/torch1.8.0/index.html

git clone https://github.com/zhyever/Monocular-Depth-Estimation-Toolbox.git
cd Monocular-Depth-Estimation-Toolbox
pip install -e .
```

#### 4. 安装 TensorBoard (用于训练过程可视化)

```bash
pip install future tensorboard
```

---

## 数据准备

建议将数据集根目录符号链接到 `$MONOCULAR-DEPTH-ESTIMATION-TOOLBOX/data`。

```bash
monocular-depth-estimation-toolbox
├── depth
├── tools
├── configs
├── splits
├── data
│   ├── kitti
│   │   ├── input
│   │   │   ├── 2011_09_26
│   │   │   ├── 2011_09_28
│   │   │   └── ...
│   │   ├── gt_depth
│   │   │   ├── 2011_09_26_drive_0001_sync
│   │   │   └── ...
│   │   ├── benchmark_test
│   │   │   ├── 0000000000.png
│   │   │   └── ...
│   │   ├── benchmark_cam
│   │   │   ├── 0000000000.txt
│   │   │   └── ...
│   │   └── split_file.txt
│   ├── nyu
│   │   ├── basement_0001a
│   │   └── ...
│   │   └── split_file.txt
│   ├── SUNRGBD
│   │   ├── SUNRGBD
│   │   │   ├── kv1
│   │   │   └── ...
│   │   └── split_file.txt
```

如果数据结构不同，需修改对应 config 中的路径。

---

## 使用方法

### 多卡训练

```bash
bash tools/dist_train.sh ${CONFIG_FILE} ${GPU_NUM} [optional arguments]
```

**常用参数**：

* `--no-validate`：关闭训练过程中的验证（不推荐）
* `--work-dir ${WORK_DIR}`：自定义保存目录
* `--resume-from ${CHECKPOINT_FILE}`：从已有权重继续训练
* `--load-from ${CHECKPOINT_FILE}`：加载权重做迁移学习
* `--deterministic`：开启确定性模式（减慢速度但保证复现）

**示例**：

```bash
bash tools/dist_train.sh configs/depthformer/depthformer_swint_w7_nyu.py 2 --work-dir work_dirs/saves/depthformer/depthformer_swint_w7_nyu
```

### 数据集测试

#### 单卡测试

```bash
python tools/test.py ${CONFIG_FILE} ${CHECKPOINT_FILE} [--out ${RESULT_FILE}] [--eval ${EVAL_METRICS}] [--show]
```

**示例**：

```bash
python tools/test.py configs/depthformer/depthformer_swinl_22k_w7_nyu.py \
    checkpoints/depthformer_swinl_22k_w7_nyu.pth \
    --show-dir depthformer_swinl_22k_w7_nyu_results
```

---

## 联系方式

如果有任何问题或建议，欢迎提 issue 或 PR 🙌
