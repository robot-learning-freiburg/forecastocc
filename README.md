# ForecastOcc: Vision-based Semantic Occupancy Forecasting

[Paper](https://arxiv.org/abs/2602.08006) |
[Project Website](https://forecastocc.cs.uni-freiburg.de/)

This repository is the official implementation of:

> **ForecastOcc: Vision-based Semantic Occupancy Forecasting**
>
> [Riya Mohan*](https://www.linkedin.com/in/riya-m-90b6a2282/), [Juana Valeria Hurtado*](https://rl.uni-freiburg.de/people/hurtado), [Rohit Mohan*](https://rl.uni-freiburg.de/people/mohan), and [Abhinav Valada](https://rl.uni-freiburg.de/people/valada)
>
> IEEE International Conference on Robotics and Automation (ICRA), 2026

<p align="center">
  <a href="https://forecastocc.cs.uni-freiburg.de/static/images/figures/overview_web.png">
    <img src="https://forecastocc.cs.uni-freiburg.de/static/images/figures/overview_web.png" alt="Overview of ForecastOcc" width="800">
  </a>
</p>

## 📖 Citation

If you find this work useful, please cite:

```bibtex
@article{mohan2026forecastocc,
  title={ForecastOcc: Vision-based Semantic Occupancy Forecasting},
  author={Mohan, Riya and Hurtado, Juana Valeria and Mohan, Rohit and Valada, Abhinav},
  journal={arXiv preprint arXiv:2602.08006},
  year={2026}
}
```

## 📔 Abstract

Autonomous driving requires forecasting both geometry and semantics over time to effectively reason about future environment states. Existing vision-based occupancy forecasting methods focus on motion-related categories such as static and dynamic objects, while semantic information remains largely absent. Recent semantic occupancy forecasting approaches address this gap but rely on past occupancy predictions obtained from separate networks. This makes current methods sensitive to error accumulation and prevents learning spatio-temporal features directly from images. In this work, we present ForecastOcc, the first framework for vision-based semantic occupancy forecasting that jointly predicts future occupancy states and semantic categories. Our framework yields semantic occupancy forecasts for multiple horizons directly from past camera images, without relying on externally estimated maps. We evaluate ForecastOcc in two complementary settings: multi-view forecasting on the Occ3D-nuScenes dataset and monocular forecasting on SemanticKITTI, where we establish the first benchmark for this task. We introduce the first baselines by adapting two 2D forecasting modules within our framework. Importantly, we propose a novel architecture that incorporates a temporal cross-attention forecasting module, a 2D-to-3D view transformer, a 3D encoder for occupancy prediction, and a semantic occupancy head for voxel-level forecasts across multiple horizons. Extensive experiments on both datasets show that ForecastOcc consistently outperforms baselines, yielding semantically rich, future-aware predictions that capture scene dynamics and semantics critical for autonomous driving.

## 🏗️ Setup

### 💻 System requirements

The code has been tested with:

- Linux
- Python 3.8
- PyTorch 1.11.0
- CUDA 11.3
- MMCV 1.5.3
- MMDetection 2.25.1
- MMSegmentation 0.25.0
- MMClassification 0.25.0

Other versions may work but are not tested.

### ⚙️ Installation

```bash
conda create -n forecastocc python=3.8 -y
conda activate forecastocc

conda install pytorch==1.11.0 torchvision==0.12.0 cudatoolkit=11.3 \
  -c pytorch
pip install mmcv-full==1.5.3 \
  -f https://download.openmmlab.com/mmcv/dist/cu113/torch1.11.0/index.html
pip install mmdet==2.25.1 mmsegmentation==0.25.0 mmcls==0.25.0
pip install -r requirements/runtime.txt
FORCE_CUDA=1 pip install -v -e .
```

The final command builds the CUDA extension used for BEV pooling and requires
a CUDA toolchain compatible with the installed PyTorch build.

## 💾 Data preparation

Download nuScenes v1.0-trainval and the occupancy labels from
[Occ3D-nuScenes](https://github.com/Tsinghua-MARS-Lab/Occ3D). Arrange the data
as follows:

```text
data/nuscenes/
├── maps/
├── samples/
├── sweeps/
├── v1.0-trainval/
└── gts/
```

Generate the ForecastOcc metadata:

```bash
python tools/create_data_bevdet_forecasting.py \
  --root-path data/nuscenes
```

This creates `forecastocc-nuscenes_infos_train.pkl` and
`forecastocc-nuscenes_infos_val.pkl` in the dataset root.

## 🏃 Running the code

### Configuration

The canonical configuration is
[`configs/forecastocc/forecastocc.py`](configs/forecastocc/forecastocc.py).
It uses `data/nuscenes` by default. Set the following variables when the data
or annotations are stored elsewhere:

```bash
export NUSCENES_DATA_ROOT=/path/to/nuscenes
export NUSCENES_ANN_ROOT=/path/to/nuscenes
```

`NUSCENES_ANN_ROOT` must contain the generated metadata files and the `gts/`
directory.

### Training

ForecastOcc is initialized from a current-occupancy checkpoint that includes
the EfficientNet-B3 image encoder. Set its path before starting distributed
training:

```bash
export FORECASTOCC_INIT_CHECKPOINT=<PATH_TO_CURRENT_OCCUPANCY_CHECKPOINT>
bash tools/dist_train_occ.sh \
  configs/forecastocc/forecastocc.py \
  8
```

Training outputs are written to `work_dirs/forecastocc` by default. Use
`--work-dir` to select another output directory.

### Evaluation

```bash
bash tools/dist_test_occ.sh \
  configs/forecastocc/forecastocc.py \
  <PATH_TO_FORECASTOCC_CHECKPOINT> \
  8 --eval mIoU \
  --work-dir work_dirs/forecastocc/eval
```

## 🎯 Pretrained model

| Model | +1 s mIoU / IoU | +2 s mIoU / IoU | +3 s mIoU / IoU | Checkpoint |
| --- | ---: | ---: | ---: | --- |
| ForecastOcc | `24.84 / 34.77` | `19.36 / 29.85` | `17.11 / 27.67` | [Download](https://forecastocc.cs.uni-freiburg.de/download/forecastocc.pth) |

## 👩‍⚖️ License

For academic usage, the code is released under the [GPLv3](https://www.gnu.org/licenses/gpl-3.0.en.html) license.
For any commercial purpose, please contact the authors.

## 🙏 Acknowledgment

ForecastOcc builds on
[MMDetection3D](https://github.com/open-mmlab/mmdetection3d),
[BEVDet](https://github.com/HuangJunJie2017/BEVDet),
[COTR](https://github.com/NotACracker/COTR), and
[Occ3D](https://github.com/Tsinghua-MARS-Lab/Occ3D).

This work was funded by the German Research Foundation (DFG) Emmy Noether Program,
grant number **468878300** and the Bosch Research collaboration on AI-driven automated
driving.

## 📬 Contacts

* [Riya Mohan](mailto:riyamohan1813@gmail.com)
* [Rohit Mohan](mailto:mohan@cs.uni-freiburg.de)
