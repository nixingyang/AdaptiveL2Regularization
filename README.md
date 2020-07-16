![Python](https://img.shields.io/badge/python-3.8-blue?style=flat-square&logo=python)
![TensorFlow](https://img.shields.io/badge/tensorflow-2.2.3-green?style=flat-square&logo=tensorflow)

# Adaptive L2 Regularization in Person Re-Identification

## Overview

We introduce an adaptive L2 regularization mechanism in the setting of person re-identification.
In the literature, it is common practice to utilize hand-picked regularization factors which remain constant throughout the training procedure.
Unlike existing approaches, the regularization factors in our proposed method are updated adaptively through backpropagation.
This is achieved by incorporating trainable scalar variables as the regularization factors, which are further fed into a scaled hard sigmoid function.
Extensive experiments on the Market-1501, DukeMTMC-reID and MSMT17 datasets validate the effectiveness of our framework.
Most notably, we obtain state-of-the-art performance on MSMT17, which is the largest dataset for person re-identification.
Source code is publicly available at https://github.com/nixingyang/AdaptiveL2Regularization.

## Environment

```bash
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
bash Miniconda3-latest-Linux-x86_64.sh
conda config --set auto_activate_base false
conda create --yes --name TensorFlow2.2 python=3.8
conda activate TensorFlow2.2
conda install --yes cudatoolkit=10.1 cudnn=7.6 -c nvidia
conda install --yes cython matplotlib numpy=1.18 pandas pydot scikit-learn
pip install tensorflow==2.2.3
pip install opencv-python
pip install albumentations --no-binary imgaug,albumentations
```

## Training

```bash
python3 -u solution.py --dataset_name "Market1501" --backbone_model_name "ResNet50"
```

- To train on other datasets, replace `"Market1501"` with `"DukeMTMC_reID"` or `"MSMT17"`.
- To train with deeper backbones, replace `"ResNet50"` with `"ResNet101"` or `"ResNet152"`.
- To evaluate on a subset of the complete test set, append `--testing_size 0.5` to the command. Alternatively, you may turn this feature off by using `--testing_size 0.0`.

## Evaluation

```bash
python3 -u solution.py --dataset_name "Market1501" --backbone_model_name "ResNet50" --pretrained_model_file_path "?.h5" --output_folder_path "evaluation_only" --evaluation_only --freeze_backbone_for_N_epochs 0 --testing_size 1.0 --evaluate_testing_every_N_epochs 1
```

- Fill in the `pretrained_model_file_path` argument using the h5 file obtained during training.
- To use the re-ranking method, append `--use_re_ranking` to the command.
- You need to run this separate evaluation procedure only if `testing_size` is not set to `1.0` during training.

## Model Zoo

| Dataset | Backbone | mAP | Weights |
| - | - | - |- |
| Market1501 | ResNet50 | 88.3 | [Link](https://1drv.ms/u/s!Av-teFsyVR6WjmR-Jys9yzGLnVqm) |
| DukeMTMC_reID | ResNet50 | 79.9 | [Link](https://1drv.ms/u/s!Av-teFsyVR6WjmOBpAY4nCdnTrH3) |
| MSMT17 | ResNet50 | 59.4 | [Link](https://1drv.ms/u/s!Av-teFsyVR6WjmJzh8DHdFc5edDK) |
| MSMT17 | ResNet152 | 62.2 | [Link](https://1drv.ms/u/s!Av-teFsyVR6WjmWPtZpcZkNSYtoi) |

## Acknowledgements

- Evaluation Metrics are adapted from [deep-person-reid](https://github.com/KaiyangZhou/deep-person-reid/blob/v1.0.6/torchreid/metrics/rank_cylib/rank_cy.pyx).
- Re-Ranking is adapted from [person-re-ranking](https://github.com/zhunzhong07/person-re-ranking/blob/master/python-version/re_ranking_ranklist.py).
- Random Erasing is adapted from [Random-Erasing](https://github.com/zhunzhong07/Random-Erasing/blob/master/transforms.py).
- Triplet Loss is adapted from [triplet-reid](https://github.com/VisualComputingInstitute/triplet-reid/blob/master/loss.py).

## Third-Party Implementation

- The [adaptive-l2-regularization-pytorch](https://github.com/duyuanchao/adaptive-l2-regularization-pytorch) repository from [duyuanchao](https://github.com/duyuanchao) in PyTorch.

## Citation

Please consider citing [this work](https://ieeexplore.ieee.org/document/9412481) if it helps your research.

```
@inproceedings{ni2021adaptive,
  author={Ni, Xingyang and Fang, Liang and Huttunen, Heikki},
  booktitle={2020 25th International Conference on Pattern Recognition (ICPR)},
  title={Adaptive L2 Regularization in Person Re-Identification},
  year={2021},
  volume={},
  number={},
  pages={9601-9607},
  doi={10.1109/ICPR48806.2021.9412481}
}
```
