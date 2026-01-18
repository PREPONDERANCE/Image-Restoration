# 基于国产深度框架 Jittor 计图的训练与部署解决方案

<p align="center">
    <br>
    <img src="assets/banner.jpeg"/>
    <br>
<p>

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.8.20-blue.svg" alt="Python 3.8.20" />
  <img src="https://img.shields.io/badge/Jittor-1.3.10.0-orange.svg" alt="Jittor 1.3.10.0" />
  <a href="https://github.com/PREPONDERANCE/Image-Restoration/pulls">
    <img src="https://img.shields.io/badge/PR-Welcome-10b981.svg" />
  </a>
  <a href="https://github.com/PREPONDERANCE/Image-Restoration/blob/main/LICENSE">
    <img src="https://img.shields.io/badge/License-Apache_2.0-e11d48.svg" />
  </a>
  <a href="README.md"><img src="https://img.shields.io/badge/README-EN-red.svg" /></a>
  <a href="README_CN.md"><img src="https://img.shields.io/badge/README-CN-yellow.svg" /></a>
</p>

## 📖 目录

- [📝 简介](#-简介)
- [🎉 新闻](#-新闻)
- [🛠️ 安装](#️-安装)
- [✨ 使用](#-使用)
- [🏛 License](#-license)
- [📎 引用](#-引用)

## 📝 简介

**图像复原**是计算机视觉领域的重要研究方向，也是提升视觉感知系统实用性与可靠性的关键基础技术。在实际应用场景中，成像过程常受到噪声、模糊、低分辨率、遮挡及退化环境等因素影响，导致获取的图像质量下降，严重制约后续的视觉理解与分析任务。因此，高性能的图像复原模型需要充分刻画图像退化机理，精准建模图像结构与纹理先验，从受损观测中重建出清晰、真实且细节丰富的高质量图像。然而，由于图像退化过程的多样性与不确定性，以及真实场景中复杂空间与语义信息的高度耦合，如何在不同退化条件下实现鲁棒、高泛化能力的图像复原，仍是该领域亟待突破的核心挑战之一。

Jittor-based Image Restoration Framework(JIRF)是由南开大学计算机视觉团队提供的官方框架，基于国产化高性能深度学习框架计图（Jittor）进行情感计算方法的训练与部署。目前，JIRF 框架已支持多种先进的图像复原任务与模型，包括图像去噪、去模糊、超分辨率及真实场景退化图像复原等方向。基于 Jittor 国产框架的高效编译与算子优化机制，图像复原模型在部署阶段的推理速度相比 PyTorch 可提升约 1.1 至 1.6 倍，从而为下游应用场景如智能安防图像增强、低照度成像质量提升、遥感影像复原及工业视觉检测等提供稳定、高效的技术支撑。

Jittor国产深度学习框架能够无缝兼容主流的PyTorch框架。以[AST](https://openaccess.thecvf.com/content/CVPR2024/papers/Zhou_Adapt_or_Perish_Adaptive_Sparse_Transformer_with_Attentive_Feature_Refinement_CVPR_2024_paper.pdf)网络架构为例，在兼容修改到JIRF中时，模型代码仅需要修改10余处即可完成转换，大幅降低了迁移成本。我们诚邀更多研究者参与，共同推进图像复原领域的国产化进程！让我们携手打造更强大的国产AI生态！

在Jittor深度学习框架助力下，该项目已支持图像复原领域中的最新工作：

| 工作                                                                                                                                                                        | 训练                                                                              | 测试                                                                             |
| --------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | --------------------------------------------------------------------------------- | -------------------------------------------------------------------------------- |
| [ECCV'24 FPro](https://arxiv.org/pdf/2404.00288)                                                                                                                            | [训练脚本](https://github.com/PREPONDERANCE/Image-Restoration/blob/main/train.sh) | [测试脚本](https://github.com/PREPONDERANCE/Image-Restoration/blob/main/test.sh) |
| [ICCV'25 HINT](https://arxiv.org/abs/2503.20174)                                                                                                                            | [训练脚本](https://github.com/PREPONDERANCE/Image-Restoration/blob/main/train.sh) | [测试脚本](https://github.com/PREPONDERANCE/Image-Restoration/blob/main/test.sh) |
| [CVPR'24 AST](https://openaccess.thecvf.com/content/CVPR2024/papers/Zhou_Adapt_or_Perish_Adaptive_Sparse_Transformer_with_Attentive_Feature_Refinement_CVPR_2024_paper.pdf) | [训练脚本](https://github.com/PREPONDERANCE/Image-Restoration/blob/main/train.sh) | [测试脚本](https://github.com/PREPONDERANCE/Image-Restoration/blob/main/test.sh) |
| [TPAMI'25 ASTv2](https://ieeexplore.ieee.org/document/11106710)                                                                                                             | [训练脚本](https://github.com/PREPONDERANCE/Image-Restoration/blob/main/train.sh) | [测试脚本](https://github.com/PREPONDERANCE/Image-Restoration/blob/main/test.sh) |

## 🎉 新闻

- 🎁 2026.1.17: 项目初始化。本项目支持四项图像复原任务，包括[FPro](https://arxiv.org/pdf/2404.00288), [HINT](https://arxiv.org/abs/2503.20174), [AST](https://openaccess.thecvf.com/content/CVPR2024/papers/Zhou_Adapt_or_Perish_Adaptive_Sparse_Transformer_with_Attentive_Feature_Refinement_CVPR_2024_paper.pdf), [ASTv2](https://ieeexplore.ieee.org/document/11106710)。所有方法均提供了训练和测试脚本。

## 🛠️ 安装

#### Pip 安装

```bash
conda create -n ir python=3.8
conda activate ir
pip install -r requirements.txt
```

#### Conda 安装

```bash
conda env create --file="env-jt.yml"
```

## ✨ 使用

### 训练

#### 命令行

```sh
sh train.sh Enhancement/Options/HINT_LOL_v2_synthetic.yml
```

- 该脚本只需要一个参数：模型的 YML 配置文件。
- 如果您希望使用其他图像复原模型，只需修改该参数来指定对应模型的配置文件，如`Dehaze/Options/RealDehazing_FPro.yml`。

#### 支持的任务和模型

| 任务     | 数据集          | FPro | HINT | AST | ASTv2 |
| -------- | --------------- | :--: | :--: | :-: | :---: |
| 雾       | SOTS            |  ✅  |  ✅  | ❌  |  ❌   |
| 摩尔纹   | TIP18           |  ✅  |  ❌  | ❌  |  ❌   |
| 噪声     | BSD68           |  ❌  |  ✅  | ❌  |  ❌   |
| 噪声     | Urban100        |  ❌  |  ✅  | ❌  |  ❌   |
| 雨       | Rain100L        |  ❌  |  ✅  | ❌  |  ❌   |
| 雨       | SPAD            |  ✅  |  ❌  | ❌  |  ❌   |
| 雨       | AGAN            |  ✅  |  ❌  | ❌  |  ❌   |
| 雪       | Snow100K        |  ❌  |  ✅  | ❌  |  ✅   |
| 低光     | LOLv2 Real      |  ❌  |  ✅  | ❌  |  ❌   |
| 低光     | LOLv2 Synthetic |  ❌  |  ✅  | ✅  |  ✅   |
| 运动模糊 | GoPro           |  ✅  |  ❌  | ❌  |  ✅   |
| 运动模糊 | RealBlur-J      |  ✅  |  ❌  | ❌  |  ✅   |
| 运动模糊 | RealBlur-R      |  ✅  |  ❌  | ❌  |  ✅   |

### 测试

#### 命令行

```sh
sh test.sh
```

- 该脚本不需要参数，所有的测试脚本已经写入该文件中。
- 使用该脚本时，请修改测试脚本参数，参数选项解析如下。

#### 参数解析

- `--opt` 模型配置文件地址，如 `Enhancement/Options/AST_LOL_v2_synthetic.yml`
- `--result_dir` 图像复原结果存储地址
- `--weights` 模型权重文件地址
- `--gpus` 测试使用的 GPU，该参数可认为是 `CUDA_VISIBLE_DEVICES` 的包装
- `--gt_dir` GT 图像存储地址
- `--input_dir` 用于额外指定缺陷图像存储地址，一般缺陷图像地址由 opt 文件指定，部分脚本显式要求该参数，具体参考脚本文件。

#### 支持的任务和模型

| 任务 | 数据集          | FPro | HINT | AST | ASTv2 |
| ---- | --------------- | :--: | :--: | :-: | :---: |
| 雾   | SOTS            |  ✅  |  ✅  | ❌  |  ❌   |
| 低光 | LOLv2 Real      |  ❌  |  ✅  | ❌  |  ❌   |
| 低光 | LOLv2 Synthetic |  ❌  |  ✅  | ✅  |  ✅   |

#### 自定义测试

所有的测试脚本均配备有上述参数（除 `input_dir`），如您需要自定义测试，请仿照现有脚本，并修改对应超参数。

## 🏛 License

本框架使用[Apache License (Version 2.0)](https://github.com/modelscope/modelscope/blob/master/LICENSE)进行许可。模型和数据集请查看原资源页面并遵守对应License。

## 📎 引用

```bibtex
@inproceedings{zhou_TPAMI25_astv2,
  title={Learning An Adaptive Sparse Transformer for Efficient Image Restoration},
  author={Zhou, Shihao and Pan, Jinshan and Yang, Jufeng},
  booktitle={TPAMI},
  year={2025}
}
```

```bibtex
@inproceedings{zhou_ICCV25_HINT,
  title={Devil is in the Uniformity: Exploring Diverse Learners within Transformer for Image Restoration},
  author={Zhou, Shihao and Li, Dayu and Pan, Jinshan and Zhou, Juncheng and Shi, Jinglei and Yang, Jufeng},
  booktitle={ICCV},
  year={2025}
}
```

```bibtex
@inproceedings{zhou_ECCV2024_FPro,
  title={Seeing the Unseen: A Frequency Prompt Guided Transformer for Image Restoration},
  author={Zhou, Shihao and Pan, Jinshan and Shi, Jinglei and Chen, Duosheng and Qu, Lishen and Yang, Jufeng},
  booktitle={ECCV},
  year={2024}
}
```

```bibtex
@inproceedings{zhou2024AST,
  title={Adapt or Perish: Adaptive Sparse Transformer with Attentive Feature Refinement for Image Restoration},
  author={Zhou, Shihao and Chen, Duosheng and Pan, Jinshan and Shi, Jinglei and Yang, Jufeng},
  booktitle={CVPR},
  year={2024}
}
```
