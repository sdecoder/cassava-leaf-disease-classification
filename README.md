<div id="top"></div>
<!--
*** Thanks for checking out the Best-README-Template. If you have a suggestion
*** that would make this better, please fork the repo and create a pull request
*** or simply open an issue with the tag "enhancement".
*** Don't forget to give the project a star!
*** Thanks again! Now go create something AMAZING! :D
-->



<!-- PROJECT SHIELDS -->
<!--
*** I'm using markdown "reference style" links for readability.
*** Reference links are enclosed in brackets [ ] instead of parentheses ( ).
*** See the bottom of this document for the declaration of the reference variables
*** for contributors-url, forks-url, etc. This is an optional, concise syntax you may use.
*** https://www.markdownguide.org/basic-syntax/#reference-style-links

[![Contributors][contributors-shield]][contributors-url]
[![Forks][forks-shield]][forks-url]
[![Stargazers][stars-shield]][stars-url]
[![Issues][issues-shield]][issues-url]
[![MIT License][license-shield]][license-url]
[![LinkedIn][linkedin-shield]][linkedin-url]
-->


<!-- PROJECT LOGO -->
<br />
<div align="center">
  <a href="https://github.com/othneildrew/Best-README-Template">
    <img src="resources/header.png" alt="Logo">
  </a>

  <h3 align="center">Cassava Leaf Disease Classification</h3>

  <!--
  <p align="center">
    An awesome README template to jumpstart your projects!
    <br />
    <a href="https://github.com/othneildrew/Best-README-Template"><strong>Explore the docs »</strong></a>
    <br />
    <br />
    <a href="https://github.com/othneildrew/Best-README-Template">View Demo</a>
    ·
    <a href="https://github.com/othneildrew/Best-README-Template/issues">Report Bug</a>
    ·
    <a href="https://github.com/othneildrew/Best-README-Template/issues">Request Feature</a>
  </p>-->
</div>



<!-- TABLE OF CONTENTS -->
<details>
  <summary>Table of Contents</summary>
  <ol>
    <li>
      <a href="#about-the-project">About This Project</a>
      <ul>
        <li><a href="#built-with">Built With</a></li>
      </ul>
    </li>
    <li>
      <a href="#getting-started">Getting Started</a>
      <ul>
        <li><a href="#prerequisites">Prerequisites</a></li>
        <li><a href="#installation">Installation</a></li>
      </ul>
    </li>
    <li><a href="#usage">Usage</a></li>
    <li><a href="#roadmap">Roadmap</a></li>
    <li><a href="#contributing">Contributing</a></li>
    <li><a href="#license">License</a></li>
    <li><a href="#contact">Contact</a></li>
    <li><a href="#acknowledgments">Acknowledgments</a></li>
  </ol>
</details>



<!-- ABOUT THE PROJECT -->
## About The Project

<!--
[![Product Name Screen Shot][product-screenshot]](https://example.com)
-->

As the second-largest provider of carbohydrates in Africa, cassava is a key food security crop grown by smallholder farmers because it can withstand harsh conditions. At least 80% of household farms in Sub-Saharan Africa grow this starchy root, but viral diseases are major sources of poor yields. With the help of data science, it may be possible to identify common diseases so they can be treated.

Existing methods of disease detection require farmers to solicit the help of government-funded agricultural experts to visually inspect and diagnose the plants. This suffers from being labor-intensive, low-supply and costly. As an added challenge, effective solutions for farmers must perform well under significant constraints, since African farmers may only have access to mobile-quality cameras with low-bandwidth.

In this competition, we introduce a dataset of 21,367 labeled images collected during a regular survey in Uganda. Most images were crowdsourced from farmers taking photos of their gardens, and annotated by experts at the National Crops Resources Research Institute (NaCRRI) in collaboration with the AI lab at Makerere University, Kampala. This is in a format that most realistically represents what farmers would need to diagnose in real life.

Your task is to classify each cassava image into four disease categories or a fifth category indicating a healthy leaf. With your help, farmers may be able to quickly identify diseased plants, potentially saving their crops before they inflict irreparable damage.

### Prerequisites

All related SDK installed on the development platform.
* TensorRT
* CUDA
* CuDNN
* Anaconda
* Pytorch
<p align="right">(<a href="#top">back to top</a>)</p>

### Installation

1. Download all prerequisites from the NVIDIA official website;
2. Install the CUDA SDK;
3. Install the CuDNN SDK;
4. Install the TensorRT SDK;
5. Setup your Anaconda environment;
6. Install the Pytorch CUDA/ROCm version;
<p align="right">(<a href="#top">back to top</a>)</p>

## Workflow

### Model training

The basic model structure is list as following:
```
Classifier(
  (model): EfficientNet(
    (conv_stem): Conv2dSame(3, 48, kernel_size=(3, 3), stride=(2, 2), bias=False)
    (bn1): BatchNormAct2d(
      48, eps=0.001, momentum=0.1, affine=True, track_running_stats=True
      (drop): Identity()
      (act): SiLU(inplace=True)
    )
    (blocks): Sequential(
      (0): Sequential(
        (0): DepthwiseSeparableConv(
          (conv_dw): Conv2d(48, 48, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), groups=48, bias=False)
          (bn1): BatchNormAct2d(
            48, eps=0.001, momentum=0.1, affine=True, track_running_stats=True
            (drop): Identity()
            (act): SiLU(inplace=True)
          )
          (se): SqueezeExcite(
            (conv_reduce): Conv2d(48, 12, kernel_size=(1, 1), stride=(1, 1))
            (act1): SiLU(inplace=True)
            (conv_expand): Conv2d(12, 48, kernel_size=(1, 1), stride=(1, 1))
            (gate): Sigmoid()
          )
          (conv_pw): Conv2d(48, 24, kernel_size=(1, 1), stride=(1, 1), bias=False)
          (bn2): BatchNormAct2d(
            24, eps=0.001, momentum=0.1, affine=True, track_running_stats=True
            (drop): Identity()
            (act): Identity()
          )
          (drop_path): Identity()
        )
        (1): DepthwiseSeparableConv(
          (conv_dw): Conv2d(24, 24, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), groups=24, bias=False)
          (bn1): BatchNormAct2d(
            24, eps=0.001, momentum=0.1, affine=True, track_running_stats=True
            (drop): Identity()
            (act): SiLU(inplace=True)
          )
          (se): SqueezeExcite(
            (conv_reduce): Conv2d(24, 6, kernel_size=(1, 1), stride=(1, 1))
            (act1): SiLU(inplace=True)
            (conv_expand): Conv2d(6, 24, kernel_size=(1, 1), stride=(1, 1))
            (gate): Sigmoid()
          )
          (conv_pw): Conv2d(24, 24, kernel_size=(1, 1), stride=(1, 1), bias=False)
          (bn2): BatchNormAct2d(
            24, eps=0.001, momentum=0.1, affine=True, track_running_stats=True
            (drop): Identity()
            (act): Identity()
          )
          (drop_path): Identity()
        )
      )
      (1): Sequential(
        (0): InvertedResidual(
          (conv_pw): Conv2d(24, 144, kernel_size=(1, 1), stride=(1, 1), bias=False)
          (bn1): BatchNormAct2d(
            144, eps=0.001, momentum=0.1, affine=True, track_running_stats=True
            (drop): Identity()
            (act): SiLU(inplace=True)
          )
          (conv_dw): Conv2dSame(144, 144, kernel_size=(3, 3), stride=(2, 2), groups=144, bias=False)
          (bn2): BatchNormAct2d(
            144, eps=0.001, momentum=0.1, affine=True, track_running_stats=True
            (drop): Identity()
            (act): SiLU(inplace=True)
          )
          (se): SqueezeExcite(
            (conv_reduce): Conv2d(144, 6, kernel_size=(1, 1), stride=(1, 1))
            (act1): SiLU(inplace=True)
            (conv_expand): Conv2d(6, 144, kernel_size=(1, 1), stride=(1, 1))
            (gate): Sigmoid()
          )
          (conv_pwl): Conv2d(144, 32, kernel_size=(1, 1), stride=(1, 1), bias=False)
          (bn3): BatchNormAct2d(
            32, eps=0.001, momentum=0.1, affine=True, track_running_stats=True
            (drop): Identity()
            (act): Identity()
          )
          (drop_path): Identity()
        )
        (1): InvertedResidual(
          (conv_pw): Conv2d(32, 192, kernel_size=(1, 1), stride=(1, 1), bias=False)
          (bn1): BatchNormAct2d(
            192, eps=0.001, momentum=0.1, affine=True, track_running_stats=True
            (drop): Identity()
            (act): SiLU(inplace=True)
          )
          (conv_dw): Conv2d(192, 192, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), groups=192, bias=False)
          (bn2): BatchNormAct2d(
            192, eps=0.001, momentum=0.1, affine=True, track_running_stats=True
            (drop): Identity()
            (act): SiLU(inplace=True)
          )
          (se): SqueezeExcite(
            (conv_reduce): Conv2d(192, 8, kernel_size=(1, 1), stride=(1, 1))
            (act1): SiLU(inplace=True)
            (conv_expand): Conv2d(8, 192, kernel_size=(1, 1), stride=(1, 1))
            (gate): Sigmoid()
          )
          (conv_pwl): Conv2d(192, 32, kernel_size=(1, 1), stride=(1, 1), bias=False)
          (bn3): BatchNormAct2d(
            32, eps=0.001, momentum=0.1, affine=True, track_running_stats=True
            (drop): Identity()
            (act): Identity()
          )
          (drop_path): Identity()
        )
        (2): InvertedResidual(
          (conv_pw): Conv2d(32, 192, kernel_size=(1, 1), stride=(1, 1), bias=False)
          (bn1): BatchNormAct2d(
            192, eps=0.001, momentum=0.1, affine=True, track_running_stats=True
            (drop): Identity()
            (act): SiLU(inplace=True)
          )
          (conv_dw): Conv2d(192, 192, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), groups=192, bias=False)
          (bn2): BatchNormAct2d(
            192, eps=0.001, momentum=0.1, affine=True, track_running_stats=True
            (drop): Identity()
            (act): SiLU(inplace=True)
          )
          (se): SqueezeExcite(
            (conv_reduce): Conv2d(192, 8, kernel_size=(1, 1), stride=(1, 1))
            (act1): SiLU(inplace=True)
            (conv_expand): Conv2d(8, 192, kernel_size=(1, 1), stride=(1, 1))
            (gate): Sigmoid()
          )
          (conv_pwl): Conv2d(192, 32, kernel_size=(1, 1), stride=(1, 1), bias=False)
          (bn3): BatchNormAct2d(
            32, eps=0.001, momentum=0.1, affine=True, track_running_stats=True
            (drop): Identity()
            (act): Identity()
          )
          (drop_path): Identity()
        )
        (3): InvertedResidual(
          (conv_pw): Conv2d(32, 192, kernel_size=(1, 1), stride=(1, 1), bias=False)
          (bn1): BatchNormAct2d(
            192, eps=0.001, momentum=0.1, affine=True, track_running_stats=True
            (drop): Identity()
            (act): SiLU(inplace=True)
          )
          (conv_dw): Conv2d(192, 192, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), groups=192, bias=False)
          (bn2): BatchNormAct2d(
            192, eps=0.001, momentum=0.1, affine=True, track_running_stats=True
            (drop): Identity()
            (act): SiLU(inplace=True)
          )
          (se): SqueezeExcite(
            (conv_reduce): Conv2d(192, 8, kernel_size=(1, 1), stride=(1, 1))
            (act1): SiLU(inplace=True)
            (conv_expand): Conv2d(8, 192, kernel_size=(1, 1), stride=(1, 1))
            (gate): Sigmoid()
          )
          (conv_pwl): Conv2d(192, 32, kernel_size=(1, 1), stride=(1, 1), bias=False)
          (bn3): BatchNormAct2d(
            32, eps=0.001, momentum=0.1, affine=True, track_running_stats=True
            (drop): Identity()
            (act): Identity()
          )
          (drop_path): Identity()
        )
      )
      (2): Sequential(
        (0): InvertedResidual(
          (conv_pw): Conv2d(32, 192, kernel_size=(1, 1), stride=(1, 1), bias=False)
          (bn1): BatchNormAct2d(
            192, eps=0.001, momentum=0.1, affine=True, track_running_stats=True
            (drop): Identity()
            (act): SiLU(inplace=True)
          )
          (conv_dw): Conv2dSame(192, 192, kernel_size=(5, 5), stride=(2, 2), groups=192, bias=False)
          (bn2): BatchNormAct2d(
            192, eps=0.001, momentum=0.1, affine=True, track_running_stats=True
            (drop): Identity()
            (act): SiLU(inplace=True)
          )
          (se): SqueezeExcite(
            (conv_reduce): Conv2d(192, 8, kernel_size=(1, 1), stride=(1, 1))
            (act1): SiLU(inplace=True)
            (conv_expand): Conv2d(8, 192, kernel_size=(1, 1), stride=(1, 1))
            (gate): Sigmoid()
          )
          (conv_pwl): Conv2d(192, 56, kernel_size=(1, 1), stride=(1, 1), bias=False)
          (bn3): BatchNormAct2d(
            56, eps=0.001, momentum=0.1, affine=True, track_running_stats=True
            (drop): Identity()
            (act): Identity()
          )
          (drop_path): Identity()
        )
        (1): InvertedResidual(
          (conv_pw): Conv2d(56, 336, kernel_size=(1, 1), stride=(1, 1), bias=False)
          (bn1): BatchNormAct2d(
            336, eps=0.001, momentum=0.1, affine=True, track_running_stats=True
            (drop): Identity()
            (act): SiLU(inplace=True)
          )
          (conv_dw): Conv2d(336, 336, kernel_size=(5, 5), stride=(1, 1), padding=(2, 2), groups=336, bias=False)
          (bn2): BatchNormAct2d(
            336, eps=0.001, momentum=0.1, affine=True, track_running_stats=True
            (drop): Identity()
            (act): SiLU(inplace=True)
          )
          (se): SqueezeExcite(
            (conv_reduce): Conv2d(336, 14, kernel_size=(1, 1), stride=(1, 1))
            (act1): SiLU(inplace=True)
            (conv_expand): Conv2d(14, 336, kernel_size=(1, 1), stride=(1, 1))
            (gate): Sigmoid()
          )
          (conv_pwl): Conv2d(336, 56, kernel_size=(1, 1), stride=(1, 1), bias=False)
          (bn3): BatchNormAct2d(
            56, eps=0.001, momentum=0.1, affine=True, track_running_stats=True
            (drop): Identity()
            (act): Identity()
          )
          (drop_path): Identity()
        )
        (2): InvertedResidual(
          (conv_pw): Conv2d(56, 336, kernel_size=(1, 1), stride=(1, 1), bias=False)
          (bn1): BatchNormAct2d(
            336, eps=0.001, momentum=0.1, affine=True, track_running_stats=True
            (drop): Identity()
            (act): SiLU(inplace=True)
          )
          (conv_dw): Conv2d(336, 336, kernel_size=(5, 5), stride=(1, 1), padding=(2, 2), groups=336, bias=False)
          (bn2): BatchNormAct2d(
            336, eps=0.001, momentum=0.1, affine=True, track_running_stats=True
            (drop): Identity()
            (act): SiLU(inplace=True)
          )
          (se): SqueezeExcite(
            (conv_reduce): Conv2d(336, 14, kernel_size=(1, 1), stride=(1, 1))
            (act1): SiLU(inplace=True)
            (conv_expand): Conv2d(14, 336, kernel_size=(1, 1), stride=(1, 1))
            (gate): Sigmoid()
          )
          (conv_pwl): Conv2d(336, 56, kernel_size=(1, 1), stride=(1, 1), bias=False)
          (bn3): BatchNormAct2d(
            56, eps=0.001, momentum=0.1, affine=True, track_running_stats=True
            (drop): Identity()
            (act): Identity()
          )
          (drop_path): Identity()
        )
        (3): InvertedResidual(
          (conv_pw): Conv2d(56, 336, kernel_size=(1, 1), stride=(1, 1), bias=False)
          (bn1): BatchNormAct2d(
            336, eps=0.001, momentum=0.1, affine=True, track_running_stats=True
            (drop): Identity()
            (act): SiLU(inplace=True)
          )
          (conv_dw): Conv2d(336, 336, kernel_size=(5, 5), stride=(1, 1), padding=(2, 2), groups=336, bias=False)
          (bn2): BatchNormAct2d(
            336, eps=0.001, momentum=0.1, affine=True, track_running_stats=True
            (drop): Identity()
            (act): SiLU(inplace=True)
          )
          (se): SqueezeExcite(
            (conv_reduce): Conv2d(336, 14, kernel_size=(1, 1), stride=(1, 1))
            (act1): SiLU(inplace=True)
            (conv_expand): Conv2d(14, 336, kernel_size=(1, 1), stride=(1, 1))
            (gate): Sigmoid()
          )
          (conv_pwl): Conv2d(336, 56, kernel_size=(1, 1), stride=(1, 1), bias=False)
          (bn3): BatchNormAct2d(
            56, eps=0.001, momentum=0.1, affine=True, track_running_stats=True
            (drop): Identity()
            (act): Identity()
          )
          (drop_path): Identity()
        )
      )
      (3): Sequential(
        (0): InvertedResidual(
          (conv_pw): Conv2d(56, 336, kernel_size=(1, 1), stride=(1, 1), bias=False)
          (bn1): BatchNormAct2d(
            336, eps=0.001, momentum=0.1, affine=True, track_running_stats=True
            (drop): Identity()
            (act): SiLU(inplace=True)
          )
          (conv_dw): Conv2dSame(336, 336, kernel_size=(3, 3), stride=(2, 2), groups=336, bias=False)
          (bn2): BatchNormAct2d(
            336, eps=0.001, momentum=0.1, affine=True, track_running_stats=True
            (drop): Identity()
            (act): SiLU(inplace=True)
          )
          (se): SqueezeExcite(
            (conv_reduce): Conv2d(336, 14, kernel_size=(1, 1), stride=(1, 1))
            (act1): SiLU(inplace=True)
            (conv_expand): Conv2d(14, 336, kernel_size=(1, 1), stride=(1, 1))
            (gate): Sigmoid()
          )
          (conv_pwl): Conv2d(336, 112, kernel_size=(1, 1), stride=(1, 1), bias=False)
          (bn3): BatchNormAct2d(
            112, eps=0.001, momentum=0.1, affine=True, track_running_stats=True
            (drop): Identity()
            (act): Identity()
          )
          (drop_path): Identity()
        )
        (1): InvertedResidual(
          (conv_pw): Conv2d(112, 672, kernel_size=(1, 1), stride=(1, 1), bias=False)
          (bn1): BatchNormAct2d(
            672, eps=0.001, momentum=0.1, affine=True, track_running_stats=True
            (drop): Identity()
            (act): SiLU(inplace=True)
          )
          (conv_dw): Conv2d(672, 672, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), groups=672, bias=False)
          (bn2): BatchNormAct2d(
            672, eps=0.001, momentum=0.1, affine=True, track_running_stats=True
            (drop): Identity()
            (act): SiLU(inplace=True)
          )
          (se): SqueezeExcite(
            (conv_reduce): Conv2d(672, 28, kernel_size=(1, 1), stride=(1, 1))
            (act1): SiLU(inplace=True)
            (conv_expand): Conv2d(28, 672, kernel_size=(1, 1), stride=(1, 1))
            (gate): Sigmoid()
          )
          (conv_pwl): Conv2d(672, 112, kernel_size=(1, 1), stride=(1, 1), bias=False)
          (bn3): BatchNormAct2d(
            112, eps=0.001, momentum=0.1, affine=True, track_running_stats=True
            (drop): Identity()
            (act): Identity()
          )
          (drop_path): Identity()
        )
        (2): InvertedResidual(
          (conv_pw): Conv2d(112, 672, kernel_size=(1, 1), stride=(1, 1), bias=False)
          (bn1): BatchNormAct2d(
            672, eps=0.001, momentum=0.1, affine=True, track_running_stats=True
            (drop): Identity()
            (act): SiLU(inplace=True)
          )
          (conv_dw): Conv2d(672, 672, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), groups=672, bias=False)
          (bn2): BatchNormAct2d(
            672, eps=0.001, momentum=0.1, affine=True, track_running_stats=True
            (drop): Identity()
            (act): SiLU(inplace=True)
          )
          (se): SqueezeExcite(
            (conv_reduce): Conv2d(672, 28, kernel_size=(1, 1), stride=(1, 1))
            (act1): SiLU(inplace=True)
            (conv_expand): Conv2d(28, 672, kernel_size=(1, 1), stride=(1, 1))
            (gate): Sigmoid()
          )
          (conv_pwl): Conv2d(672, 112, kernel_size=(1, 1), stride=(1, 1), bias=False)
          (bn3): BatchNormAct2d(
            112, eps=0.001, momentum=0.1, affine=True, track_running_stats=True
            (drop): Identity()
            (act): Identity()
          )
          (drop_path): Identity()
        )
        (3): InvertedResidual(
          (conv_pw): Conv2d(112, 672, kernel_size=(1, 1), stride=(1, 1), bias=False)
          (bn1): BatchNormAct2d(
            672, eps=0.001, momentum=0.1, affine=True, track_running_stats=True
            (drop): Identity()
            (act): SiLU(inplace=True)
          )
          (conv_dw): Conv2d(672, 672, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), groups=672, bias=False)
          (bn2): BatchNormAct2d(
            672, eps=0.001, momentum=0.1, affine=True, track_running_stats=True
            (drop): Identity()
            (act): SiLU(inplace=True)
          )
          (se): SqueezeExcite(
            (conv_reduce): Conv2d(672, 28, kernel_size=(1, 1), stride=(1, 1))
            (act1): SiLU(inplace=True)
            (conv_expand): Conv2d(28, 672, kernel_size=(1, 1), stride=(1, 1))
            (gate): Sigmoid()
          )
          (conv_pwl): Conv2d(672, 112, kernel_size=(1, 1), stride=(1, 1), bias=False)
          (bn3): BatchNormAct2d(
            112, eps=0.001, momentum=0.1, affine=True, track_running_stats=True
            (drop): Identity()
            (act): Identity()
          )
          (drop_path): Identity()
        )
        (4): InvertedResidual(
          (conv_pw): Conv2d(112, 672, kernel_size=(1, 1), stride=(1, 1), bias=False)
          (bn1): BatchNormAct2d(
            672, eps=0.001, momentum=0.1, affine=True, track_running_stats=True
            (drop): Identity()
            (act): SiLU(inplace=True)
          )
          (conv_dw): Conv2d(672, 672, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), groups=672, bias=False)
          (bn2): BatchNormAct2d(
            672, eps=0.001, momentum=0.1, affine=True, track_running_stats=True
            (drop): Identity()
            (act): SiLU(inplace=True)
          )
          (se): SqueezeExcite(
            (conv_reduce): Conv2d(672, 28, kernel_size=(1, 1), stride=(1, 1))
            (act1): SiLU(inplace=True)
            (conv_expand): Conv2d(28, 672, kernel_size=(1, 1), stride=(1, 1))
            (gate): Sigmoid()
          )
          (conv_pwl): Conv2d(672, 112, kernel_size=(1, 1), stride=(1, 1), bias=False)
          (bn3): BatchNormAct2d(
            112, eps=0.001, momentum=0.1, affine=True, track_running_stats=True
            (drop): Identity()
            (act): Identity()
          )
          (drop_path): Identity()
        )
        (5): InvertedResidual(
          (conv_pw): Conv2d(112, 672, kernel_size=(1, 1), stride=(1, 1), bias=False)
          (bn1): BatchNormAct2d(
            672, eps=0.001, momentum=0.1, affine=True, track_running_stats=True
            (drop): Identity()
            (act): SiLU(inplace=True)
          )
          (conv_dw): Conv2d(672, 672, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), groups=672, bias=False)
          (bn2): BatchNormAct2d(
            672, eps=0.001, momentum=0.1, affine=True, track_running_stats=True
            (drop): Identity()
            (act): SiLU(inplace=True)
          )
          (se): SqueezeExcite(
            (conv_reduce): Conv2d(672, 28, kernel_size=(1, 1), stride=(1, 1))
            (act1): SiLU(inplace=True)
            (conv_expand): Conv2d(28, 672, kernel_size=(1, 1), stride=(1, 1))
            (gate): Sigmoid()
          )
          (conv_pwl): Conv2d(672, 112, kernel_size=(1, 1), stride=(1, 1), bias=False)
          (bn3): BatchNormAct2d(
            112, eps=0.001, momentum=0.1, affine=True, track_running_stats=True
            (drop): Identity()
            (act): Identity()
          )
          (drop_path): Identity()
        )
      )
      (4): Sequential(
        (0): InvertedResidual(
          (conv_pw): Conv2d(112, 672, kernel_size=(1, 1), stride=(1, 1), bias=False)
          (bn1): BatchNormAct2d(
            672, eps=0.001, momentum=0.1, affine=True, track_running_stats=True
            (drop): Identity()
            (act): SiLU(inplace=True)
          )
          (conv_dw): Conv2d(672, 672, kernel_size=(5, 5), stride=(1, 1), padding=(2, 2), groups=672, bias=False)
          (bn2): BatchNormAct2d(
            672, eps=0.001, momentum=0.1, affine=True, track_running_stats=True
            (drop): Identity()
            (act): SiLU(inplace=True)
          )
          (se): SqueezeExcite(
            (conv_reduce): Conv2d(672, 28, kernel_size=(1, 1), stride=(1, 1))
            (act1): SiLU(inplace=True)
            (conv_expand): Conv2d(28, 672, kernel_size=(1, 1), stride=(1, 1))
            (gate): Sigmoid()
          )
          (conv_pwl): Conv2d(672, 160, kernel_size=(1, 1), stride=(1, 1), bias=False)
          (bn3): BatchNormAct2d(
            160, eps=0.001, momentum=0.1, affine=True, track_running_stats=True
            (drop): Identity()
            (act): Identity()
          )
          (drop_path): Identity()
        )
        (1): InvertedResidual(
          (conv_pw): Conv2d(160, 960, kernel_size=(1, 1), stride=(1, 1), bias=False)
          (bn1): BatchNormAct2d(
            960, eps=0.001, momentum=0.1, affine=True, track_running_stats=True
            (drop): Identity()
            (act): SiLU(inplace=True)
          )
          (conv_dw): Conv2d(960, 960, kernel_size=(5, 5), stride=(1, 1), padding=(2, 2), groups=960, bias=False)
          (bn2): BatchNormAct2d(
            960, eps=0.001, momentum=0.1, affine=True, track_running_stats=True
            (drop): Identity()
            (act): SiLU(inplace=True)
          )
          (se): SqueezeExcite(
            (conv_reduce): Conv2d(960, 40, kernel_size=(1, 1), stride=(1, 1))
            (act1): SiLU(inplace=True)
            (conv_expand): Conv2d(40, 960, kernel_size=(1, 1), stride=(1, 1))
            (gate): Sigmoid()
          )
          (conv_pwl): Conv2d(960, 160, kernel_size=(1, 1), stride=(1, 1), bias=False)
          (bn3): BatchNormAct2d(
            160, eps=0.001, momentum=0.1, affine=True, track_running_stats=True
            (drop): Identity()
            (act): Identity()
          )
          (drop_path): Identity()
        )
        (2): InvertedResidual(
          (conv_pw): Conv2d(160, 960, kernel_size=(1, 1), stride=(1, 1), bias=False)
          (bn1): BatchNormAct2d(
            960, eps=0.001, momentum=0.1, affine=True, track_running_stats=True
            (drop): Identity()
            (act): SiLU(inplace=True)
          )
          (conv_dw): Conv2d(960, 960, kernel_size=(5, 5), stride=(1, 1), padding=(2, 2), groups=960, bias=False)
          (bn2): BatchNormAct2d(
            960, eps=0.001, momentum=0.1, affine=True, track_running_stats=True
            (drop): Identity()
            (act): SiLU(inplace=True)
          )
          (se): SqueezeExcite(
            (conv_reduce): Conv2d(960, 40, kernel_size=(1, 1), stride=(1, 1))
            (act1): SiLU(inplace=True)
            (conv_expand): Conv2d(40, 960, kernel_size=(1, 1), stride=(1, 1))
            (gate): Sigmoid()
          )
          (conv_pwl): Conv2d(960, 160, kernel_size=(1, 1), stride=(1, 1), bias=False)
          (bn3): BatchNormAct2d(
            160, eps=0.001, momentum=0.1, affine=True, track_running_stats=True
            (drop): Identity()
            (act): Identity()
          )
          (drop_path): Identity()
        )
        (3): InvertedResidual(
          (conv_pw): Conv2d(160, 960, kernel_size=(1, 1), stride=(1, 1), bias=False)
          (bn1): BatchNormAct2d(
            960, eps=0.001, momentum=0.1, affine=True, track_running_stats=True
            (drop): Identity()
            (act): SiLU(inplace=True)
          )
          (conv_dw): Conv2d(960, 960, kernel_size=(5, 5), stride=(1, 1), padding=(2, 2), groups=960, bias=False)
          (bn2): BatchNormAct2d(
            960, eps=0.001, momentum=0.1, affine=True, track_running_stats=True
            (drop): Identity()
            (act): SiLU(inplace=True)
          )
          (se): SqueezeExcite(
            (conv_reduce): Conv2d(960, 40, kernel_size=(1, 1), stride=(1, 1))
            (act1): SiLU(inplace=True)
            (conv_expand): Conv2d(40, 960, kernel_size=(1, 1), stride=(1, 1))
            (gate): Sigmoid()
          )
          (conv_pwl): Conv2d(960, 160, kernel_size=(1, 1), stride=(1, 1), bias=False)
          (bn3): BatchNormAct2d(
            160, eps=0.001, momentum=0.1, affine=True, track_running_stats=True
            (drop): Identity()
            (act): Identity()
          )
          (drop_path): Identity()
        )
        (4): InvertedResidual(
          (conv_pw): Conv2d(160, 960, kernel_size=(1, 1), stride=(1, 1), bias=False)
          (bn1): BatchNormAct2d(
            960, eps=0.001, momentum=0.1, affine=True, track_running_stats=True
            (drop): Identity()
            (act): SiLU(inplace=True)
          )
          (conv_dw): Conv2d(960, 960, kernel_size=(5, 5), stride=(1, 1), padding=(2, 2), groups=960, bias=False)
          (bn2): BatchNormAct2d(
            960, eps=0.001, momentum=0.1, affine=True, track_running_stats=True
            (drop): Identity()
            (act): SiLU(inplace=True)
          )
          (se): SqueezeExcite(
            (conv_reduce): Conv2d(960, 40, kernel_size=(1, 1), stride=(1, 1))
            (act1): SiLU(inplace=True)
            (conv_expand): Conv2d(40, 960, kernel_size=(1, 1), stride=(1, 1))
            (gate): Sigmoid()
          )
          (conv_pwl): Conv2d(960, 160, kernel_size=(1, 1), stride=(1, 1), bias=False)
          (bn3): BatchNormAct2d(
            160, eps=0.001, momentum=0.1, affine=True, track_running_stats=True
            (drop): Identity()
            (act): Identity()
          )
          (drop_path): Identity()
        )
        (5): InvertedResidual(
          (conv_pw): Conv2d(160, 960, kernel_size=(1, 1), stride=(1, 1), bias=False)
          (bn1): BatchNormAct2d(
            960, eps=0.001, momentum=0.1, affine=True, track_running_stats=True
            (drop): Identity()
            (act): SiLU(inplace=True)
          )
          (conv_dw): Conv2d(960, 960, kernel_size=(5, 5), stride=(1, 1), padding=(2, 2), groups=960, bias=False)
          (bn2): BatchNormAct2d(
            960, eps=0.001, momentum=0.1, affine=True, track_running_stats=True
            (drop): Identity()
            (act): SiLU(inplace=True)
          )
          (se): SqueezeExcite(
            (conv_reduce): Conv2d(960, 40, kernel_size=(1, 1), stride=(1, 1))
            (act1): SiLU(inplace=True)
            (conv_expand): Conv2d(40, 960, kernel_size=(1, 1), stride=(1, 1))
            (gate): Sigmoid()
          )
          (conv_pwl): Conv2d(960, 160, kernel_size=(1, 1), stride=(1, 1), bias=False)
          (bn3): BatchNormAct2d(
            160, eps=0.001, momentum=0.1, affine=True, track_running_stats=True
            (drop): Identity()
            (act): Identity()
          )
          (drop_path): Identity()
        )
      )
      (5): Sequential(
        (0): InvertedResidual(
          (conv_pw): Conv2d(160, 960, kernel_size=(1, 1), stride=(1, 1), bias=False)
          (bn1): BatchNormAct2d(
            960, eps=0.001, momentum=0.1, affine=True, track_running_stats=True
            (drop): Identity()
            (act): SiLU(inplace=True)
          )
          (conv_dw): Conv2dSame(960, 960, kernel_size=(5, 5), stride=(2, 2), groups=960, bias=False)
          (bn2): BatchNormAct2d(
            960, eps=0.001, momentum=0.1, affine=True, track_running_stats=True
            (drop): Identity()
            (act): SiLU(inplace=True)
          )
          (se): SqueezeExcite(
            (conv_reduce): Conv2d(960, 40, kernel_size=(1, 1), stride=(1, 1))
            (act1): SiLU(inplace=True)
            (conv_expand): Conv2d(40, 960, kernel_size=(1, 1), stride=(1, 1))
            (gate): Sigmoid()
          )
          (conv_pwl): Conv2d(960, 272, kernel_size=(1, 1), stride=(1, 1), bias=False)
          (bn3): BatchNormAct2d(
            272, eps=0.001, momentum=0.1, affine=True, track_running_stats=True
            (drop): Identity()
            (act): Identity()
          )
          (drop_path): Identity()
        )
        (1): InvertedResidual(
          (conv_pw): Conv2d(272, 1632, kernel_size=(1, 1), stride=(1, 1), bias=False)
          (bn1): BatchNormAct2d(
            1632, eps=0.001, momentum=0.1, affine=True, track_running_stats=True
            (drop): Identity()
            (act): SiLU(inplace=True)
          )
          (conv_dw): Conv2d(1632, 1632, kernel_size=(5, 5), stride=(1, 1), padding=(2, 2), groups=1632, bias=False)
          (bn2): BatchNormAct2d(
            1632, eps=0.001, momentum=0.1, affine=True, track_running_stats=True
            (drop): Identity()
            (act): SiLU(inplace=True)
          )
          (se): SqueezeExcite(
            (conv_reduce): Conv2d(1632, 68, kernel_size=(1, 1), stride=(1, 1))
            (act1): SiLU(inplace=True)
            (conv_expand): Conv2d(68, 1632, kernel_size=(1, 1), stride=(1, 1))
            (gate): Sigmoid()
          )
          (conv_pwl): Conv2d(1632, 272, kernel_size=(1, 1), stride=(1, 1), bias=False)
          (bn3): BatchNormAct2d(
            272, eps=0.001, momentum=0.1, affine=True, track_running_stats=True
            (drop): Identity()
            (act): Identity()
          )
          (drop_path): Identity()
        )
        (2): InvertedResidual(
          (conv_pw): Conv2d(272, 1632, kernel_size=(1, 1), stride=(1, 1), bias=False)
          (bn1): BatchNormAct2d(
            1632, eps=0.001, momentum=0.1, affine=True, track_running_stats=True
            (drop): Identity()
            (act): SiLU(inplace=True)
          )
          (conv_dw): Conv2d(1632, 1632, kernel_size=(5, 5), stride=(1, 1), padding=(2, 2), groups=1632, bias=False)
          (bn2): BatchNormAct2d(
            1632, eps=0.001, momentum=0.1, affine=True, track_running_stats=True
            (drop): Identity()
            (act): SiLU(inplace=True)
          )
          (se): SqueezeExcite(
            (conv_reduce): Conv2d(1632, 68, kernel_size=(1, 1), stride=(1, 1))
            (act1): SiLU(inplace=True)
            (conv_expand): Conv2d(68, 1632, kernel_size=(1, 1), stride=(1, 1))
            (gate): Sigmoid()
          )
          (conv_pwl): Conv2d(1632, 272, kernel_size=(1, 1), stride=(1, 1), bias=False)
          (bn3): BatchNormAct2d(
            272, eps=0.001, momentum=0.1, affine=True, track_running_stats=True
            (drop): Identity()
            (act): Identity()
          )
          (drop_path): Identity()
        )
        (3): InvertedResidual(
          (conv_pw): Conv2d(272, 1632, kernel_size=(1, 1), stride=(1, 1), bias=False)
          (bn1): BatchNormAct2d(
            1632, eps=0.001, momentum=0.1, affine=True, track_running_stats=True
            (drop): Identity()
            (act): SiLU(inplace=True)
          )
          (conv_dw): Conv2d(1632, 1632, kernel_size=(5, 5), stride=(1, 1), padding=(2, 2), groups=1632, bias=False)
          (bn2): BatchNormAct2d(
            1632, eps=0.001, momentum=0.1, affine=True, track_running_stats=True
            (drop): Identity()
            (act): SiLU(inplace=True)
          )
          (se): SqueezeExcite(
            (conv_reduce): Conv2d(1632, 68, kernel_size=(1, 1), stride=(1, 1))
            (act1): SiLU(inplace=True)
            (conv_expand): Conv2d(68, 1632, kernel_size=(1, 1), stride=(1, 1))
            (gate): Sigmoid()
          )
          (conv_pwl): Conv2d(1632, 272, kernel_size=(1, 1), stride=(1, 1), bias=False)
          (bn3): BatchNormAct2d(
            272, eps=0.001, momentum=0.1, affine=True, track_running_stats=True
            (drop): Identity()
            (act): Identity()
          )
          (drop_path): Identity()
        )
        (4): InvertedResidual(
          (conv_pw): Conv2d(272, 1632, kernel_size=(1, 1), stride=(1, 1), bias=False)
          (bn1): BatchNormAct2d(
            1632, eps=0.001, momentum=0.1, affine=True, track_running_stats=True
            (drop): Identity()
            (act): SiLU(inplace=True)
          )
          (conv_dw): Conv2d(1632, 1632, kernel_size=(5, 5), stride=(1, 1), padding=(2, 2), groups=1632, bias=False)
          (bn2): BatchNormAct2d(
            1632, eps=0.001, momentum=0.1, affine=True, track_running_stats=True
            (drop): Identity()
            (act): SiLU(inplace=True)
          )
          (se): SqueezeExcite(
            (conv_reduce): Conv2d(1632, 68, kernel_size=(1, 1), stride=(1, 1))
            (act1): SiLU(inplace=True)
            (conv_expand): Conv2d(68, 1632, kernel_size=(1, 1), stride=(1, 1))
            (gate): Sigmoid()
          )
          (conv_pwl): Conv2d(1632, 272, kernel_size=(1, 1), stride=(1, 1), bias=False)
          (bn3): BatchNormAct2d(
            272, eps=0.001, momentum=0.1, affine=True, track_running_stats=True
            (drop): Identity()
            (act): Identity()
          )
          (drop_path): Identity()
        )
        (5): InvertedResidual(
          (conv_pw): Conv2d(272, 1632, kernel_size=(1, 1), stride=(1, 1), bias=False)
          (bn1): BatchNormAct2d(
            1632, eps=0.001, momentum=0.1, affine=True, track_running_stats=True
            (drop): Identity()
            (act): SiLU(inplace=True)
          )
          (conv_dw): Conv2d(1632, 1632, kernel_size=(5, 5), stride=(1, 1), padding=(2, 2), groups=1632, bias=False)
          (bn2): BatchNormAct2d(
            1632, eps=0.001, momentum=0.1, affine=True, track_running_stats=True
            (drop): Identity()
            (act): SiLU(inplace=True)
          )
          (se): SqueezeExcite(
            (conv_reduce): Conv2d(1632, 68, kernel_size=(1, 1), stride=(1, 1))
            (act1): SiLU(inplace=True)
            (conv_expand): Conv2d(68, 1632, kernel_size=(1, 1), stride=(1, 1))
            (gate): Sigmoid()
          )
          (conv_pwl): Conv2d(1632, 272, kernel_size=(1, 1), stride=(1, 1), bias=False)
          (bn3): BatchNormAct2d(
            272, eps=0.001, momentum=0.1, affine=True, track_running_stats=True
            (drop): Identity()
            (act): Identity()
          )
          (drop_path): Identity()
        )
        (6): InvertedResidual(
          (conv_pw): Conv2d(272, 1632, kernel_size=(1, 1), stride=(1, 1), bias=False)
          (bn1): BatchNormAct2d(
            1632, eps=0.001, momentum=0.1, affine=True, track_running_stats=True
            (drop): Identity()
            (act): SiLU(inplace=True)
          )
          (conv_dw): Conv2d(1632, 1632, kernel_size=(5, 5), stride=(1, 1), padding=(2, 2), groups=1632, bias=False)
          (bn2): BatchNormAct2d(
            1632, eps=0.001, momentum=0.1, affine=True, track_running_stats=True
            (drop): Identity()
            (act): SiLU(inplace=True)
          )
          (se): SqueezeExcite(
            (conv_reduce): Conv2d(1632, 68, kernel_size=(1, 1), stride=(1, 1))
            (act1): SiLU(inplace=True)
            (conv_expand): Conv2d(68, 1632, kernel_size=(1, 1), stride=(1, 1))
            (gate): Sigmoid()
          )
          (conv_pwl): Conv2d(1632, 272, kernel_size=(1, 1), stride=(1, 1), bias=False)
          (bn3): BatchNormAct2d(
            272, eps=0.001, momentum=0.1, affine=True, track_running_stats=True
            (drop): Identity()
            (act): Identity()
          )
          (drop_path): Identity()
        )
        (7): InvertedResidual(
          (conv_pw): Conv2d(272, 1632, kernel_size=(1, 1), stride=(1, 1), bias=False)
          (bn1): BatchNormAct2d(
            1632, eps=0.001, momentum=0.1, affine=True, track_running_stats=True
            (drop): Identity()
            (act): SiLU(inplace=True)
          )
          (conv_dw): Conv2d(1632, 1632, kernel_size=(5, 5), stride=(1, 1), padding=(2, 2), groups=1632, bias=False)
          (bn2): BatchNormAct2d(
            1632, eps=0.001, momentum=0.1, affine=True, track_running_stats=True
            (drop): Identity()
            (act): SiLU(inplace=True)
          )
          (se): SqueezeExcite(
            (conv_reduce): Conv2d(1632, 68, kernel_size=(1, 1), stride=(1, 1))
            (act1): SiLU(inplace=True)
            (conv_expand): Conv2d(68, 1632, kernel_size=(1, 1), stride=(1, 1))
            (gate): Sigmoid()
          )
          (conv_pwl): Conv2d(1632, 272, kernel_size=(1, 1), stride=(1, 1), bias=False)
          (bn3): BatchNormAct2d(
            272, eps=0.001, momentum=0.1, affine=True, track_running_stats=True
            (drop): Identity()
            (act): Identity()
          )
          (drop_path): Identity()
        )
      )
      (6): Sequential(
        (0): InvertedResidual(
          (conv_pw): Conv2d(272, 1632, kernel_size=(1, 1), stride=(1, 1), bias=False)
          (bn1): BatchNormAct2d(
            1632, eps=0.001, momentum=0.1, affine=True, track_running_stats=True
            (drop): Identity()
            (act): SiLU(inplace=True)
          )
          (conv_dw): Conv2d(1632, 1632, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), groups=1632, bias=False)
          (bn2): BatchNormAct2d(
            1632, eps=0.001, momentum=0.1, affine=True, track_running_stats=True
            (drop): Identity()
            (act): SiLU(inplace=True)
          )
          (se): SqueezeExcite(
            (conv_reduce): Conv2d(1632, 68, kernel_size=(1, 1), stride=(1, 1))
            (act1): SiLU(inplace=True)
            (conv_expand): Conv2d(68, 1632, kernel_size=(1, 1), stride=(1, 1))
            (gate): Sigmoid()
          )
          (conv_pwl): Conv2d(1632, 448, kernel_size=(1, 1), stride=(1, 1), bias=False)
          (bn3): BatchNormAct2d(
            448, eps=0.001, momentum=0.1, affine=True, track_running_stats=True
            (drop): Identity()
            (act): Identity()
          )
          (drop_path): Identity()
        )
        (1): InvertedResidual(
          (conv_pw): Conv2d(448, 2688, kernel_size=(1, 1), stride=(1, 1), bias=False)
          (bn1): BatchNormAct2d(
            2688, eps=0.001, momentum=0.1, affine=True, track_running_stats=True
            (drop): Identity()
            (act): SiLU(inplace=True)
          )
          (conv_dw): Conv2d(2688, 2688, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), groups=2688, bias=False)
          (bn2): BatchNormAct2d(
            2688, eps=0.001, momentum=0.1, affine=True, track_running_stats=True
            (drop): Identity()
            (act): SiLU(inplace=True)
          )
          (se): SqueezeExcite(
            (conv_reduce): Conv2d(2688, 112, kernel_size=(1, 1), stride=(1, 1))
            (act1): SiLU(inplace=True)
            (conv_expand): Conv2d(112, 2688, kernel_size=(1, 1), stride=(1, 1))
            (gate): Sigmoid()
          )
          (conv_pwl): Conv2d(2688, 448, kernel_size=(1, 1), stride=(1, 1), bias=False)
          (bn3): BatchNormAct2d(
            448, eps=0.001, momentum=0.1, affine=True, track_running_stats=True
            (drop): Identity()
            (act): Identity()
          )
          (drop_path): Identity()
        )
      )
    )
    (conv_head): Conv2d(448, 1792, kernel_size=(1, 1), stride=(1, 1), bias=False)
    (bn2): BatchNormAct2d(
      1792, eps=0.001, momentum=0.1, affine=True, track_running_stats=True
      (drop): Identity()
      (act): SiLU(inplace=True)
    )
    (global_pool): SelectAdaptivePool2d (pool_type=avg, flatten=Flatten(start_dim=1, end_dim=-1))
    (classifier): Linear(in_features=1792, out_features=5, bias=True)
  )
)
```

This model is trained using Pytorch ROCm.
The training result can reach 99% precision.
After training, the weight is saved to the .pt file.

### Export to ONNX file

Reload the .pt file to construct the original Pytorch model.
Export the pytorch model to the .onnx format using following code snippet.

```python
# https://pytorch.org/tutorials/advanced/super_resolution_with_onnxruntime.html
# Input to the model
x = torch.randn(batch_size, 1, 224, 224, requires_grad=True)
torch_out = torch_model(x)

# Export the model
torch.onnx.export(torch_model,               # model being run
                  x,                         # model input (or a tuple for multiple inputs)
                  "super_resolution.onnx",   # where to save the model (can be a file or file-like object)
                  export_params=True,        # store the trained parameter weights inside the model file
                  opset_version=10,          # the ONNX version to export the model to
                  do_constant_folding=True,  # whether to execute constant folding for optimization
                  input_names = ['input'],   # the model's input names
                  output_names = ['output'], # the model's output names
                  dynamic_axes={'input' : {0 : 'batch_size'},    # variable length axes
                                'output' : {0 : 'batch_size'}})


```

Once the exporting is done, we can load the .onnx file for validation purpose.
The following snippet is used to load the .onnx file and use it as the inference engine.

```python
import onnx
onnx_model = onnx.load("super_resolution.onnx")
onnx.checker.check_model(onnx_model)

import onnxruntime
ort_session = onnxruntime.InferenceSession("super_resolution.onnx")
def to_numpy(tensor):
    return tensor.detach().cpu().numpy() if tensor.requires_grad else tensor.cpu().numpy()

# compute ONNX Runtime output prediction
ort_inputs = {ort_session.get_inputs()[0].name: to_numpy(x)}
ort_outs = ort_session.run(None, ort_inputs)
# compare ONNX Runtime and PyTorch results
np.testing.assert_allclose(to_numpy(torch_out), ort_outs[0], rtol=1e-03, atol=1e-05)
print("Exported model has been tested with ONNXRuntime, and the result looks good!")

```

### Export to TensorRT engine file
Once the conversion and validation for the .onnx file is done,
we can convert it to the TensorRT engine at this point.
Once the TensorRT engine is generated, we run the test again to make sure its precision keeps.

In our experiment setup, all INT8/FP16/TF32/FP32 configuration remains at 99% precision.
Note: If the selected precision is INT8, the calibrator dataset should be provided.


### Performance evaluation

The performance evaluation is done using a Tesla M40 card.
Corresponding specification can be found here:
https://www.techpowerup.com/gpu-specs/tesla-m40-24-gb.c3838



1. for the INT8 precision:
```shell
trtexec --loadEngine=resnet50.int8.engine --batch=8192 --streams=8 --verbose --avgRuns=10
[07/22/2022-23:37:09] [I] === Performance summary ===
[07/22/2022-23:37:09] [I] Throughput: 1.29477e+08 qps
[07/22/2022-23:37:09] [I] Latency: min = 0.360107 ms, max = 0.677246 ms, mean = 0.521576 ms, median = 0.522217 ms, percentile(99%) = 0.615723 ms
[07/22/2022-23:37:09] [I] Enqueue Time: min = 0.0136719 ms, max = 0.0453186 ms, mean = 0.0148635 ms, median = 0.0142822 ms, percentile(99%) = 0.0231934 ms
[07/22/2022-23:37:09] [I] H2D Latency: min = 0.00708008 ms, max = 0.0322266 ms, mean = 0.0125107 ms, median = 0.0122681 ms, percentile(99%) = 0.0178833 ms
[07/22/2022-23:37:09] [I] GPU Compute Time: min = 0.338623 ms, max = 0.656982 ms, mean = 0.499129 ms, median = 0.499756 ms, percentile(99%) = 0.593628 ms
[07/22/2022-23:37:09] [I] D2H Latency: min = 0.00509644 ms, max = 0.0241699 ms, mean = 0.00993502 ms, median = 0.00976562 ms, percentile(99%) = 0.0151367 ms
[07/22/2022-23:37:09] [I] Total Host Walltime: 3.00126 s
[07/22/2022-23:37:09] [I] Total GPU Compute Time: 23.6767 s
[07/22/2022-23:37:09] [W] * GPU compute time is unstable, with coefficient of variance = 7.05215%.
[07/22/2022-23:37:09] [W]   If not already in use, locking GPU clock frequency or adding --useSpinWait may improve the stability.
[07/22/2022-23:37:09] [I] Explanations of the performance metrics are printed in the verbose logs.

```
2. for the FP16 precision:

```shell
trtexec --loadEngine=resnet50.fp16.engine --batch=8192 --streams=8 --verbose --avgRuns=10
[07/22/2022-23:56:06] [I] === Performance summary ===
[07/22/2022-23:56:06] [I] Throughput: 1.24304e+08 qps
[07/22/2022-23:56:06] [I] Latency: min = 0.118164 ms, max = 1.01929 ms, mean = 0.539201 ms, median = 0.538818 ms, percentile(99%) = 0.639679 ms
[07/22/2022-23:56:06] [I] Enqueue Time: min = 0.0134277 ms, max = 0.133057 ms, mean = 0.0165005 ms, median = 0.0148926 ms, percentile(99%) = 0.0349121 ms
[07/22/2022-23:56:06] [I] H2D Latency: min = 0.00549316 ms, max = 0.395752 ms, mean = 0.0115327 ms, median = 0.0113525 ms, percentile(99%) = 0.0170898 ms
[07/22/2022-23:56:06] [I] GPU Compute Time: min = 0.105591 ms, max = 0.750916 ms, mean = 0.517884 ms, median = 0.517578 ms, percentile(99%) = 0.618896 ms
[07/22/2022-23:56:06] [I] D2H Latency: min = 0.00500488 ms, max = 0.0252075 ms, mean = 0.00978184 ms, median = 0.00965881 ms, percentile(99%) = 0.0144043 ms
[07/22/2022-23:56:06] [I] Total Host Walltime: 3.00188 s
[07/22/2022-23:56:06] [I] Total GPU Compute Time: 23.5896 s
[07/22/2022-23:56:06] [W] * GPU compute time is unstable, with coefficient of variance = 7.3408%.
[07/22/2022-23:56:06] [W]   If not already in use, locking GPU clock frequency or adding --useSpinWait may improve the stability.
[07/22/2022-23:56:06] [I] Explanations of the performance metrics are printed in the verbose logs.
```

3. for the TF32 precision:

```shell
trtexec --loadEngine=resnet50.tf32.engine --batch=8192 --streams=8 --verbose --avgRuns=10
[07/22/2022-23:59:12] [I] === Performance summary ===
[07/22/2022-23:59:12] [I] Throughput: 1.25112e+08 qps
[07/22/2022-23:59:12] [I] Latency: min = 0.391785 ms, max = 0.707275 ms, mean = 0.537911 ms, median = 0.5354 ms, percentile(99%) = 0.636963 ms
[07/22/2022-23:59:12] [I] Enqueue Time: min = 0.0134277 ms, max = 0.0513916 ms, mean = 0.0146272 ms, median = 0.0141602 ms, percentile(99%) = 0.0234375 ms
[07/22/2022-23:59:12] [I] H2D Latency: min = 0.00683594 ms, max = 0.0310059 ms, mean = 0.011475 ms, median = 0.0112305 ms, percentile(99%) = 0.0170898 ms
[07/22/2022-23:59:12] [I] GPU Compute Time: min = 0.371033 ms, max = 0.688965 ms, mean = 0.516601 ms, median = 0.514069 ms, percentile(99%) = 0.61499 ms
[07/22/2022-23:59:12] [I] D2H Latency: min = 0.00488281 ms, max = 0.0249023q ms, mean = 0.00983961 ms, median = 0.00976562 ms, percentile(99%) = 0.0146484 ms
[07/22/2022-23:59:12] [I] Total Host Walltime: 3.00147 s
[07/22/2022-23:59:12] [I] Total GPU Compute Time: 23.681 s
[07/22/2022-23:59:12] [W] * GPU compute time is unstable, with coefficient of variance = 5.8492%.
[07/22/2022-23:59:12] [W]   If not already in use, locking GPU clock frequency or adding --useSpinWait may improve the stability.
[07/22/2022-23:59:12] [I] Explanations of the performance metrics are printed in the verbose logs.
```

4. for the FP32 precision:

```shell
trtexec --loadEngine=resnet50.fp32.engine --batch=8192 --streams=8 --verbose --avgRuns=10
[07/23/2022-00:01:31] [I] === Performance summary ===
[07/23/2022-00:01:31] [I] Throughput: 1.24417e+08 qps
[07/23/2022-00:01:31] [I] Latency: min = 0.395233 ms, max = 1.01245 ms, mean = 0.541006 ms, median = 0.540039 ms, percentile(99%) = 0.636841 ms
[07/23/2022-00:01:31] [I] Enqueue Time: min = 0.0134277 ms, max = 0.052887 ms, mean = 0.0152779 ms, median = 0.0143433 ms, percentile(99%) = 0.0285645 ms
[07/23/2022-00:01:31] [I] H2D Latency: min = 0.00683594 ms, max = 0.377075 ms, mean = 0.0115587 ms, median = 0.0114136 ms, percentile(99%) = 0.0166626 ms
[07/23/2022-00:01:31] [I] GPU Compute Time: min = 0.373535 ms, max = 0.713257 ms, mean = 0.519689 ms, median = 0.518799 ms, percentile(99%) = 0.615234 ms
[07/23/2022-00:01:31] [I] D2H Latency: min = 0.00488281 ms, max = 0.0258789 ms, mean = 0.00976126 ms, median = 0.00964355 ms, percentile(99%) = 0.0147705 ms
[07/23/2022-00:01:31] [I] Total Host Walltime: 3.00139 s
[07/23/2022-00:01:31] [I] Total GPU Compute Time: 23.6895 s
[07/23/2022-00:01:31] [W] * GPU compute time is unstable, with coefficient of variance = 5.72335%.
[07/23/2022-00:01:31] [W]   If not already in use, locking GPU clock frequency or adding --useSpinWait may improve the stability.
[07/23/2022-00:01:31] [I] Explanations of the performance metrics are printed in the verbose logs.
```

### Conclusion:

1. The file size of TensorRT engines don't vary too much.
2. The INT8 has achieved best performance.
3. Because we are using a small network in our template, so we may not expect too much performance boost.


<!-- BUILIT WITH
### Built With

This section should list any major frameworks/libraries used to bootstrap your project. Leave any add-ons/plugins for the acknowledgements section. Here are a few examples.
* [![Next][Next.js]][TensorRT]
<p align="right">(<a href="#top">back to top</a>)</p>
 -->

<!-- GETTING STARTED
## Getting Started

This is an example of how you may give instructions on setting up your project locally.
To get a local copy up and running follow these simple example steps.
-->




<!-- ROADMAP -->
## Roadmap

- [x] Add Changelog
- [x] Add back to top links
- [ ] Explorer more possibility
- [ ] Add "components" document to easily copy & paste sections of the readme
- [ ] Multi-language Support
    - [ ] Chinese
    - [ ] Spanish

See the [open issues](https://github.com/othneildrew/Best-README-Template/issues) for a full list of proposed features (and known issues).

<p align="right">(<a href="#top">back to top</a>)</p>



<!-- CONTRIBUTING
## Contributing

Contributions are what make the open source community such an amazing place to learn, inspire, and create. Any contributions you make are **greatly appreciated**.

If you have a suggestion that would make this better, please fork the repo and create a pull request. You can also simply open an issue with the tag "enhancement".
Don't forget to give the project a star! Thanks again!

1. Fork the Project
2. Create your Feature Branch (`git checkout -b feature/AmazingFeature`)
3. Commit your Changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the Branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

<p align="right">(<a href="#top">back to top</a>)</p> -->



<!-- LICENSE -->
## License

Distributed under the MIT License. See `LICENSE.txt` for more information.

<p align="right">(<a href="#top">back to top</a>)</p>



<!-- CONTACT
## Contact

Your Name - [@your_twitter](https://twitter.com/your_username) - email@example.com

Project Link: [https://github.com/your_username/repo_name](https://github.com/your_username/repo_name)

<p align="right">(<a href="#top">back to top</a>)</p> -->



<!-- ACKNOWLEDGMENTS
## Acknowledgments

Use this space to list resources you find helpful and would like to give credit to. I've included a few of my favorites to kick things off!

* [Choose an Open Source License](https://choosealicense.com)
* [GitHub Emoji Cheat Sheet](https://www.webpagefx.com/tools/emoji-cheat-sheet)
* [Malven's Flexbox Cheatsheet](https://flexbox.malven.co/)
* [Malven's Grid Cheatsheet](https://grid.malven.co/)
* [Img Shields](https://shields.io)
* [GitHub Pages](https://pages.github.com)
* [Font Awesome](https://fontawesome.com)
* [React Icons](https://react-icons.github.io/react-icons/search)

<p align="right">(<a href="#top">back to top</a>)</p> -->



<!-- MARKDOWN LINKS & IMAGES -->
<!-- https://www.markdownguide.org/basic-syntax/#reference-style-links -->
[contributors-shield]: https://img.shields.io/github/contributors/othneildrew/Best-README-Template.svg?style=for-the-badge
[contributors-url]: https://github.com/othneildrew/Best-README-Template/graphs/contributors
[forks-shield]: https://img.shields.io/github/forks/othneildrew/Best-README-Template.svg?style=for-the-badge
[forks-url]: https://github.com/othneildrew/Best-README-Template/network/members
[stars-shield]: https://img.shields.io/github/stars/othneildrew/Best-README-Template.svg?style=for-the-badge
[stars-url]: https://github.com/othneildrew/Best-README-Template/stargazers
[issues-shield]: https://img.shields.io/github/issues/othneildrew/Best-README-Template.svg?style=for-the-badge
[issues-url]: https://github.com/othneildrew/Best-README-Template/issues
[license-shield]: https://img.shields.io/github/license/othneildrew/Best-README-Template.svg?style=for-the-badge
[license-url]: https://github.com/othneildrew/Best-README-Template/blob/master/LICENSE.txt
[linkedin-shield]: https://img.shields.io/badge/-LinkedIn-black.svg?style=for-the-badge&logo=linkedin&colorB=555
[linkedin-url]: https://linkedin.com/in/othneildrew
[product-screenshot]: images/screenshot.png
[Next.js]: https://img.shields.io/badge/next.js-000000?style=for-the-badge&logo=nextdotjs&logoColor=white
[Next-url]: https://nextjs.org/
[React.js]: https://img.shields.io/badge/React-20232A?style=for-the-badge&logo=react&logoColor=61DAFB
[React-url]: https://reactjs.org/
[Vue.js]: https://img.shields.io/badge/Vue.js-35495E?style=for-the-badge&logo=vuedotjs&logoColor=4FC08D
[Vue-url]: https://vuejs.org/
[Angular.io]: https://img.shields.io/badge/Angular-DD0031?style=for-the-badge&logo=angular&logoColor=white
[Angular-url]: https://angular.io/
[Svelte.dev]: https://img.shields.io/badge/Svelte-4A4A55?style=for-the-badge&logo=svelte&logoColor=FF3E00
[Svelte-url]: https://svelte.dev/
[Laravel.com]: https://img.shields.io/badge/Laravel-FF2D20?style=for-the-badge&logo=laravel&logoColor=white
[Laravel-url]: https://laravel.com
[Bootstrap.com]: https://img.shields.io/badge/Bootstrap-563D7C?style=for-the-badge&logo=bootstrap&logoColor=white
[Bootstrap-url]: https://getbootstrap.com
[JQuery.com]: https://img.shields.io/badge/jQuery-0769AD?style=for-the-badge&logo=jquery&logoColor=white
[JQuery-url]: https://jquery.com
[TensorRT]: https://developer.nvidia.com/tensorrt

<!-- data

-->