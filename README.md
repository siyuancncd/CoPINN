# CoPINN

<p align="left">
  <img src="https://img.shields.io/badge/license-MIT-brightgreen" alt="License">
    <img src="https://komarev.com/ghpvc/?username=siyuancncd&repo=FUME" alt="GitHub Views">
</p>

Siyuan Duan*, Wenyuan Wu*, Peng Hu, Zhenwen Ren, Dezhong Peng, and Yuan Sun#. "[CoPINN: Cognitive Physics-Informed Neural Networks](https://openreview.net/forum?id=4vAa0A98xI&referrer=%5BAuthor%20Console%5D(%2Fgroup%3Fid%3DICML.cc%2F2025%2FConference%2FAuthors%23your-submissions))". (ICML 2025, Spotlight (acc rate = 2.6%), JAX Code)

- Poster is available at [here](https://github.com/siyuancncd/CoPINN/blob/main/CoPINN_poster.png).
- PPT is available at [here](https://github.com/siyuancncd/CoPINN/blob/main/CoPINN.pdf).
- Report is available at [here](https://mp.weixin.qq.com/s/3E3FdngVW6KShT5iiFAGRw).
- 中文讲解 在 [这里](https://www.bilibili.com/video/BV1sQ32zWEXA/?spm_id_from=333.1387.collection.video_card.click&vd_source=6c3619d976449503d4bb811d9044ee7f&t=6408).

<!--
## 

:bangbang: **I’m actively seeking a PhD position for Fall 2026 entry.** If you believe my background aligns with your research needs, please feel free to contact me via email at siyuanduancn@gmail.com.

-->
**Spotlight presentation**
<!--
https://github.com/user-attachments/assets/18d5e918-4d31-4ddb-b421-8f075f0fcf93
-->

<!--
https://github.com/user-attachments/assets/d279f7ac-a986-4bca-a1b9-9b34e6cbefa2
-->



https://github.com/user-attachments/assets/bdd80ddb-0a38-42ad-a25a-362b1317a2c1



## Abstract
Physics-informed neural networks (PINNs) aim to constrain the outputs and gradients of deep learning models to satisfy specified governing physics equations, which have demonstrated significant potential for solving partial differential equations (PDEs). Although existing PINN methods have achieved pleasing performance, they always treat both easy and hard sample points indiscriminately, especially ones in the physical boundaries. This easily causes the PINN model to fall into undesirable local minima and unstable learning, thereby resulting in an Unbalanced Prediction Problem (UPP). To deal with this daunting problem, we propose a novel framework named Cognitive Physics-Informed Neural Network (**CoPINN**) that imitates the human cognitive learning manner from easy to hard. Specifically, we first employ separable subnetworks to encode independent one-dimensional coordinates and apply an aggregation scheme to generate multi-dimensional predicted physical variables. Then, during the training phase, we dynamically evaluate the difficulty of each sample according to the gradient of the PDE residuals. Finally, we propose a cognitive training scheduler to progressively optimize the entire sampling regions from easy to hard, thereby embracing robustness and generalization against predicting physical boundary regions. Extensive experiments demonstrate that our CoPINN achieves state-of-the-art performance, particularly significantly reducing prediction errors in stubborn regions. 

## Motivation

<p align="center">
<img src="https://github.com/siyuancncd/CoPINN/blob/main/CoPINN_motivation.png" width="660" height="360">
</p>

The 2D and 3D visualization of the absolute error between the predicted and exact values. The left graph illustrates the absolute error of the entire three-dimensional space. The middle graph demonstrates the absolute error of the boundary when $y=-1$, while the right graph displays the absolute error of the cross-section when $y=0$. (a) The absolute error of SPINN (SOTA) on the Helmholtz equation. These results indicate that SPINN exhibits significantly larger errors near the physical boundary region compared to the middle region, which reveals the Unbalanced Prediction Problem (UPP). (b) The absolute error of our CoPINN on the Helmholtz equation, which shows that CoPINN maintains consistent small absolute errors both near the physical boundary and in the middle region.

## Framework

<p align="center">
<img src="https://github.com/siyuancncd/CoPINN/blob/main/CoPINN_framework.png" width="960" height="485">
</p>

## Experimental Results

<p align="center">
<img src="https://github.com/siyuancncd/CoPINN/blob/main/CoPINN_results.png" width="800" height="700">
</p>

## Requirements

Follow [Separable PINN](https://github.com/stnamjef/SPINN).

## Train and test

```
# Run 3D-helmholtz

python helmholtz3d.py

# Run 3D-klein_gordon

python klein_gordon3d.py

# Run 4D-klein_gordon

python klein_gordon4d.py

# Run 3D-diffusion 

python diffusion3d.py

```

## Citation

```
@inproceedings{duancopinn,
  title={CoPINN: Cognitive Physics-Informed Neural Networks},
  author={Duan, Siyuan and Wu, Wenyuan and Hu, Peng and Ren, Zhenwen and Peng, Dezhong and Sun, Yuan},
  booktitle={Forty-second International Conference on Machine Learning},
  year={2025}
}
```

```
@article{WU2025106750,
title = {Deep fuzzy physics-informed neural networks for forward and inverse PDE problems},
journal = {Neural Networks},
volume = {181},
pages = {106750},
year = {2025},
issn = {0893-6080},
author = {Wenyuan Wu and Siyuan Duan and Yuan Sun and Yang Yu and Dong Liu and Dezhong Peng},
}
```

## Question?

If you have any questions, please email:
* sunyuan_work@163.com (Yuan Sun);
* siyuanduancn@gmail.com (Siyuan Duan);
* wuwenyuan97@gmail.com (Wenyuan Wu).

## Acknowledgement

The code is inspired by [Separable PINN](https://github.com/stnamjef/SPINN).
