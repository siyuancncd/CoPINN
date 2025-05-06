# CoPINN

Siyuan Duan*, Wenyuan Wu*, Peng Hu, Zhenwen Ren, Dezhong Peng, and Yuan Sun. "CoPINN: Cognitive Physics-Informed Neural Network". (ICML 2025, Spotlight (acc rate = 2.6%), JAX Code)

## 

:bangbang: **I’m actively seeking a PhD position for Fall 2026 entry.** If you believe my background aligns with your research needs, please feel free to contact me via email at siyuanduancn@gmail.com.

## Abstract
Physics-informed neural networks (PINNs) aim to constrain the outputs and gradients of deep learning models to satisfy specified governing physics equations, which have demonstrated significant potential for solving partial differential equations (PDEs). Although existing PINN methods have achieved pleasing performance, they always treat both easy and hard sample points indiscriminately, especially ones in the physical boundaries. This easily causes the PINN model to fall into undesirable local minima and unstable learning, thereby resulting in an Unbalanced Prediction Problem (UPP). To deal with this daunting problem, we propose a novel framework named Cognitive Physics-Informed Neural Network (**CoPINN**) that imitates the human cognitive learning manner from easy to hard. Specifically, we first employ separable subnetworks to encode independent one-dimensional coordinates and apply an aggregation scheme to generate multi-dimensional predicted physical variables. Then, during the training phase, we dynamically evaluate the difficulty of each sample according to the gradient of the PDE residuals. Finally, we propose a cognitive training scheduler to progressively optimize the entire sampling regions from easy to hard, thereby embracing robustness and generalization against predicting physical boundary regions. Extensive experiments demonstrate that our CoPINN achieves state-of-the-art performance, particularly significantly reducing prediction errors in stubborn regions. 

## Motivation

<p align="center">
<img src="https://github.com/siyuancncd/CoPINN/blob/main/CoPINN_motivation.png" width="440" height="240">
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

## Train and test

The code is coming soon...

## Citation

coming soon...

## Question?

If you have any questions, please email ddzz12277315 AT 163 DOT com or siyuanduancn AT gmail DOT com.

## Acknowledgement

The code is inspired by [Separable PINN](https://github.com/stnamjef/SPINN).
