# CoPINN

Siyuan Duan*, Wenyuan Wu*, Peng Hu, Zhenwen Ren, Dezhong Peng, and Yuan Sun. "CoPINN: Cognitive Physics-informed Neural Network". (ICML 2025, JAX Code)

## 

:bangbang: **I’m actively seeking a PhD position for Fall 2026 entry.** If you believe my background aligns with your research needs, please feel free to contact me via email at siyuanduancn@gmail.com.

## Abstract
Physics-informed neural networks (PINNs) aim to constrain the outputs and gradients of deep learning models to satisfy specified governing physics equations, which have demonstrated significant potential for solving partial differential equations (PDEs). Although existing PINN methods have achieved pleasing performance, they always treat both easy and hard sample points indiscriminately, especially ones in the physical boundaries. This easily causes the PINN model to fall into undesirable local minima and unstable learning, thereby resulting in an Unbalanced Prediction Problem (UPP). To deal with this daunting problem, we propose a novel framework named Cognitive Physical Informed Neural Network (**CoPINN**) that imitates the human cognitive learning manner from easy to hard. Specifically, we first employ separable subnetworks to encode independent one-dimensional coordinates and apply an aggregation scheme to generate multi-dimensional predicted physical variables. Then, during the training phase, we dynamically evaluate the difficulty of each sample according to the gradient of the PDE residuals. Finally, we propose a cognitive training scheduler to progressively optimize the entire sampling regions from easy to hard, thereby embracing robustness and generalization against predicting physical boundary regions. Extensive experiments demonstrate that our CoPINN achieves state-of-the-art performance, particularly significantly reducing prediction errors in stubborn regions. 

## Motivation

## Framework

## Experimental Results

## Requirements

## Train and test

The code is coming soon...

## Citation

coming soon...

## Question?

If you have any questions, please email ddzz12277315 AT 163 DOT com or siyuanduancn AT gmail DOT com.

## Acknowledgement

The code is inspired by [Separable PINN](https://github.com/stnamjef/SPINN).
