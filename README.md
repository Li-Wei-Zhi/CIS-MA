# Semantics-Empowered Non-orthogonal Multiple  Access for Downlink Transmission of Correlated  Information Source

Abstract: _In this paper, we introduce a non-orthogonal multiple access (NOMA) framework for the downlink transmission of correlated information sources in the multi-user scenario, in which the data required or transmitted by multiple users share similar content. To enhance the end-to-end transmission performance, we resort to the semantic communication paradigm and build our system based on the deep joint sourcechannel coding (D-JSCC) scheme. Inspired by Wyner’s common information, an information theoretical concept, the common information (CI) extraction is proposed to effectively capture the correlation between multiple users. By relaxing the constraint of the object function, equivalency can be established between common information extraction and mutual information maximization. Thereby, the Jenson-Shannon divergence (JSD) is adopted in the loss function for learning the common information representation (CIR). In order to categorize the theoretical performance limit of the proposed system, semantic synonymous mapping (SSM) based information theory is applied for analyzing the effect of correlation level and different decoding schemes on the achievable channel capacity. Specifically, the analytical expression of channel capacity under additive white Gaussian noise (AWGN) and Rayleigh channel is derived and verified by Monte-Carlo experiments. By conducting simulations on three different image datasets, it is verified that our proposed scheme can outperform a series of other state-of-the-art (SoTA) multiple access or distributed source coding (DSC) schemes under up to seven-user scenarios. The visualization and ablation study results validate the effectiveness of the common information extraction._

# Dataset preparation 

This paper include three datasets MNIST [Page](http://yann.lecun.com/exdb/mnist/), Wildtrack [Page](https://www.epfl.ch/labs/cvlab/data/data-wildtrack/) and MultiViewC [Page](https://github.com/Jiahao-Ma/MultiviewC)

You should download these datasets and specify their paths in the config files. 
## Requirements

You can install the following combined environment.

- CUDA=11.6

- Python 3.8
1. Create environment:

```
conda create -n CISMA python=3.8
conda activate CISMA
pip install -r requirements
```
## Train

Take the MNIST dataset as example
```
# First stage training
python train.py 
--config configs/config_stage_one.yaml
--device your_gpu_id
```

Afterwards, specify the path of the saved model file in the config/config_stage_two.yaml file. Then, run
```
# Second stage training
python train.py 
--config configs/config_stage_two.yaml
--device your_gpu_id
```

## Test

```
# Testing
python train.py 
--config configs/config_test.yaml
--device your_gpu_id
```
