# Disentangled Graph Collaborative Filtering
This is our Tensorflow implementation for the paper:

>Xiang Wang, Hongye Jin, An Zhang, Xiangnan He, Tong Xu, and Tat-Seng Chua (2020). Disentangled Graph Collaborative Filtering, [Paper in arXiv](https://arxiv.org/abs/2007.01764). In SIGIR'20, Xi'an, China, July 25-30, 2020.

Author: Dr. Xiang Wang (xiangwang at u.nus.edu)

## Introduction
Disentangled Graph Collaborative Filtering (DGCF) is an explainable recommendation framework, which is equipped with (1) dynamic routing mechanism of capsule networks, to refine the strengths of user-item interactions in intent-aware graphs, (2) embedding propagation mechanism of graph neural networks, to distill the pertinent information from higher-order connectivity, and (3) distance correlation of independence modeling, to ensure the independence among intents. As such, we explicitly disentangle the hidden intents of users in the representation learning.

## Citation 
If you want to use our codes and datasets in your research, please cite:
```
@inproceedings{DGCF19,
  author    = {Xiang Wang and
               Hongye Jin and
               An Zhang and
               Xiangnan He and
               Tong Xu and
               Tat{-}Seng Chua},
  title     = {Disentangled Graph Collaborative Filtering},
  booktitle = {Proceedings of the 43nd International {ACM} {SIGIR} Conference on
               Research and Development in Information Retrieval, {SIGIR} 2020, Xi'an,
               China, July 25-30, 2020.},
  year      = {2020},
}
```
## Environment Requirement
We recommend to run this code in GPUs. The code has been tested running under Python 3.6.5. The required packages are as follows:
* tensorflow_gpu == 1.14.0
* numpy == 1.14.3
* scipy == 1.1.0
* sklearn == 0.19.1

## Versions
We released the implementation based on the NGCF code as DGCF_v1. Later, we will release another implementation based on the LightGCN code as DGCF_v2, which is equipped with some speedup techniques.

## Example to Run the Codes
The instruction of commands has been clearly stated in the codes (see the parser function in DGCF/utility/parser.py).
* Gowalla dataset
```
CUDA_VISIBLE_DEVICES=0 python GDCF.py --dataset gowalla --batch_size 2000 --n_layers 1 --n_iterations 2 --corDecay 0.01 --n_factors 4 --show_step 3 --lr 0.001 
```

Some important arguments (additional to that of NGCF):
* `cor_flag`
  * It specifies whether the distance correlation (i.e., independence modeling) is activated..
  * Here we provide two options:
    * 1 (by default), which activates the distance correlation in [Disentangled Graph Collaborative Filtering](https://arxiv.org/abs/2007.01764), SIGIR2020. Usage: `--cor_flag 1`.
    * 0, which disables the distance correlation. Usage: `--cor_flag 0`.

* `corDecay`
  * It specifies the weight to control the distance correlation.
  * Here we provide four options:
    * 0.0 (by default), which similarly disables the distance correlation and makes DGCF rely only on the dynamic routing mechanism to disentangle the user intents. Usage: `--corDecay 0.0`.
    * other scales like 0.1, which uses 0.1 to control the strengths of distance correlation. Usage: `--corDecay 0.1`.

* `n_factors`
  * It indicates the number of latent intents to disentangle the holistic representation into chunked intent-aware representations. Usage: `--n_factors 4`.
  * Note that the arguement `embed_size` needs to be exactly divisible by the arguement `n_factors`.

* `n_iterations`
  * It indicates the number of iterations to perform the dynamic routing mechanism. Usage `--n_iterations 2`.

## Dataset
Following our prior work NGCF and LightGCN, We provide three processed datasets: Gowalla, Amazon-book, and Yelp2018.
Note that the Yelp2018 dataset used in DGCF is slightly different from the original in NGCF, since we found some bugs in the preprocessing code to construct the Yelp2018 dataset. We rerun the experiments and report the performance in the corrected dataset.

## Acknowledgement

This research is supported by the National Research Foundation, Singapore under its International Research Centres in Singapore Funding Initiative. Any opinions, findings and conclusions or recommendations expressed in this material are those of the author(s) and do not reflect the views of National Research Foundation, Singapore.
