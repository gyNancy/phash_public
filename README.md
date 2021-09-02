# It's Not What It Looks Like: Manipulating Perceptual Hashing based Applications
The repository contains the dataset and major code used for attacking perceptual hashing based image search. This
is done by generating attack images that effectively enlarge the hash
distance to the original image while introducing minimal visual changes under the black-box setting. Visual distances could be l2 distances or any perceptual distances (i.e., https://github.com/richzhang/PerceptualSimilarity/) as we used in our attacks. Our attacks focus on subverting standard pHash algorithms referred from https://github.com/JohannesBuchner/imagehash and could also be used to attack the more robust Blockhash algorithms https://github.com/commonsmachinery/blockhash-python. 

Further details can be found in our paper "It's Not What It Looks Like: Manipulating Perceptual Hashing based Applications" by Qingying Hao, Licheng Luo, Steve T.K. Jan, Gang Wang (CCS 2021). (https://qingyinghao.web.illinois.edu/files/ccs21_pHash_preprint.pdf)


## 1. Installation

Python 3.6.8 and Tensorflow 1.14.0 are used. Other tensorflow versions >=1.13.0 should work but our code is not compatible with Tensorflow >=2.0. The full environment file could be found in requirement.txt as a reference. 

## 2. Examples
You could define a number of command line arugments to run different attacks for different datasets. You could refer to test_attack_black.py for more information. We give examples to run basic and advanced pHash attacks for selected ImageNet dataset. The other datasets we use are in Face_results and IMD_results. 

2.1 Basic Attack

The basic attack has been split into grayscale initialization and RGB attack pipelines to improve the attack efficiency. In the following example, we aim at generating attacks images that differ 10 bits (out of 64 bits hash) with the original images for phash attacks. You could also look at the test.sh file in ImageNet_results folder. 
```bash
Grayscale attack: 
python3 test_attack_black.py --untargeted -a black -d imagenet --reset_adam -n 50 --solver adam -b 2 -p 1 --hash 10 --use_resize --method "tanh" --batch 256 --gpu 0 --lr 0.01 -s "black_results_imagenet" --start_idx=0 --dist_metrics "pdist" --save_ckpts "best_modifier_imagenet"

RGB attack:
python3 test_attack_black_rgb.py --untargeted -a black -d imagenet --reset_adam -n 50 --solver adam -b 2 -p 1 --hash 10 --use_resize --method "tanh" --batch 256 --gpu 2 --lr 0.01 -s "black_results_imagenet_rgb" --start_idx=0 --dist_metrics "pdist" --save_ckpts "best_modifier_imagenet"
```
2.2 Advanced Attack

In order to improve the robustness of our basic attacks, we introduce different techniques for advanced attacks. AoE attacks generates the attack image that differs x hash bits (e.g., 10 bits) with not only the original images but also slightly transformed original images(s). You could also define your own choices of slightly modified originial images. AoT attacks generate the attack image based on the slightly transformed image. 

2.2.1 Attack over Input Ensemble (AoE) 

This example uses three versions of transformed original images, including images that have been central cropped, rotated and Disproportionate scaled.  
```bash
python3 test_attack_black_aoe.py --untargeted -a black -d imagenet --reset_adam -n 50 --solver adam -b 2 -p 1 --hash 10 --use_resize --method "tanh" --batch 512 --gpu 1 --lr 0.01 -s "black_results_imagenet_eot" --start_idx=0 --dist_metrics "pdist" --save_ckpts "best_modifier_imagenet_eot"
```
2.2.2 Attack over Input Transformation (AoT) 

This example creates the attack image based on the central cropped original image. 
```bash
python3 test_attack_black_aot.py --untargeted -a black -d imagenet --reset_adam -n 50 --solver adam -b 2 -p 1 --hash 10 --use_resize --method "tanh" --batch 512 --gpu 1 --lr 0.01 -s "black_results_imagenet_aot" --start_idx=0 --dist_metrics "pdist" --save_ckpts "best_modifier_imagenet_aot" --transform "crop"
```

## 3. Citation
If you find this repo useful for your research, please use the following.
```
@inproceedings {hao-ccs21,
author = {Qingying Hao and Licheng Luo and Steve TK Jan and Gang Wang},
title = {It's Not What It Looks Like: Manipulating Perceptual Hashing based Applications},
booktitle = {Proceedings of The ACM Conference on Computer and Communications Security (CCS)},
year = {2021},
}
```
If you have any questions, please contact Qingying Hao (qhao2@illinois.edu). 