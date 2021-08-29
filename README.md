# It's Not What It Looks Like: Manipulating Perceptual Hashing based Applications
The repository contains the dataset and major code used for attacking perceptual hashing based image search. Our attacks focus on subverting standard perceptual hashing algorithms referred from https://github.com/JohannesBuchner/imagehash and could also be used to attack the more robust Blockhash https://github.com/commonsmachinery/blockhash-python. 

Further details can be found in our paper "It's Not What It Looks Like: Manipulating Perceptual Hashing based Applications" by Qingying Hao, Licheng Luo, Steve T.K. Jan, Gang Wang (CCS 2021). (https://qingyinghao.web.illinois.edu/files/ccs21_pHash_preprint.pdf)


## 1. Installation

Python 3.6.8 and Tensorflow 1.14.0 are used. Other tensorflow versions >=1.13.0 should work but our repo is not compatible with Tensorflow >=2.0. The full environment file could be found in requirement.txt as a reference. 

## 2. Usage

## 3. Examples
3.1 Basic Attack 
```bash
Greyscale attack: 
python3 test_attack_black.py --untargeted -a black -d imagenet --reset_adam -n 50 --solver adam -b 2 -p 1 --hash 10 --use_resize --method "tanh" --batch 256 --gpu 0 --lr 0.01 -s "black_results_imagenet" --start_idx=0 --dist_metrics "pdist" --save_ckpts "best_modifier_imagenet"

RGB attack:
python3 test_attack_black_rgb.py --untargeted -a black -d imagenet --reset_adam -n 50 --solver adam -b 2 -p 1 --hash 10 --use_resize --method "tanh" --batch 256 --gpu 2 --lr 0.01 -s "black_results_imagenet_rgb" --start_idx=0 --dist_metrics "pdist" --save_ckpts "best_modifier_imagenet"
```
3.2 Advanced Attack

3.2.1 Attack over Input Ensemble (AoE) 
```bash
python3 test_attack_black_aoe.py --untargeted -a black -d imagenet --reset_adam -n 50 --solver adam -b 2 -p 1 --hash 20 --use_resize --method "tanh" --batch 512 --gpu 1 --lr 0.01 -s "black_results_imagenet_eot" --start_idx=0 --dist_metrics "pdist" --save_ckpts "best_modifier_imagenet_eot"
```
3.2.3 Attack over Input Transformation (AoT) 
    using cropping as the example
```bash
python3 test_attack_black_aot.py --untargeted -a black -d imagenet --reset_adam -n 50 --solver adam -b 2 -p 1 --hash 20 --use_resize --method "tanh" --batch 512 --gpu 1 --lr 0.01 -s "black_results_imagenet_aot" --start_idx=0 --dist_metrics "pdist" --save_ckpts "best_modifier_imagenet_aot" --transform "crop"
```
