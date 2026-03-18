#!/bin/bash
# Task: Online test-time adaptation with CLIP model
# Dataset: CIFAR10, CIFAR100, Food101, ImageNet-1k, StanfordCars, RSICD, EuroSAT, RESISC45, PatternNet, MLRSNet
# Model: CLIP

# ViT-B/32
# python -m torch.distributed.launch --nproc_per_node 1 --master_port 19345 online_clip.py --model ViT-B/32 --dataset cifar10 --split test --iters 1 --save_model --save_path ./results/online_clip/vit-b32/cifar10_test/
# python -m torch.distributed.launch --nproc_per_node 1 --master_port 19346 online_clip.py --model ViT-B/32 --dataset cifar100 --split test --iters 1 --save_model --save_path ./results/online_clip/vit-b32/cifar100_test/
# python -m torch.distributed.launch --nproc_per_node 1 --master_port 19347 online_clip.py --model ViT-B/32 --dataset food101 --split validation --iters 1 --save_model --save_path ./results/online_clip/vit-b32/food101_val/
# python -m torch.distributed.launch --nproc_per_node 1 --master_port 19372 online_clip.py --model ViT-B/32 --dataset imagenet-1k --split validation --iters 1 --save_model --save_path ./results/online_clip/vit-b32/imagenet-1k_val/
# python -m torch.distributed.launch --nproc_per_node 1 --master_port 19381 online_clip.py --model ViT-B/32 --dataset Multimodal-Fatima_StanfordCars_test --split test --iters 1 --save_model --save_path ./results/online_clip/vit-b32/stanfordcars_test/
python -m torch.distributed.launch --nproc_per_node 1 --master_port 19387 online_clip.py --model ViT-B/32 --dataset rsicd --split test --iters 1 --save_model --save_path ./results/online_clip/vit-b32/rsicd_test/
python -m torch.distributed.launch --nproc_per_node 1 --master_port 19388 online_clip.py --model ViT-B/32 --dataset eurosat --split test --iters 1 --save_model --save_path ./results/online_clip/vit-b32/eurosat_test/
python -m torch.distributed.launch --nproc_per_node 1 --master_port 19389 online_clip.py --model ViT-B/32 --dataset resisc45 --split test --iters 1 --save_model --save_path ./results/online_clip/vit-b32/resisc45_test/
# python -m torch.distributed.launch --nproc_per_node 1 --master_port 19390 online_clip.py --model ViT-B/32 --dataset patternnet --split train --iters 1 --save_model --save_path ./results/online_clip/vit-b32/patternnet_train/
# python -m torch.distributed.launch --nproc_per_node 1 --master_port 19391 online_clip.py --model ViT-B/32 --dataset mlrsnet --split train --iters 1 --save_model --save_path ./results/online_clip/vit-b32/mlrsnet_train/

# ViT-B/16
# python -m torch.distributed.launch --nproc_per_node 1 --master_port 19348 online_clip.py --model ViT-B/16 --dataset cifar10 --split test --iters 1 --save_model --save_path ./results/online_clip/vit-b16/cifar10_test/
# python -m torch.distributed.launch --nproc_per_node 1 --master_port 19349 online_clip.py --model ViT-B/16 --dataset cifar100 --split test --iters 1 --save_model --save_path ./results/online_clip/vit-b16/cifar100_test/
# python -m torch.distributed.launch --nproc_per_node 1 --master_port 19350 online_clip.py --model ViT-B/16 --dataset food101 --split validation --iters 1 --save_model --save_path ./results/online_clip/vit-b16/food101_val/
# python -m torch.distributed.launch --nproc_per_node 1 --master_port 19373 online_clip.py --model ViT-B/16 --dataset imagenet-1k --split validation --iters 1 --save_model --save_path ./results/online_clip/vit-b16/imagenet-1k_val/
# python -m torch.distributed.launch --nproc_per_node 1 --master_port 19382 online_clip.py --model ViT-B/16 --dataset Multimodal-Fatima_StanfordCars_test --split test --iters 1 --save_model --save_path ./results/online_clip/vit-b16/stanfordcars_test/
python -m torch.distributed.launch --nproc_per_node 1 --master_port 19392 online_clip.py --model ViT-B/16 --dataset rsicd --split test --iters 1 --save_model --save_path ./results/online_clip/vit-b16/rsicd_test/
python -m torch.distributed.launch --nproc_per_node 1 --master_port 19393 online_clip.py --model ViT-B/16 --dataset eurosat --split test --iters 1 --save_model --save_path ./results/online_clip/vit-b16/eurosat_test/
python -m torch.distributed.launch --nproc_per_node 1 --master_port 19394 online_clip.py --model ViT-B/16 --dataset resisc45 --split test --iters 1 --save_model --save_path ./results/online_clip/vit-b16/resisc45_test/
# python -m torch.distributed.launch --nproc_per_node 1 --master_port 19395 online_clip.py --model ViT-B/16 --dataset patternnet --split train --iters 1 --save_model --save_path ./results/online_clip/vit-b16/patternnet_train/
# python -m torch.distributed.launch --nproc_per_node 1 --master_port 19396 online_clip.py --model ViT-B/16 --dataset mlrsnet --split train --iters 1 --save_model --save_path ./results/online_clip/vit-b16/mlrsnet_train/

# RN50
# python -m torch.distributed.launch --nproc_per_node 1 --master_port 19357 online_clip.py --model RN50 --dataset cifar10 --split test --iters 1 --save_model --save_path ./results/online_clip/rn50//cifar10_test/
# python -m torch.distributed.launch --nproc_per_node 1 --master_port 19358 online_clip.py --model RN50 --dataset cifar100 --split test --iters 1 --save_model --save_path ./results/online_clip/rn50//cifar100_test/
# python -m torch.distributed.launch --nproc_per_node 1 --master_port 19359 online_clip.py --model RN50 --dataset food101 --split validation --iters 1 --save_model --save_path ./results/online_clip/rn50//food101_val/
# python -m torch.distributed.launch --nproc_per_node 1 --master_port 19376 online_clip.py --model RN50 --dataset imagenet-1k --split validation --iters 1 --save_model --save_path ./results/online_clip/rn50//imagenet-1k_val/
# python -m torch.distributed.launch --nproc_per_node 1 --master_port 19385 online_clip.py --model RN50 --dataset Multimodal-Fatima_StanfordCars_test --split test --iters 1 --save_model --save_path ./results/online_clip/rn50//stanfordcars_test/
python -m torch.distributed.launch --nproc_per_node 1 --master_port 19397 online_clip.py --model RN50 --dataset rsicd --split test --iters 1 --save_model --save_path ./results/online_clip/rn50/rsicd_test/
python -m torch.distributed.launch --nproc_per_node 1 --master_port 19398 online_clip.py --model RN50 --dataset eurosat --split test --iters 1 --save_model --save_path ./results/online_clip/rn50/eurosat_test/
python -m torch.distributed.launch --nproc_per_node 1 --master_port 19399 online_clip.py --model RN50 --dataset resisc45 --split test --iters 1 --save_model --save_path ./results/online_clip/rn50/resisc45_test/
# python -m torch.distributed.launch --nproc_per_node 1 --master_port 19400 online_clip.py --model RN50 --dataset patternnet --split train --iters 1 --save_model --save_path ./results/online_clip/rn50/patternnet_train/
# python -m torch.distributed.launch --nproc_per_node 1 --master_port 19401 online_clip.py --model RN50 --dataset mlrsnet --split train --iters 1 --save_model --save_path ./results/online_clip/rn50/mlrsnet_train/

# RN101
# python -m torch.distributed.launch --nproc_per_node 1 --master_port 19360 online_clip.py --model RN101 --dataset cifar10 --split test --iters 1 --save_model --save_path ./results/online_clip/rn101/cifar10_test/
# python -m torch.distributed.launch --nproc_per_node 1 --master_port 19361 online_clip.py --model RN101 --dataset cifar100 --split test --iters 1 --save_model --save_path ./results/online_clip/rn101/cifar100_test/
# python -m torch.distributed.launch --nproc_per_node 1 --master_port 19362 online_clip.py --model RN101 --dataset food101 --split validation --iters 1 --save_model --save_path ./results/online_clip/rn101/food101_val/
# python -m torch.distributed.launch --nproc_per_node 1 --master_port 19377 online_clip.py --model RN101 --dataset imagenet-1k --split validation --iters 1 --save_model --save_path ./results/online_clip/rn101/imagenet-1k_val/
# python -m torch.distributed.launch --nproc_per_node 1 --master_port 19386 online_clip.py --model RN101 --dataset Multimodal-Fatima_StanfordCars_test --split test --iters 1 --save_model --save_path ./results/online_clip/rn101/stanfordcars_test/
python -m torch.distributed.launch --nproc_per_node 1 --master_port 19402 online_clip.py --model RN101 --dataset rsicd --split test --iters 1 --save_model --save_path ./results/online_clip/rn101/rsicd_test/
python -m torch.distributed.launch --nproc_per_node 1 --master_port 19403 online_clip.py --model RN101 --dataset eurosat --split test --iters 1 --save_model --save_path ./results/online_clip/rn101/eurosat_test/
python -m torch.distributed.launch --nproc_per_node 1 --master_port 19404 online_clip.py --model RN101 --dataset resisc45 --split test --iters 1 --save_model --save_path ./results/online_clip/rn101/resisc45_test/
# python -m torch.distributed.launch --nproc_per_node 1 --master_port 19405 online_clip.py --model RN101 --dataset patternnet --split train --iters 1 --save_model --save_path ./results/online_clip/rn101/patternnet_train/
# python -m torch.distributed.launch --nproc_per_node 1 --master_port 19406 online_clip.py --model RN101 --dataset mlrsnet --split train --iters 1 --save_model --save_path ./results/online_clip/rn101/mlrsnet_train/
