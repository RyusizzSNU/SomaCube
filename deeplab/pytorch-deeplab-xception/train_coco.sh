CUDA_VISIBLE_DEVICES=0,1 python train.py --backbone resnet --lr 0.01 --workers 1 --epochs 40 --batch-size 2 --gpu-ids 0,1 --checkname deeplab-resnet --eval-interval 1 --dataset coco
