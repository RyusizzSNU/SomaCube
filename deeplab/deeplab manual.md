# 사용방법

##### by befreor@gmail.com in SNU

## 0. environment 구성

> deeplab 코드는 다음의 github을 참고하였고 environment 구성은 anaconda로 설명되어 있는 그대로 구성하였음
>
> https://github.com/jfzhang95/pytorch-deeplab-xception?source=post_page--------------------------- 
>
> 단, pycocotools는 다음 링크를 참조하여 깔아야 문제 없이 python에서 pycocotools를 사용할 수 있을 것
>
> https://github.com/philferriere/cocoapi

## 1. code 실행

- for training: 'pytorch-deeplab-xception' 에 들어가서 train_coco.sh 실행

- for valdating: 'pytorch-deeplab-xception' 에 들어가서 val_coco.sh 실행

  >  run 디렉토리의 최신 experiment 폴더에 저장될 것임
  >
  > training 시에 조금 시간이 지난 후에 model parameter saving이 됨
  >
  > tensorboard 켜서 groundtruth, prediction 확인 가능

## 2. custom dataset configuration

- COCO dataset 형식에 따라 dataset을 만들었음

  > train data 500개, val data 50개로 구성handai_annotations_[train/val].json 에 annotation 데이터 저장handai_dataset_img_[train/val] 에 img 데이터 저장

- cube dataset 은 dataset 디렉토리에 있음

> handai_annotations_train.json: annotation json file for training
>
> handai_annotations_val.json: annotation json file for validation
>
> handai_dataset_img_train: img files for training
>
> handai_dataset_img_val: img files for validation

## 3. Dataset Preprocessing 관련

- *'pytorch-deeplab-xception/datasloaders/datasets/coco.py'* 코드를 읽어보면 됨

- *line 16*: NUM_CLASSES = background + cube 7 종류 = 8 카테고리

- *line 18*: 기존에는 COCO dataset에서 사용할 카테고리 index 를 선택적을 넣었으나, 우리는 0~7 모두를 쓸 것이므로 0~7 모두 기입

- *line 31*: annotation specification json file 경로인데, split 에 따라 다른 파일 참조

  > dataset 디렉토리에 보면, 'handai_dataset_ann_train' 과 'handai_dataset_ann_val'가 그것임

- *line32*: handai_ann_ids 경로인데, deeplab 에서 필요한 파일인거 같음

  > train_coco.sh 나 val_coco.sh 돌리면 handai_ann_ids_[train|val].pth 가 만들어지는데, 해당 파일이 없을때 처음에만 processing 되므로, dataset에 image나 annotation을 추가 혹은 변경하였다면 handai_ann_ids[train|val].pth 를 삭제하고 코드를 실행시킬 것

- *line 35*: handai_dataset_img 경로이고 마찬가지로 split에 다라 다른 파일 참조

  > dataset 디렉토리에 보면, 'handai_dataset_img_train' 과 'handai_dataset_img_val'이 그것임

## 4. trained model

- val_coco.sh의 *—resume* 인자에 적혀있는 'run/coco/deeplab-resnet/experiment_1/checkpoint.pth.tar'가 trained model임
- val.py 는 validation img들을 masked_imgs 디렉토리에 저장하고 iou score를 iou.csv 파일에 저장
- *line 173* ~ *line 192* : iou, masked img 저장 코드