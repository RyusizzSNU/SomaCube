annotation
================

- cocohandai550/, data4cv/: 디렉토리는 딥러닝 학습용, data4cv는 computer vision 코드의 검증용으로 사용. 세부적인 레이블링의 기준이 다를 뿐 같은 코드를 사용.

- cocohandai/annotations_train.json: training annotation 파일 (레이블)

- cocohandai/annotations_train.json: validation annotation 파일 (레이블)

- cocohandai550/train: training 이미지

- cocohandai550/val: validation 이미지

- cocohandai550/ann_train: training 이미지에서 이미지들의 annotation 조각(json파일)들의 모음

- cocohandai550/ann_val: validation 이미지에서 이미지들의 annotation 조각(json파일)들의 모음

- bbseg.py: 이미지의 annotation(segementation, bounding box)를 만들어 ann_train, ann_val에 저장함

- collect.py: 사진기와 pc를 연결해 사진을 찍고 저장함

- merge_ann_train.py: 이미지들의 annotation 조각(train)을 합침

- merge_ann_val.py: 이미지들의 annotation 조각(val)을 합침



