# 사용자가 사진 데이터가 있는 디렉토리에서 bounding box와 segmentation 데이터를 레이블링해
# Coco anotation 데이터의 json list로 저장하는 프로그램

import argparse
import os
import os.path as osp
import pygame
import json
import time
from pycocotools.coco import COCO

# Soma Cube의 색깔을 정의함
colors = [
  pygame.Color("red"),
  pygame.Color("blue"),
  pygame.Color("brown"),
  pygame.Color("green"),
  pygame.Color("purple"),
  pygame.Color("orange"),
  pygame.Color("yellow"),
]

# 각 Cube의 category id
cat_map = {
  "red": 1,
  "blue": 2,
  "brown": 3,
  "green": 4,
  "purple": 5,
  "orange": 6,
  "yellow": 7
}

'''
  "annotations": [
    {
      "id": 1,
      "category_id": 1,
      "iscrowd": 0,
      "segmentation": [[0.0, 0.0, 100.0, 0.0, 100.0, 100.0, 0.0, 100.0]],
      "image_id": 1,
      "bbox": [0.0, 0.0, 100.0, 100.0]
    }
  ]
'''

#python3
# 정렬된 image 파일 이름과 "annFile"이라는 coco 파일 상의 id를 불러옴
def get_sorted_img_id_list(annFile):
  with open(annFile, "r") as f:
    raw = json.load(f)

  return [(x["file_name"], x["id"]) for x in sorted(raw["images"], key=lambda x: x["id"])]

try:
  input = raw_input
except NameError:
  pass

# 사용자에게 정보를 프린트함
def print_info():
  print(
'''
========================================
[A]: new category annotation (exit/save current annotation)
[S]: next point group for current annotation
[R]: redo current image
[N]: next image (exit/save current annotation)
[Q]: quit unexpectedly (not recommended)
========================================
''')


def main():
  prs = argparse.ArgumentParser()
  prs.add_argument("-d", "--dataset_path", type=str, choices=['train', 'val'])
  prs.add_argument("-a", "--annotation", type=str, choices=['annotations_train.json', 'annotations_val.json'], default='annotations_train.json')
  prs.add_argument("-o", "--output_path", type=str, choices=['ann_train', 'ann_val'], default='ann_train')
  prs.add_argument("-n", "--output_name", type=str, default='out.json')
  prs.add_argument("-s", "--start_index", type=int, default=0)
  prs.add_argument("-e", "--end_index", type=int, default=500)
  prs.add_argument("-c", "--single_category", type=int, default=-1)
  '''
  [실행 예]
  python3 bbseg.py -d train -a annotation_train.json -o ann_train -n out.json -s 0 -e 20
  => 0 부터 20번째 training 데이터를 out.json 데이터로 저장함
  python3 bbseg.py -d val -a annotation_val.json -o ann_val -n out_block1.json -s 20 -e 40 -c 1
  => 20 부터 40번째 validation 데이터를 1번 카테고리만 레이블링하고 out_block1.json데이터로 저장함
  '''


  # argmuent를 파싱하고 coco 모듈, 이미지 리스트를 불러옴
  args = prs.parse_args()

  assert ((args.dataset_path in args.annotation) and (args.dataset_path in args.output_path))


  os.makedirs(args.output_path, exist_ok=True)

  coco=COCO(args.annotation)

  dataset_path = args.dataset_path

  img_id_list = get_sorted_img_id_list(args.annotation)[args.start_index:args.end_index]

  n = len(img_id_list)
  i = 0

  assert n > 0
  pygame.init()
  pic = pygame.image.load(osp.join(dataset_path, img_id_list[0][0]))
  pygame.display.set_mode(pic.get_size())
  screen = pygame.display.get_surface()

  done = False

  ann_id = 1
  cat_id = None
  annotations = []

  # 정해진 범위에 이미지들이 끝날 때 까지 루프
  while i < n and not done:
    file_name = img_id_list[i][0]
    file_id = img_id_list[i][1]
    file_full_name = osp.join(dataset_path,file_name)
    # 이미지를 로드함
    pic = pygame.image.load(file_full_name)
    screen.blit(pic, (0, 0))
    pygame.display.flip()


    img_done = False
    # 한 이미지의 레이블링이 끝날 때 까지 루프
    while not(img_done):
      #print("1")
      time.sleep(0.1)

      # single category의 경우 category 고정. 나머지는 손으로 cat_id를 입력함
      if args.single_category > 0:
        cat_id = args.single_category
      else:
        cat_id = input("Catergory (1~7/color)? ")
        if cat_id in '1234567':
          cat_id = int(cat_id)
        elif cat_id in cat_map.keys():
          cat_id = cat_map[cat_id]
        elif cat_id == 'n':
          img_done = True
          i += 1
          continue
        else:
          continue

      ann_done = False

      # 하나의 annotation 정보 (비어있음)
      ann = {
        "id": ann_id,
        "category_id": cat_id,
        "iscrowd": 0,
        "segmentation": [],
        "image_id": file_id,
        "bbox": []
      }


      n_pt = 0
      seg = [[]]
      # 하나의 annotation이 끝날 때 까지 루프
      while not(ann_done):
        #print("2")

        key_down = False
        print_info()

        # 계속해서 key를 받아 작업을 수행함
        '''
        [A]: 현재 cat_id 의 annotation을 종료하고 새로운 annotation 시작
        [S]: 현재 annotation 중 다음 segmentation group (overlapping 이 있을 경우)
        [R]: 현재 이미지 재시작
        [N]: 다음 이미지 (현재 이미지 작업이 끝남)
        [Q]: 프로그램 종료
        '''
        while not(key_down):
          #print("3")

          time.sleep(0.01)
          for event in pygame.event.get():
            if event.type == pygame.QUIT:
              key_down = True
              key = "q"
            elif event.type == pygame.KEYDOWN:
              #print(event)
              if event.key == pygame.K_a:
                key = "a"
                key_down = True
              elif event.key == pygame.K_s:
                key = "s"
                key_down = True
              elif event.key == pygame.K_r:
                key = "r"
                key_down = True
              elif event.key == pygame.K_n:
                key = "n"
                key_down = True
              elif event.key == pygame.K_q:
                key = "q"
                key_down = True
            elif event.type == pygame.MOUSEBUTTONDOWN:
              pos = pygame.mouse.get_pos()
              pos = [float(pos[0]), float(pos[1])]
              n_pt += 1
              if n_pt >= 2:
                pygame.draw.line(screen, colors[cat_id-1], prev_pos, pos, 1)
              pygame.display.flip()
              prev_pos = pos
              seg[-1].extend(pos)

        if key in 'anr':
          ann_done = True

        if key == 's':
          seg.append([])
          n_pt = 0

        if key == 'n':
          img_done = True
          i += 1

        if key == 'r':
          while len(annotations) > 0 and annotations[-1]['image_id'] == file_id:
            annotations = annotations[:-1]
          img_done = True

        if key == 'q':
          done = img_done = ann_done = True


        # segmentation 정보를 통해 bounding box 계산
        if ann_done and key != 'r':
          ann["segmentation"] = seg
          flat = [item for sublist in seg for item in sublist]
          x_flat = flat[::2]
          y_flat = flat[1::2]
          x_min = min(x_flat)
          y_min = min(y_flat)
          x_max = max(x_flat)
          y_max = max(y_flat)
          bbox = [x_min, y_min, x_max-x_min, y_max-y_min]
          ann["bbox"] = bbox
          annotations.append(ann)
          ann_id += 1

  # 저장
  print(json.dumps(annotations))
  with open(osp.join(args.output_path, args.output_name), 'w') as f:
    json.dump(annotations, f)

if __name__ == '__main__':
  main()
