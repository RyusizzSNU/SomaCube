import json

file_list = [
  "ann_train/out_block1.json",
  "ann_train/out_block2.json",
  "ann_train/out_block3.json",
  "ann_train/out_block4.json",
  "ann_train/out_block5.json",
  "ann_train/out_block6.json",
  "ann_train/out_block7.json",
  "ann_train/out_scatter_block1.json",
  "ann_train/out_scatter_block2.json",
  "ann_train/out_scatter_block3.json",
  "ann_train/out_scatter_block4.json",
  "ann_train/out_scatter_block5.json",
  "ann_train/out_scatter_block6.json",
  "ann_train/out_scatter_block7.json",
  "ann_train/out_cube.json",
  "ann_train/out2_block1.json",
  "ann_train/out2_block2.json",
  "ann_train/out2_block3.json",
  "ann_train/out2_block4.json",
  "ann_train/out2_block5.json",
  "ann_train/out2_block6.json",
  "ann_train/out2_block7.json",
  "ann_train/out2_scatter_block1.json",
  "ann_train/out2_scatter_block2.json",
  "ann_train/out2_scatter_block3.json",
  "ann_train/out2_scatter_block4.json",
  "ann_train/out2_scatter_block5.json",
  "ann_train/out2_scatter_block6.json",
  "ann_train/out2_scatter_block7.json",
  "ann_train/out2_cube1.json",
  "ann_train/out2_cube2.json",
  "ann_train/out2_cube3.json"]

data = []
id_offset = 0
for file_name in file_list:
  with open(file_name, "r") as f:
    loaded_data = json.load(f)

  for ann in loaded_data:
    ann["id"] += id_offset

  data.extend(loaded_data)

  id_offset += len(loaded_data)
  print(id_offset)



with open("ann_train/out_merged.json", "w") as f:
  json.dump(data, f)