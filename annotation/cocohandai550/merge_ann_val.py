import json

file_list = [
  "ann_val/out_block1.json",
  "ann_val/out_block2.json",
  "ann_val/out_block3.json",
  "ann_val/out_block4.json",
  "ann_val/out_block5.json",
  "ann_val/out_block6.json",
  "ann_val/out_block7.json",
  "ann_val/out_scatter_block1.json",
  "ann_val/out_scatter_block2.json",
  "ann_val/out_scatter_block3.json",
  "ann_val/out_scatter_block4.json",
  "ann_val/out_scatter_block5.json",
  "ann_val/out_scatter_block6.json",
  "ann_val/out_scatter_block7.json",
  "ann_val/out_cube.json"]

data = []
id_offset = 20000
for file_name in file_list:
  with open(file_name, "r") as f:
    loaded_data = json.load(f)

  for ann in loaded_data:
    ann["id"] += id_offset

  data.extend(loaded_data)

  id_offset += len(loaded_data)
  print(id_offset)



with open("ann_val/out_merged.json", "w") as f:
  json.dump(data, f)