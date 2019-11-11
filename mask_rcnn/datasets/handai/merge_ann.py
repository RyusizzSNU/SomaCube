import json

file_list = [
  "handai_dataset_ann/out_block1.json",
  "handai_dataset_ann/out_block2.json",
  "handai_dataset_ann/out_block3.json",
  "handai_dataset_ann/out_block4.json",
  "handai_dataset_ann/out_block5.json",
  "handai_dataset_ann/out_block6.json",
  "handai_dataset_ann/out_block7.json",
  "handai_dataset_ann/out_scatter_block1.json",
  "handai_dataset_ann/out_scatter_block2.json",
  "handai_dataset_ann/out_scatter_block3.json",
  "handai_dataset_ann/out_scatter_block4.json",
  "handai_dataset_ann/out_scatter_block5.json",
  "handai_dataset_ann/out_scatter_block6.json",
  "handai_dataset_ann/out_scatter_block7.json",
  "handai_dataset_ann/out_cube.json"]

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



with open("handai_dataset_ann/out_merged.json", "w") as f:
  json.dump(data, f)