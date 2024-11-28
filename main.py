import torch
import os
import numpy as np
import pandas as pd
from ultralytics import YOLO

torch.cuda.set_device(0)
dection_path = os.listdir("../count/frames")
model = YOLO('./model/yolo11x.pt')
model.to('cuda')

output = []
for data in dection_path:

    results = model("../count/frames/"+data, conf = 0.02, classes = 0)
    results[0].save_txt(f'results.txt')

    with open("results.txt") as f:
        count_result = sum(1 for _ in f)
    output.append(count_result)
    os.remove("results.txt")
    print(count_result)

arr = np.array(output)
df = pd.DataFrame (arr)
file
path = 'result0.02.xlsx'
df.to_excel(filepath, index=False)
