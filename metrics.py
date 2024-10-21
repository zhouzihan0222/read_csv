from ultralytics import YOLO
import numpy as np
import pandas as pd

import glob, os
from PIL import Image

results = list()
metric_values = dict()

model = YOLO(r'./runs/detect/fold_1_box/weights/best.pt')
result = model.val(data='./datasets/Data/fold_1.yaml',split='test')

results.append(result)

model = YOLO(r'./runs/detect/fold_2_box/weights/best.pt')
result = model.val(data='./datasets/Data/fold_2.yaml',split='test')
results.append(result)

model = YOLO(r'./runs/detect/fold_3_box/weights/best.pt')
result = model.val(data='./datasets/Data/fold_3.yaml',split='test')
results.append(result)

model = YOLO(r'./runs/detect/fold_4_box/weights/best.pt')
result = model.val(data='./datasets/Data/fold_4.yaml',split='test')
results.append(result)

model = YOLO(r'./runs/detect/fold_5_box/weights/best.pt')
result = model.val(data='./datasets/Data/fold_5.yaml',split='test')
results.append(result)


metric_values = dict()

for result in results:
    for metric, metric_val in result.results_dict.items():
        if metric not in metric_values:
            metric_values[metric] = []
        metric_values[metric].append(metric_val)

metric_df = pd.DataFrame.from_dict(metric_values)
visualize_metric = ['mean', 'std', 'min', 'max']

metric = metric_df.describe().loc[visualize_metric]
print(metric)

metric.to_csv('./runs/detect/5val_metrics_box_summary_test.csv', index=True)
print("Metrics saved to metrics_box_summary.csv")
