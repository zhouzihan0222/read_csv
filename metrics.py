from ultralytics import YOLO
import numpy as np
import pandas as pd

import glob, os
from PIL import Image

results = list()
metric_values = dict()

model = YOLO(r'./runs/segment/fold_1_seg/weights/best.pt')
result = model.val(data='./datasets/Data/fold_1.yaml',split='test')

results.append(result)

model = YOLO(r'./runs/segment/fold_2_seg/weights/best.pt')
result = model.val(data='./datasets/Data/fold_2.yaml',split='test')
results.append(result)

model = YOLO(r'./runs/segment/fold_3_seg/weights/best.pt')
result = model.val(data='./datasets/Data/fold_3.yaml',split='test')
results.append(result)

model = YOLO(r'./runs/segment/fold_1_seg/weights/best.pt')
result = model.val(data='./datasets/Data/fold_4.yaml',split='test')
results.append(result)

model = YOLO(r'./runs/segment/fold_5_seg/weights/best.pt')
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

metric.to_csv('./runs/segment/5val_metrics_summary_test.csv', index=True)
print("Metrics saved to metrics_summary.csv")
