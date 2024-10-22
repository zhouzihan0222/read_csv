from ultralytics import YOLO
import numpy as np
import pandas as pd

import glob, os
from PIL import Image

results = list()
metric_values = dict()

path ='/home/zihan/ultralytics/datasets/Data/kaggle_seg_aug/fold_1.yaml'
#path = '/home/zihan/ultralytics/datasets/Data/stanford/stanford.yaml'

model = YOLO(r'/home/zihan/ultralytics/runs/segment/fold_1_seg_aug/weights/best.pt')
result = model.val(data=path,split='test')

results.append(result)

model = YOLO(r'/home/zihan/ultralytics/runs/segment/fold_2_seg_aug/weights/best.pt')
result = model.val(data=path,split='test')
results.append(result)

model = YOLO(r'/home/zihan/ultralytics/runs/segment/fold_3_seg_aug/weights/best.pt')
result = model.val(data=path,split='test')
results.append(result)

model = YOLO(r'/home/zihan/ultralytics/runs/segment/fold_4_seg_aug/weights/best.pt')
result = model.val(data=path,split='test')
results.append(result)

model = YOLO(r'/home/zihan/ultralytics/runs/segment/fold_5_seg_aug/weights/best.pt')
result = model.val(data=path,split='test')
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

metric.to_csv('/home/zihan/ultralytics/runs/segment/stanford_5val_metrics_aug_kaggle_seg_summary_test.csv', index=True)
print("Metrics saved to metrics_box_summary.csv")
