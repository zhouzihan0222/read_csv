from ultralytics import YOLO
import numpy as np
import pandas as pd

import glob, os
from PIL import Image

results = list()
metric_values = dict()

model = YOLO(r'/home/zihan/ultralytics/runs/detect/kaggle_box_fold1/weights/best.pt')
result = model.val(data='/home/zihan/ultralytics/datasets/Data/kaggle_box/fold_1.yaml',split='test')

results.append(result)

model = YOLO(r'/home/zihan/ultralytics/runs/detect/kaggle_box_fold2/weights/best.pt')
result = model.val(data='/home/zihan/ultralytics/datasets/Data/kaggle_box/fold_2.yaml',split='test')
results.append(result)

model = YOLO(r'/home/zihan/ultralytics/runs/detect/kaggle_box_fold3/weights/best.pt')
result = model.val(data='/home/zihan/ultralytics/datasets/Data/kaggle_box/fold_3.yaml',split='test')
results.append(result)

model = YOLO(r'/home/zihan/ultralytics/runs/detect/kaggle_box_fold4/weights/best.pt')
result = model.val(data='/home/zihan/ultralytics/datasets/Data/kaggle_box/fold_4.yaml',split='test')
results.append(result)

model = YOLO(r'/home/zihan/ultralytics/runs/detect/kaggle_box_fold5/weights/best.pt')
result = model.val(data='/home/zihan/ultralytics/datasets/Data/kaggle_box/fold_5.yaml',split='test')
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

metric.to_csv('/home/zihan/ultralytics/runs/detect/5val_metrics_kaggle_box_summary_test_new.csv', index=True)
print("Metrics saved to metrics_box_summary.csv")
