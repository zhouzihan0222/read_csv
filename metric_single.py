from ultralytics import YOLO
import numpy as np
import pandas as pd
import glob, os
from PIL import Image

results = list()
metric_values = dict()

model = YOLO(r'./runs/segment/fold_1_seg_stanford_1002/weights/best.pt')
result = model.val(data='./datasets/Data/fold_1.yaml', split='test')

# 打印 result 对象的内容，以了解它的结构
print("Result object:")
print(result)

# 如果 result 是一个列表，我们应该遍历它
if isinstance(result, list):
    results = result
else:
    # 如果 result 不是列表，我们假设它是单个结果对象
    results = [result]

for result in results:
    # 打印每个 result 对象的 results_dict
    print("Result's results_dict:")
    print(result.results_dict)
    
    for metric, metric_val in result.results_dict.items():
        if metric not in metric_values:
            metric_values[metric] = []
        metric_values[metric].append(metric_val)

# 打印收集到的 metric_values
print("Collected metric_values:")
print(metric_values)

metric_df = pd.DataFrame.from_dict(metric_values)

# 打印 metric_df 的内容和形状
print("Metric DataFrame:")
print(metric_df)
print("DataFrame shape:", metric_df.shape)

if not metric_df.empty:
    visualize_metric = ['mean', 'std', 'min', 'max']
    metric = metric_df.describe().loc[visualize_metric]
    print(metric)
    metric.to_csv('runs/segment/fold_1_seg_stanford_1002/metrics_mask_summary_test.csv', index=True)
    print("Metrics saved to metrics_mask_summary_test.csv")
else:
    print("The DataFrame is empty. No metrics to save.")