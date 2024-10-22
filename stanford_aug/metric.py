from ultralytics import YOLO
import pandas as pd

path = '/home/zihan/ultralytics/datasets/Data/kaggle_seg_aug/fold_1.yaml'
#path ='/home/zihan/ultralytics/datasets/Data/stanford_aug/stanford_aug.yaml'
# 加载模型
model = YOLO(r'/home/zihan/ultralytics/runs/segment/stanford_aug_seg/weights/best.pt')

# 运行验证
result = model.val(data=path, split='test')

# 将结果转换为DataFrame
metric_df = pd.DataFrame([result.results_dict])

# 打印结果
print(metric_df)

# 保存结果到CSV文件
output_path = '/home/zihan/ultralytics/runs/segment/kaggle_val_metrics_stanford_aug_summary_test.csv'
metric_df.to_csv(output_path, index=False)
print(f"Metrics saved to {output_path}")