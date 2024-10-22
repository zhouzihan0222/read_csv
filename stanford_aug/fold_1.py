from ultralytics import YOLO

# 加载模型
pre_model_name = 'yolo11n-seg.pt'

data_yaml_path = '/home/zihan/ultralytics/datasets/Data/stanford_aug/stanford_aug.yaml'

model_yaml_path = "/home/zihan/ultralytics/ultralytics/cfg/models/11/yolo11-seg.yaml"


if __name__ == '__main__':
    #加载预训练模型
    model = YOLO(model_yaml_path).load(pre_model_name)
    #训练模型
    results = model.train(
        data=data_yaml_path,
        epochs=100,
        batch=64,
        patience=15,
        lr0=1e-3,
        lrf=0.01,
        augment=True,
        save_period=10,
        name='stanford_aug_seg'
    )