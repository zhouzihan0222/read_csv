from ultralytics import YOLO

pre_model_name = 'yolo11n-seg.pt'
data_yaml_path = 'datasets/Data/fold_1.yaml'
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
        name='fold_1_seg_stanford_100'
    )