from ultralytics import YOLO

# 加载模型
pre_model_name = 'yolo11n-seg.pt'
data_yaml_path = '/home/zihan/ultralytics/datasets/Data/stanford/stanford.yaml'
model_yaml_path = "/home/zihan/ultralytics/ultralytics/cfg/models/11/yolo11-seg.yaml"

if __name__ == '__main__':
    # 加载预训练模型
    model = YOLO(model_yaml_path).load(pre_model_name)
    
    # 第一次训练，使用增强
    results_with_augment = model.train(
        data=data_yaml_path,
        epochs=32,
        batch=64,
        augment=True,
        name='stanford_seg_with_augment'
    )
    
    # 第二次训练，不使用增强
    results_without_augment = model.train(
        data=data_yaml_path,
        epochs=32,
        batch=64,
        augment=False,
        name='stanford_seg_without_augment'
    )