from ultralytics import YOLO

# 加载模型
pre_model_name = 'yolo11n-seg.pt'

data_yaml_path = 'datasets/Data/fold_3.yaml'

model_yaml_path = "/home/zihan/ultralytics/ultralytics/cfg/models/11/yolo11-seg.yaml"


if __name__ == '__main__':
    #加载预训练模型
    model = YOLO(model_yaml_path).load(pre_model_name)
    #训练模型
    results = model.train(data=data_yaml_path,
                          epochs=64,
                          batch=-1,
                          name='fold_3_seg')
