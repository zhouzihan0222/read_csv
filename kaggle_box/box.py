from ultralytics import YOLO

# 加载模型
pre_model_name = 'yolo11n.pt'
data_yaml_path = '/home/zihan/ultralytics/datasets/Data/kaggle_box/fold_1.yaml'
model_yaml_path = "/home/zihan/ultralytics/ultralytics/cfg/models/11/yolo11.yaml"

if __name__ == '__main__':
    #加载预训练模型
    model = YOLO(model_yaml_path).load(pre_model_name)
    #训练模型
    results = model.train(
        data=data_yaml_path,
        epochs=64,
        batch=64,
        #augment=True,
        name='kaggle_box_fold1'
    )

pre_model_name = 'yolo11n.pt'

data_yaml_path = '/home/zihan/ultralytics/datasets/Data/kaggle_box/fold_2.yaml'

model_yaml_path = "/home/zihan/ultralytics/ultralytics/cfg/models/11/yolo11.yaml"


if __name__ == '__main__':
    #加载预训练模型
    model = YOLO(model_yaml_path).load(pre_model_name)
    #训练模型
    results = model.train(
        data=data_yaml_path,
        epochs=64,
        batch=64,
        #augment=True,
        name='kaggle_box_fold2'
    )

pre_model_name = 'yolo11n.pt'

data_yaml_path = '/home/zihan/ultralytics/datasets/Data/kaggle_box/fold_3.yaml'

model_yaml_path = "/home/zihan/ultralytics/ultralytics/cfg/models/11/yolo11.yaml"


if __name__ == '__main__':
    #加载预训练模型
    model = YOLO(model_yaml_path).load(pre_model_name)
    #训练模型
    results = model.train(
        data=data_yaml_path,
        epochs=64,
        batch=64,
        #augment=True,
        name='kaggle_box_fold3'
    )

pre_model_name = 'yolo11n.pt'

data_yaml_path = '/home/zihan/ultralytics/datasets/Data/kaggle_box/fold_4.yaml'

model_yaml_path = "/home/zihan/ultralytics/ultralytics/cfg/models/11/yolo11.yaml"


if __name__ == '__main__':
    #加载预训练模型
    model = YOLO(model_yaml_path).load(pre_model_name)
    #训练模型
    results = model.train(
        data=data_yaml_path,
        epochs=64,
        batch=64,
        #augment=True,
        name='kaggle_box_fold4'
    )




pre_model_name = 'yolo11n.pt'

data_yaml_path = '/home/zihan/ultralytics/datasets/Data/kaggle_box/fold_5.yaml'

model_yaml_path = "/home/zihan/ultralytics/ultralytics/cfg/models/11/yolo11.yaml"

if __name__ == '__main__':
    #加载预训练模型
    model = YOLO(model_yaml_path).load(pre_model_name)
    #训练模型
    results = model.train(
        data=data_yaml_path,
        epochs=64,
        batch=64,
        #augment=True,
        name='kaggle_box_fold5'
    )