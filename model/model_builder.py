import tensorflow as tf
from tensorflow import keras
from model.model import csnet_extra_model
from model.model import csnet_extra_model_post
from tensorflow.keras import layers

# train.py에서 priors를 변경하면 여기도 수정해야함
def model_build(model_mode, target_transform=None, train=True, image_size=[512, 512], num_priors=[3, 3, 3, 3, 3]):
    if model_mode == 'voc':
        classes = 21
    else:
        classes = 81

    if train:
        inputs, output = csnet_extra_model(num_classes=classes, IMAGE_SIZE=image_size)
    else:
        inputs, output = csnet_extra_model_post(num_classes=classes, IMAGE_SIZE=image_size, target_transform=target_transform)

    model = keras.Model(inputs, outputs=output)
    return model
