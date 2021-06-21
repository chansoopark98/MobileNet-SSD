import tensorflow as tf
from tensorflow import keras
from model.model import csnet_extra_model
from model.model import csnet_extra_model_post
from tensorflow.keras import layers

# train.py에서 priors를 변경하면 여기도 수정해야함
def model_build(model_mode, base_model_name, target_transform=None, train=True, pretrained=True, backbone_trainable=True, image_size=[512, 512],
                normalizations=[20, 20, 20, -1, -1], num_priors=[3, 3, 3, 3, 3]):
    if model_mode == 'voc':
        classes = 21
    elif model_mode == 'coco':
        classes = 81

    if train:
        inputs, output = csnet_extra_model(base_model_name, pretrained, image_size, backbone_trainable=backbone_trainable)
    else:
        inputs, output = csnet_extra_model_post(base_model_name, pretrained, image_size, target_transform=target_transform,
                                           backbone_trainable=backbone_trainable)
    model = keras.Model(inputs, outputs=output)
    return model
