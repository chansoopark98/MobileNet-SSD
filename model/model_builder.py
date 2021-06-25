import tensorflow as tf
from tensorflow import keras
from model.model import csnet_extra_model
# from model.model import csnet_extra_model_post
from tensorflow.keras import layers

# train.py에서 priors를 변경하면 여기도 수정해야함
def model_build(model_mode, target_transform=None, train=True, image_size=[512, 512]):
    normalizations = [20, 20, 20, -1, -1, -1]
    num_priors = [4, 6, 6, 6, 4, 4]
    if model_mode == 'voc':
        classes = 21
    else:
        classes = 81

    if train:
        inputs, output = csnet_extra_model(normalizations, num_priors, num_classes=classes, IMAGE_SIZE=image_size,
                                           convert_tfjs=False, target_transform=None)
    else:
        inputs, output = csnet_extra_model(normalizations, num_priors, num_classes=classes, IMAGE_SIZE=image_size,
                                           convert_tfjs=True, target_transform=target_transform)

    model = keras.Model(inputs, outputs=output)
    return model
