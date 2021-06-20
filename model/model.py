import tensorflow as tf
from tensorflow.keras.layers import Conv2D, Activation, BatchNormalization, ReLU
from tensorflow.keras.layers import Concatenate, Flatten, Reshape
from tensorflow.keras.applications import mobilenet_v2

MOMENTUM = 0.999
EPSILON = 1e-3

def create_vgg16(base_model_name, pretrained=True, IMAGE_SIZE=[300, 300], trainable=True):
    weights = "imagenet"
    base = mobilenet_v2.MobileNetV2(weights=weights, include_top=False, input_shape=[*IMAGE_SIZE, 3])
    return base

def csnet_extra_model(base_model_name, pretrained=True, IMAGE_SIZE=[300, 300], backbone_trainable=True):
    base = create_vgg16(base_model_name, pretrained, IMAGE_SIZE, trainable=backbone_trainable)

    x2 = base.get_layer('block_6_expand_relu').output # 38x38 @ 192
    x3 = base.get_layer('block_13_expand_relu').output # 19x19 @ 576
    x4 = base.get_layer('block_16_project_BN').output # 10x10 @ 320

    x5 = Conv2D(128, kernel_size=3, strides=1, padding='same', use_bias=True)(x4)
    x5 = BatchNormalization(momentum=MOMENTUM, epsilon=EPSILON)(x5)
    x5 = ReLU(6.)(x5)
    x5 = Conv2D(256, kernel_size=3, strides=2, padding='same', use_bias=True)(x5)
    x5 = BatchNormalization(momentum=MOMENTUM, epsilon=EPSILON)(x5)
    x5 = ReLU(6.)(x5)

    x6 = Conv2D(128, kernel_size=3, strides=1, padding='same', use_bias=True)(x5)
    x6 = BatchNormalization(momentum=MOMENTUM, epsilon=EPSILON)(x6)
    x6 = ReLU(6.)(x6)
    x6 = Conv2D(256, kernel_size=3, strides=1, padding='valid', use_bias=True)(x6)
    x6 = BatchNormalization(momentum=MOMENTUM, epsilon=EPSILON)(x6)
    x6 = ReLU(6.)(x6)

    x7 = Conv2D(128, kernel_size=3, strides=1, padding='same', use_bias=True)(x6)
    x7 = BatchNormalization(momentum=MOMENTUM, epsilon=EPSILON)(x7)
    x7 = ReLU(6.)(x7)
    x7 = Conv2D(256, kernel_size=3, strides=1, padding='valid', use_bias=True)(x7)
    x7 = BatchNormalization(momentum=MOMENTUM, epsilon=EPSILON)(x7)
    x7 = ReLU(6.)(x7)


    features = [x2, x3, x4, x5, x6, x7]

    mbox_conf = []
    mbox_loc = []

    num_priors = [4, 6, 6, 6, 4, 4]
    num_classes = 21

    for i, layer in enumerate(features):
        x = layer
        name = x.name.split(':')[0] # name만 추출 (ex: block3b_add)
        x1 = Conv2D(num_priors[i] * num_classes, 3, padding='same')(x)
        x1 = Flatten(name=name + '_mbox_conf_flat')(x1)

        mbox_conf.append(x1)
        x2 = Conv2D(num_priors[i] * 4, 3, padding='same')(x)
        x2 = Flatten()(x2)

        mbox_loc.append(x2)

    mbox_loc = Concatenate(axis=1, name='mbox_loc')(mbox_loc)
    mbox_loc = Reshape((-1, 4), name='mbox_loc_final')(mbox_loc)

    mbox_conf = Concatenate(axis=1, name='mbox_conf')(mbox_conf)
    mbox_conf = Reshape((-1, num_classes), name='mbox_conf_logits')(mbox_conf)

    predictions = Concatenate(axis=2, name='predictions', dtype=tf.float32)([mbox_loc, mbox_conf])

    return base.input, predictions
