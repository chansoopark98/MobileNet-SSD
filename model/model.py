import tensorflow as tf
from tensorflow.keras.layers import Conv2D, BatchNormalization, ReLU, Concatenate, Flatten, Reshape, SeparableConv2D
from tensorflow.keras.activations import swish
from tensorflow.keras.applications import mobilenet_v2, vgg16, mobilenet
from utils.model_post_processing import post_process, post_process_predict
from tensorflow.keras import initializers, Input

# Import for L2 Normalize
from tensorflow.keras.layers import Layer
import tensorflow.keras.backend as K


MOMENTUM = 0.999
EPSILON = 1e-3


class Normalize(Layer):
    def __init__(self, scale=20, **kwargs):
        self.scale = scale
        super(Normalize, self).__init__(**kwargs)

    def build(self, input_shape):
        self.gamma = self.add_weight(name=self.name + '_gamma',
                                     shape=(input_shape[-1],),
                                     initializer=initializers.Constant(self.scale),
                                     trainable=True)
        super(Normalize, self).build(input_shape)

    def call(self, x, mask=None):
        return self.gamma * K.l2_normalize(x, axis=-1)

    def get_config(self):
        config = super().get_config().copy()
        return config

def build_backbone(IMAGE_SIZE=[300, 300]):
    weights = "imagenet"
    input_tensor = Input(shape=(IMAGE_SIZE[0],IMAGE_SIZE[1],3))
    base = mobilenet_v2.MobileNetV2(weights=weights, include_top=False, input_shape=[*IMAGE_SIZE, 3], input_tensor=input_tensor)
    base.summary()
    return base

def csnet_extra_model(normalizations, num_priors, num_classes=21, IMAGE_SIZE=[300, 300], convert_tfjs=False, target_transform=None):
    base = build_backbone(IMAGE_SIZE)
    # mobilenET V2
    # x2 = base.get_layer('block_6_expand_relu').output # 38x38 @ 192
    # x3 = base.get_layer('block_13_expand_relu').output # 19x19 @ 576
    # x4 = base.get_layer('block_16_project_BN').output # 10x10 @ 320

    # VGG16
    # x3 = base.get_layer('block4_conv3').output # 19x19 @ 576
    # x4 = base.get_layer('block5_conv3').output # 10x10 @ 320

    base_channel = 128

    x2 = base.get_layer('block_5_add').output # 38x38 @ 192
    x3 = base.get_layer('block_12_add').output # 19x19 @ 576
    x4 = base.get_layer('block_16_project_BN').output # 10x10 @ 320

    x5 = Conv2D(base_channel, kernel_size=1, strides=1, padding='same', use_bias=True)(x4)
    x5 = BatchNormalization(momentum=MOMENTUM, epsilon=EPSILON)(x5)
    # x5 = ReLU(6.)(x5)
    x5 = swish(x5)
    x5 = SeparableConv2D(base_channel * 2, kernel_size=3, strides=2, padding='same', use_bias=True)(x5)
    x5 = BatchNormalization(momentum=MOMENTUM, epsilon=EPSILON)(x5)
    # x5 = ReLU(6.)(x5)
    x5 = swish(x5)

    x6 = Conv2D(base_channel, kernel_size=1, strides=1, padding='same', use_bias=True)(x5)
    x6 = BatchNormalization(momentum=MOMENTUM, epsilon=EPSILON)(x6)
    # x6 = ReLU(6.)(x6)
    x6 = swish(x6)
    x6 = SeparableConv2D(base_channel * 2, kernel_size=3, strides=1, padding='valid', use_bias=True)(x6)
    x6 = BatchNormalization(momentum=MOMENTUM, epsilon=EPSILON)(x6)
    # x6 = ReLU(6.)(x6)
    x6 = swish(x6)

    x7 = Conv2D(base_channel, kernel_size=1, strides=1, padding='same', use_bias=True)(x6)
    x7 = BatchNormalization(momentum=MOMENTUM, epsilon=EPSILON)(x7)
    # x7 = ReLU(6.)(x7)
    x7 = swish(x7)
    x7 = SeparableConv2D(base_channel * 2, kernel_size=3, strides=1, padding='valid', use_bias=True)(x7)
    x7 = BatchNormalization(momentum=MOMENTUM, epsilon=EPSILON)(x7)
    # x7 = ReLU(6.)(x7)
    x7 = swish(x7)


    features = [x2, x3, x4, x5, x6, x7]


    mbox_conf = []
    mbox_loc = []

    num_priors = [4, 6, 6, 6, 4, 4]

    for i, layer in enumerate(features):
        x = layer
        name = x.name.split(':')[0] # name만 추출 (ex: block3b_add)

        if normalizations is not None and normalizations[i] > 0:
            x = Normalize(normalizations[i], name=name + '_norm')(x)

        x1 = SeparableConv2D(num_priors[i] * num_classes, 3, padding='same',
                             depthwise_initializer=initializers.VarianceScaling(),
                             pointwise_initializer=initializers.VarianceScaling(),
                             name= name + '_mbox_conf_1')(x)

        x1 = Flatten(name=name + '_mbox_conf_flat')(x1)
        mbox_conf.append(x1)

        x2 = SeparableConv2D(num_priors[i] * 4, 3, padding='same',
                             depthwise_initializer=initializers.VarianceScaling(),
                             pointwise_initializer=initializers.VarianceScaling(),
                             name= name + '_mbox_loc_1')(x)

        x2 = Flatten(name=name + '_mbox_loc_flat')(x2)
        mbox_loc.append(x2)

    mbox_loc = Concatenate(axis=1, name='mbox_loc')(mbox_loc)
    mbox_loc = Reshape((-1, 4), name='mbox_loc_final')(mbox_loc)

    mbox_conf = Concatenate(axis=1, name='mbox_conf')(mbox_conf)
    mbox_conf = Reshape((-1, num_classes), name='mbox_conf_logits')(mbox_conf)

    predictions = Concatenate(axis=2, name='predictions', dtype=tf.float32)([mbox_loc, mbox_conf])

    if convert_tfjs:
        predictions = post_process_predict(predictions, target_transform, confidence_threshold=0.05, classes=21)

    return base.input, predictions


# def csnet_extra_model_post(num_classes=21, IMAGE_SIZE=[300, 300], normalizations=[20, 20, 20, -1, -1], target_transform=None):
#     base = build_backbone(IMAGE_SIZE)
#
#     x2 = base.get_layer('block_6_expand_relu').output # 38x38 @ 192
#     x3 = base.get_layer('block_13_expand_relu').output # 19x19 @ 576
#     x4 = base.get_layer('block_16_project_BN').output # 10x10 @ 320
#
#     x5 = Conv2D(128, kernel_size=3, strides=1, padding='same', use_bias=True)(x4)
#     x5 = BatchNormalization(momentum=MOMENTUM, epsilon=EPSILON)(x5)
#     x5 = ReLU(6.)(x5)
#     x5 = Conv2D(256, kernel_size=3, strides=2, padding='same', use_bias=True)(x5)
#     x5 = BatchNormalization(momentum=MOMENTUM, epsilon=EPSILON)(x5)
#     x5 = ReLU(6.)(x5)
#
#     x6 = Conv2D(128, kernel_size=3, strides=1, padding='same', use_bias=True)(x5)
#     x6 = BatchNormalization(momentum=MOMENTUM, epsilon=EPSILON)(x6)
#     x6 = ReLU(6.)(x6)
#     x6 = Conv2D(256, kernel_size=3, strides=1, padding='valid', use_bias=True)(x6)
#     x6 = BatchNormalization(momentum=MOMENTUM, epsilon=EPSILON)(x6)
#     x6 = ReLU(6.)(x6)
#
#     x7 = Conv2D(128, kernel_size=3, strides=1, padding='same', use_bias=True)(x6)
#     x7 = BatchNormalization(momentum=MOMENTUM, epsilon=EPSILON)(x7)
#     x7 = ReLU(6.)(x7)
#     x7 = Conv2D(256, kernel_size=3, strides=1, padding='valid', use_bias=True)(x7)
#     x7 = BatchNormalization(momentum=MOMENTUM, epsilon=EPSILON)(x7)
#     x7 = ReLU(6.)(x7)
#
#
#     features = [x2, x3, x4, x5, x6, x7]
#
#     mbox_conf = []
#     mbox_loc = []
#
#     num_priors = [4, 6, 6, 6, 4, 4]
#
#
#     for i, layer in enumerate(features):
#         x = layer
#         name = x.name.split(':')[0] # name만 추출 (ex: block3b_add)
#
#         if normalizations is not None and normalizations[i] > 0:
#             x = Normalize(normalizations[i], name=name + '_norm')(x)
#
#         x1 = SeparableConv2D(num_priors[i] * num_classes, 3, padding='same',
#                              depthwise_initializer=initializers.VarianceScaling(),
#                              pointwise_initializer=initializers.VarianceScaling(),
#                              name= name + '_mbox_conf_1')(x)
#
#         x1 = Flatten(name=name + '_mbox_conf_flat')(x1)
#         mbox_conf.append(x1)
#
#         x2 = SeparableConv2D(num_priors[i] * 4, 3, padding='same',
#                              depthwise_initializer=initializers.VarianceScaling(),
#                              pointwise_initializer=initializers.VarianceScaling(),
#                              name= name + '_mbox_loc_1')(x)
#
#         x2 = Flatten(name=name + '_mbox_loc_flat')(x2)
#         mbox_loc.append(x2)
#
#     mbox_loc = Concatenate(axis=1, name='mbox_loc')(mbox_loc)
#     mbox_loc = Reshape((-1, 4), name='mbox_loc_final')(mbox_loc)
#
#     mbox_conf = Concatenate(axis=1, name='mbox_conf')(mbox_conf)
#     mbox_conf = Reshape((-1, num_classes), name='mbox_conf_logits')(mbox_conf)
#
#     predictions = Concatenate(axis=2, name='predictions', dtype=tf.float32)([mbox_loc, mbox_conf])
#
#     predictions = post_process_predict(predictions, target_transform, confidence_threshold=0.05, classes=21)
#
#     return base.input, predictions



