import efficientnet.keras as efn
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.layers import Conv2D, Add, Activation,  UpSampling2D, MaxPooling2D
from tensorflow.keras.layers import Concatenate, Flatten, Reshape, Dropout
from tensorflow.keras.applications import vgg16
from tensorflow.keras.applications import mobilenet_v2


def create_vgg16(base_model_name, pretrained=True, IMAGE_SIZE=[512, 512], trainable=True):
    weights = "imagenet"
    base = vgg16.VGG16(weights=weights, include_top=False, input_shape=[*IMAGE_SIZE, 3])

    return base



def SeparableConvBlock(feature, num_channels, kernel_size, strides, name, freeze_bn=False):
    # f1 = SeparableConv2D(num_channels, kernel_size=kernel_size, strides=strides, padding='same',
    #                             use_bias=True, name=name+'/conv')
    f1 = Conv2D(num_channels, kernel_size=kernel_size, strides=strides, padding='same',
                                use_bias=True, name=name+'/conv')(feature)
    # f2 = BatchNormalization(momentum=MOMENTUM, epsilon=EPSILON, name=name+'/bn')(f1)
    # f2 = BatchNormalization(freeze=freeze_bn, name=f'{name}/bn')
    return f1

def csnet_extra_model(base_model_name, pretrained=True, IMAGE_SIZE=[512, 512], backbone_trainable=True):



    p3 = base.get_layer('block4_conv3').output # 64 64 40
    p5 = base.get_layer('block5_conv3').output # 32 32 112
    p7 = SeparableConvBlock(p5, NUM_CHANNELS[MODEL_NAME[base_model_name]],3,2, 'test_conv')


    features = [p3, p5, p7]

    if base_model_name == 'B0':
        feature_resize = False
    else:
        feature_resize = True


    features = build_BiFPN(features=features, num_channels=NUM_CHANNELS[MODEL_NAME[base_model_name]],
                           id=0, resize=feature_resize, bn_trainable=bn_trainable)

    # predict features
    source_layers.append(features[0])
    source_layers.append(features[1])
    source_layers.append(features[2])
    source_layers.append(features[3])
    source_layers.append(features[4])
    print(source_layers)
    mbox_conf = []
    mbox_loc = []
    num_priors = [3, 3, 3, 3, 3]
    num_classes = 21
    for i, layer in enumerate(source_layers):
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


    # return base.input, source_layers, CLS_TIEMS[MODEL_NAME[base_model_name]]