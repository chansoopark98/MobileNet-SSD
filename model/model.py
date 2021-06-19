import efficientnet.keras as efn
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.layers import Conv2D, Add, Activation,  UpSampling2D, MaxPooling2D
from tensorflow.keras.layers import Concatenate, Flatten, Reshape, Dropout
from tensorflow.keras.applications import vgg16

NUM_CHANNELS = [64, 88, 112, 160, 224, 288, 384]
FPN_TIMES = [3, 4, 5, 6, 7, 7, 8]
CLS_TIEMS = [3, 3, 3, 4, 4, 4, 5]


MOMENTUM = 0.997
EPSILON = 1e-4

GET_EFFICIENT_NAME = {
    'B0': ['block3b_add', 'block5c_add', 'block7a_project_bn'],
    'B1': ['block3c_add', 'block5d_add', 'block7b_add'],
    'B2': ['block3c_add', 'block5d_add', 'block7b_add'],
    'B3': ['block3c_add', 'block5e_add', 'block7b_add'],
    'B4': ['block3d_add', 'block5f_add', 'block7b_add'],
    'B5': ['block3e_add', 'block5g_add', 'block7c_add'],
    'B6': ['block3f_add', 'block5h_add', 'block7c_add'],
    'B7': ['block3g_add', 'block5j_add', 'block7d_add'],
}

MODEL_NAME = {
    'B0': 0,
    'B1': 1,
    'B2': 2,
    'B3': 3,
    'B4': 4,
    'B5': 5,
    'B6': 6,
    'B7': 7
}


def create_vgg16(base_model_name, pretrained=True, IMAGE_SIZE=[512, 512], trainable=True):
    weights = "imagenet"
    base = vgg16.VGG16(weights=weights, include_top=False, input_shape=[*IMAGE_SIZE, 3])

    return base


def remove_dropout(model):
    for layer in model.layers:
        if isinstance(layer, Dropout):
            layer.rate = 0
    model_copy = keras.models.clone_model(model)
    model_copy.set_weights(model.get_weights())
    del model

    return model_copy



def create_efficientNet(base_model_name, pretrained=True, IMAGE_SIZE=[512, 512], trainable=True):
    if pretrained is False:
        weights = None



    else:
        weights = "imagenet"

    if base_model_name == 'B0':
        base = efn.EfficientNetB0(weights=weights, include_top=False, input_shape=[*IMAGE_SIZE, 3])

    elif base_model_name == 'B1':
        base = efn.EfficientNetB1(weights=weights, include_top=False, input_shape=[*IMAGE_SIZE, 3])

    elif base_model_name == 'B2':
        base = efn.EfficientNetB2(weights=weights, include_top=False, input_shape=[*IMAGE_SIZE, 3])

    elif base_model_name == 'B3':
        base = efn.EfficientNetB3(weights=weights, include_top=False, input_shape=[*IMAGE_SIZE, 3])

    elif base_model_name == 'B4':
        base = efn.EfficientNetB4(weights=weights, include_top=False, input_shape=[*IMAGE_SIZE, 3])

    elif base_model_name == 'B5':
        base = efn.EfficientNetB5(weights=weights, include_top=False, input_shape=[*IMAGE_SIZE, 3])

    elif base_model_name == 'B6':
        base = efn.EfficientNetB6(weights=weights, include_top=False, input_shape=[*IMAGE_SIZE, 3])

    elif base_model_name == 'B7':
        base = efn.EfficientNetB7(weights=weights, include_top=False, input_shape=[*IMAGE_SIZE, 3])

    base = remove_dropout(base)
    base.trainable = trainable


    return base




def SeparableConvBlock(feature, num_channels, kernel_size, strides, name, freeze_bn=False):
    # f1 = SeparableConv2D(num_channels, kernel_size=kernel_size, strides=strides, padding='same',
    #                             use_bias=True, name=name+'/conv')
    f1 = Conv2D(num_channels, kernel_size=kernel_size, strides=strides, padding='same',
                                use_bias=True, name=name+'/conv')(feature)
    # f2 = BatchNormalization(momentum=MOMENTUM, epsilon=EPSILON, name=name+'/bn')(f1)
    # f2 = BatchNormalization(freeze=freeze_bn, name=f'{name}/bn')
    return f1

def build_BiFPN(features, num_channels=64 , id=0, resize=False, bn_trainable=True):
    if resize:
        padding = 'valid'
    else:
        padding = 'same'

    if id == 0:


        C3, C4, C5 = features
        P3_in = C3 # 36x36
        P4_in = C4 # 18x18
        P5_in = C5 # 9x9

        P6_in = Conv2D(num_channels, kernel_size=1, padding='same', name='resample_p6/conv2d')(C5)
        # P6_in = BatchNormalization(momentum=MOMENTUM, epsilon=EPSILON,  trainable=bn_trainable, name='resample_p6/bn')(P6_in)

        # padding
        P6_in = MaxPooling2D(pool_size=3, strides=2, padding=padding, name='resample_p6/maxpool')(P6_in) # 4x4

        P7_in = MaxPooling2D(pool_size=3, strides=2, padding='same', name='resample_p7/maxpool')(P6_in) # 2x2


        if resize:
            P7_U = tf.image.resize(P7_in, (P6_in.shape[1:3]))
        else:
            P7_U = UpSampling2D()(P7_in) # 2x2 to 4x4

        P6_td = Add(name='fpn_cells/cell_/fnode0/add')([P6_in, P7_U])
        P6_td = Activation('relu')(P6_td)
        P6_td = SeparableConvBlock(P6_td, num_channels=num_channels, kernel_size=3, strides=1,
                                   name='fpn_cells/cell_/fnode0/op_after_combine5')
        P5_in_1 = Conv2D(num_channels, kernel_size=1, padding='same',
                                name='fpn_cells/cell_/fnode1/resample_0_2_6/conv2d')(P5_in)
        # P5_in_1 = BatchNormalization(momentum=MOMENTUM, epsilon=EPSILON, trainable=bn_trainable,
        #                                     name='fpn_cells/cell_/fnode1/resample_0_2_6/bn')(P5_in_1)

        if resize:
            P6_U = tf.image.resize(P6_td, (P5_in_1.shape[1:3]))
        else:
            P6_U = UpSampling2D()(P6_td)

        P5_td = Add(name='fpn_cells/cell_/fnode1/add')([P5_in_1, P6_U]) # 9x9
        P5_td = Activation('relu')(P5_td)
        P5_td = SeparableConvBlock(P5_td, num_channels=num_channels, kernel_size=3, strides=1,
                                   name='fpn_cells/cell_/fnode1/op_after_combine6')
        P4_in_1 = Conv2D(num_channels, kernel_size=1, padding='same',
                                name='fpn_cells/cell_/fnode2/resample_0_1_7/conv2d')(P4_in) # 18x18
        # P4_in_1 = BatchNormalization(momentum=MOMENTUM, epsilon=EPSILON, trainable=bn_trainable,
        #                                     name='fpn_cells/cell_/fnode2/resample_0_1_7/bn')(P4_in_1)

        P5_U = UpSampling2D()(P5_td)
        P4_td = Add(name='fpn_cells/cell_/fnode2/add')([P4_in_1, P5_U]) # 18x18
        P4_td = Activation('relu')(P4_td)
        P4_td = SeparableConvBlock(P4_td, num_channels=num_channels, kernel_size=3, strides=1,
                                   name='fpn_cells/cell_/fnode2/op_after_combine7')
        P3_in = Conv2D(num_channels, kernel_size=1, padding='same',
                              name='fpn_cells/cell_/fnode3/resample_0_0_8/conv2d')(P3_in) # 36x36
        # P3_in = BatchNormalization(momentum=MOMENTUM, epsilon=EPSILON, trainable=bn_trainable,
        #                                   name=f'fpn_cells/cell_/fnode3/resample_0_0_8/bn')(P3_in)

        P4_U = UpSampling2D()(P4_td) # 18x18 to 36x36
        P3_out = Add(name='fpn_cells/cell_/fnode3/add')([P3_in, P4_U])
        P3_out = Activation('relu')(P3_out)
        P3_out = SeparableConvBlock(P3_out, num_channels=num_channels, kernel_size=3, strides=1,
                                    name='fpn_cells/cell_/fnode3/op_after_combine8')
        P4_in_2 = Conv2D(num_channels, kernel_size=1, padding='same',
                                name='fpn_cells/cell_/fnode4/resample_0_1_9/conv2d')(P4_in)
        # P4_in_2 = BatchNormalization(momentum=MOMENTUM, epsilon=EPSILON, trainable=bn_trainable,
        #                                     name='fpn_cells/cell_/fnode4/resample_0_1_9/bn')(P4_in_2)

        P3_D = MaxPooling2D(pool_size=3, strides=2, padding='same')(P3_out)
        P4_out = Add(name='fpn_cells/cell_/fnode4/add')([P4_in_2, P4_td, P3_D])
        P4_out = Activation('relu')(P4_out)
        P4_out = SeparableConvBlock(P4_out, num_channels=num_channels, kernel_size=3, strides=1,
                                    name='fpn_cells/cell_/fnode4/op_after_combine9')

        P5_in_2 = Conv2D(num_channels, kernel_size=1, padding='same',
                                name='fpn_cells/cell_/fnode5/resample_0_2_10/conv2d')(P5_in)
        # P5_in_2 = BatchNormalization(momentum=MOMENTUM, epsilon=EPSILON, trainable=bn_trainable,
        #                                     name='fpn_cells/cell_/fnode5/resample_0_2_10/bn')(P5_in_2)

        P4_D = MaxPooling2D(pool_size=3, strides=2, padding='same')(P4_out)
        P5_out = Add(name='fpn_cells/cell_/fnode5/add')([P5_in_2, P5_td, P4_D]) # 9x9
        P5_out = Activation('relu')(P5_out)
        P5_out = SeparableConvBlock(P5_out, num_channels=num_channels, kernel_size=3, strides=1,
                                    name='fpn_cells/cell_/fnode5/op_after_combine10')

        # padding
        P5_D = MaxPooling2D(pool_size=3, strides=2, padding=padding)(P5_out) # 9x9 to 4x4

        P6_out = Add(name='fpn_cells/cell_/fnode6/add')([P6_in, P6_td, P5_D])
        P6_out = Activation('relu')(P6_out)
        P6_out = SeparableConvBlock(P6_out, num_channels=num_channels, kernel_size=3, strides=1,
                                    name='fpn_cells/cell_/fnode6/op_after_combine11')

        P6_D = MaxPooling2D(pool_size=3, strides=2, padding='same')(P6_out)
        P7_out = Add(name='fpn_cells/cell_/fnode7/add')([P7_in, P6_D])
        P7_out = Activation('relu')(P7_out)
        P7_out = SeparableConvBlock(P7_out, num_channels=num_channels, kernel_size=3, strides=1,
                                    name='fpn_cells/cell_/fnode7/op_after_combine12')


        return [P3_out, P4_td, P5_td, P6_td, P7_out]

    else:

        P3_in, P4_in, P5_in, P6_in, P7_in = features



        if resize:
            P7_U = tf.image.resize(P7_in, (P6_in.shape[1:3]))
        else:
            P7_U = UpSampling2D()(P7_in) # 2x2 to 4x4

        P6_td = Add(name=f'fpn_cells/cell_{id}/fnode0/add')([P6_in, P7_U])
        P6_td = Activation('relu')(P6_td)
        P6_td = SeparableConvBlock(num_channels=num_channels, kernel_size=3, strides=1,
                                   name=f'fpn_cells/cell_{id}/fnode0/op_after_combine5')(P6_td)

        if resize:
            P6_U = tf.image.resize(P6_td, (P5_in.shape[1:3]))
        else:
            P6_U = UpSampling2D()(P6_td) # 4x4 to 9x9

        P5_td = Add(name=f'fpn_cells/cell_{id}/fnode1/add')([P5_in, P6_U]) # 9x9
        P5_td = Activation('relu')(P5_td)
        P5_td = SeparableConvBlock(num_channels=num_channels, kernel_size=3, strides=1,
                                   name=f'fpn_cells/cell_{id}/fnode1/op_after_combine6')(P5_td)
        P5_U = UpSampling2D()(P5_td) # 9x9 to 18x18
        P4_td = Add(name=f'fpn_cells/cell_{id}/fnode2/add')([P4_in, P5_U]) # 18x18
        P4_td = Activation('relu')(P4_td)
        P4_td = SeparableConvBlock(num_channels=num_channels, kernel_size=3, strides=1,
                                   name=f'fpn_cells/cell_{id}/fnode2/op_after_combine7')(P4_td)
        P4_U = UpSampling2D()(P4_td) # 18x18 to 36x36
        P3_out = Add(name=f'fpn_cells/cell_{id}/fnode3/add')([P3_in, P4_U])
        P3_out = Activation('relu')(P3_out)
        P3_out = SeparableConvBlock(num_channels=num_channels, kernel_size=3, strides=1,
                                    name=f'fpn_cells/cell_{id}/fnode3/op_after_combine8')(P3_out)
        P3_D = MaxPooling2D(pool_size=3, strides=2, padding='same')(P3_out) # 36x36 to 18x18
        P4_out = Add(name=f'fpn_cells/cell_{id}/fnode4/add')([P4_in, P4_td, P3_D])
        P4_out = Activation('relu''relu')(P4_out)
        P4_out = SeparableConvBlock(num_channels=num_channels, kernel_size=3, strides=1,
                                    name=f'fpn_cells/cell_{id}/fnode4/op_after_combine9')(P4_out)

        P4_D = MaxPooling2D(pool_size=3, strides=2, padding='same')(P4_out) # 18x18 to 9x9
        P5_out = Add(name=f'fpn_cells/cell_{id}/fnode5/add')([P5_in, P5_td, P4_D])
        P5_out = Activation('relu')(P5_out)
        P5_out = SeparableConvBlock(num_channels=num_channels, kernel_size=3, strides=1,
                                    name=f'fpn_cells/cell_{id}/fnode5/op_after_combine10')(P5_out)

        # padding
        P5_D = MaxPooling2D(pool_size=3, strides=2, padding=padding)(P5_out)  # 9x9 to 4x4

        P6_out = Add(name=f'fpn_cells/cell_{id}/fnode6/add')([P6_in, P6_td, P5_D])
        P6_out = Activation('relu')(P6_out)
        P6_out = SeparableConvBlock(num_channels=num_channels, kernel_size=3, strides=1,
                                    name=f'fpn_cells/cell_{id}/fnode6/op_after_combine11')(P6_out)

        P6_D = MaxPooling2D(pool_size=3, strides=2, padding='same')(P6_out)
        P7_out = Add(name=f'fpn_cells/cell_{id}/fnode7/add')([P7_in, P6_D])
        P7_out = Activation('relu')(P7_out)
        P7_out = SeparableConvBlock(num_channels=num_channels, kernel_size=3, strides=1,
                                    name=f'fpn_cells/cell_{id}/fnode7/op_after_combine12')(P7_out)



        return [P3_out, P4_td, P5_td, P6_td, P7_out]


def csnet_extra_model(base_model_name, pretrained=True, IMAGE_SIZE=[512, 512], backbone_trainable=True):

    if backbone_trainable == True:
        bn_trainable = True
    else:
        bn_trainable = True

    print("backbone_trainable", backbone_trainable)
    print("bn_trainable", bn_trainable)
    source_layers = []
    # base = create_efficientNet(base_model_name, pretrained, IMAGE_SIZE, trainable=backbone_trainable)
    #
    # layer_names = GET_EFFICIENT_NAME[base_model_name]
    base = create_vgg16(base_model_name, pretrained, IMAGE_SIZE, trainable=backbone_trainable)


    # get extra layer
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
        # source_layers
        # name='block3b_add/add_1:0  shape=(batch, 38, 38, 40)
        # name='block5c_add/add_1:0 shape=(batch, 19, 19, 112)
        # name='block7a_project_bn/cond_1/Identity:0' shape=(batch, 10, 10, 320)
        # name='activation_1/Relu:0' shape=(batch, 5, 5, 256)
        # name='activation_3/Relu:0' shape=(batch, 3, 3, 256)
        # name='activation_5/Relu:0' shape=(batch, 1, 1, 256)
        x = layer
        # name = x.name.split('/')[0] # name만 추출 (ex: block3b_add)
        name = x.name.split(':')[0] # name만 추출 (ex: block3b_add)

        # <<< reduce norm
        # if normalizations is not None and normalizations[i] > 0:
        #    x = Normalize(normalizations[i], name=name + '_norm')(x)
           #print('norm_feature : '+x.name)

        # x = activation_5/Relu:0, shape=(Batch, 1, 1, 256)
        # print("start_multibox_head.py")
        # print("num_priors[i]",num_priors[i]) #6 (첫 번째 38,38일 경우)
        # print("num_classes",num_classes) #21
        # print("num_priors[i] * num_classes",num_priors[i] * num_classes) # 126

        ## original ----
        # x1 = Conv2D(num_priors[i] * num_classes, 3, padding='same', name= name + '_mbox_conf')(x)
        x1 = Conv2D(num_priors[i] * num_classes, 3, padding='same')(x)
        # x1 = SeparableConv2D(num_priors[i] * num_classes, 3, padding='same', use_bias=False, kernel_regularizer=l2(5e-4), name= name + '_mbox_conf')(x)
        # x1 = Flatten()(x1)
        x1 = Flatten(name=name + '_mbox_conf_flat')(x1)


        # x1 = activation_b5_mbox_conf_flat/Reshape:0 , shape=(Batch , 84)
        mbox_conf.append(x1)

        # x2 = Conv2D(num_priors[i] * 4, 3, padding='same', name= name + '_mbox_loc')(x)
        x2 = Conv2D(num_priors[i] * 4, 3, padding='same')(x)
        # x2 = SeparableConv2D(num_priors[i] * 4, 3, padding='same', use_bias=False, kernel_regularizer=l2(5e-4),name= name + '_mbox_loc')(x)


        # x2 = Flatten(name=name + '_mbox_loc_flat')(x2)
        x2 = Flatten()(x2)
        # x2 = activation_b5_mbox_loc_flat/Reshape:0 , shape=(Batch , 16)
        mbox_loc.append(x2)

    # mbox_loc/concat:0 , shape=(Batch, 34928)
    mbox_loc = Concatenate(axis=1, name='mbox_loc')(mbox_loc)
    # mbox_loc_final/Reshape:0, shape=(Batch, 8732, 4)
    mbox_loc = Reshape((-1, 4), name='mbox_loc_final')(mbox_loc)

    # mobx_conf/concat:0, shape=(Batch, 183372)
    mbox_conf = Concatenate(axis=1, name='mbox_conf')(mbox_conf)
    # mbox_conf_logits/Reshape:0, shape=(None, 8732, 21)
    mbox_conf = Reshape((-1, num_classes), name='mbox_conf_logits')(mbox_conf)
    # mbox_conf = Activation('softmax', name='mbox_conf_final')(mbox_conf)

    # predictions/concat:0, shape=(Batch, 8732, 25)
    predictions = Concatenate(axis=2, name='predictions', dtype=tf.float32)([mbox_loc, mbox_conf])

    return base.input, predictions


    # return base.input, source_layers, CLS_TIEMS[MODEL_NAME[base_model_name]]