from utils.priors import *

INPUT_SIZE = [300, 300]
iou_threshold = 0.5 # 0.5
center_variance = 0.1 # 0.1
size_variance = 0.2 # 0.2

# tensorflowjs_converter --input_format=keras ./checkpoints/save_model.h5 ./checkpoints/tfjs_model

# http://127.168.0.1:3000/predict-with-tfjs.html

"""
tensorflowjs_converter --input_format=tf_frozen_model --output_node_names='tf_op_layer_GatherV2_5' ./checkpoints/graph_model/saved_model.pb ./checkpoints/tfjs_frozen
tensorflowjs_converter --input_format=tf_frozen_model --output_node_names='Identity' --quantize_float16 ./checkpoints/new_tfjs_frozen/frozen_graph.pb ./checkpoints/test_frozen
tensorflowjs_converter --input_format=tf_frozen_model --output_node_names='Identity' ./checkpoints/new_tfjs_frozen/frozen_graph.pb ./checkpoints/test_frozen
"""

class TrainHyperParams:
    def __init__(self):
        self.optimizer_name = 'sgd'
        self.weight_decay = 0.0005
        self.learning_rate = 0.001
        self.momentum = 0.9
        self.lr_decay_steps = 200
        self.epochs = 200

    def setOptimizers(self):
        try:
            if self.optimizer_name == 'sgd':
                return tf.keras.optimizers.SGD(learning_rate=self.learning_rate, momentum=self.momentum)

            elif self.optimizer_name == 'adam':
                return tf.keras.optimizers.Adam(learning_rate=self.learning_rate)

            elif self.optimizer_name == 'rmsprop':
                return tf.keras.optimizers.RMSprop(learning_rate=self.learning_rate)
        except:
            print("check optimizers name")

def set_priorBox():
    return [
        Spec(38, 8, BoxSizes(15, 45), [2]),
        Spec(19, 16, BoxSizes(45, 80), [2, 3]),
        Spec(10, 32, BoxSizes(80, 140), [2, 3]),
        Spec(5, 64, BoxSizes(140, 213), [2, 3]),
        Spec(3, 100, BoxSizes(213, 264), [2]),
        Spec(1, 300, BoxSizes(264, 315), [2])
    ]


""" 
input 300x300
mobileNetv2 
baseChannel = 64

        Spec(38, 8, BoxSizes(9, 12), [2]),
        Spec(19, 16, BoxSizes(23, 33), [2, 3]),
        Spec(10, 32, BoxSizes(54, 108), [2, 3]),
        Spec(5, 64, BoxSizes(113, 134), [2, 3]),
        Spec(3, 100, BoxSizes(182, 226), [2]),
        Spec(1, 300, BoxSizes(264, 315), [2])
    
{'aeroplane': 0.7962120042370029,
 'bicycle': 0.817976018790892,
 'bird': 0.7216239435742163,
 'boat': 0.68508530315104,
 'bottle': 0.4759281174408228,
 'bus': 0.7668981164377825,
 'car': 0.8157523789742411,
 'cat': 0.8649200682170428,
 'chair': 0.5803828373773986,
 'cow': 0.7296196078538791,
 'diningtable': 0.6804619787584391,
 'dog': 0.8161611645577099,
 'horse': 0.8306460116884138,
 'motorbike': 0.8141273917050574,
 'person': 0.7991368157445411,
 'pottedplant': 0.49468281876967757,
 'sheep': 0.719652959462821,
 'sofa': 0.6850938417446898,
 'train': 0.8842312352470967,
 'tvmonitor': 0.6855982979434505}
mAP결과: 0.7332095455838107

"""


