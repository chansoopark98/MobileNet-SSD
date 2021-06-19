from utils.priors import *

iou_threshold = 0.5 # 0.5
center_variance = 0.1 # 0.1
size_variance = 0.2 # 0.2

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
        Spec(38, 8, BoxSizes(30, 60), [2]),
        Spec(19, 16, BoxSizes(60, 111), [2, 3]),
        Spec(10, 32, BoxSizes(111, 162), [2, 3]),
        Spec(5, 64, BoxSizes(162, 213), [2, 3]),
        Spec(3, 100, BoxSizes(213, 264), [2]),
        Spec(1, 300, BoxSizes(264, 315), [2])
    ]


