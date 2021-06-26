from tensorflow.keras.callbacks import ReduceLROnPlateau, ModelCheckpoint
from tensorflow.keras.mixed_precision import experimental as mixed_precision
from callbacks import Scalar_LR
from metrics import CreateMetrics
from config import *
from utils.load_datasets import GenerateDatasets
from model.model_builder import model_build
from model.loss import Total_loss
import argparse
import time
import os

tf.keras.backend.clear_session()
policy = mixed_precision.Policy('mixed_float16', loss_scale=1024)
mixed_precision.set_policy(policy)

parser = argparse.ArgumentParser()
parser.add_argument("--batch_size",     type=int,   help="배치 사이즈값 설정", default=128)
parser.add_argument("--epoch",          type=int,   help="에폭 설정", default=300)
parser.add_argument("--lr",             type=float, help="Learning rate 설정", default=0.001)
parser.add_argument("--weight_decay",   type=float, help="Weight Decay 설정", default=0.0005)
parser.add_argument("--model_name",     type=str,   help="저장될 모델 이름",
                    default=str(time.strftime('%m%d', time.localtime(time.time()))))
parser.add_argument("--dataset_dir",    type=str,   help="데이터셋 다운로드 디렉토리 설정", default='./datasets/')
parser.add_argument("--checkpoint_dir", type=str,   help="모델 저장 디렉토리 설정", default='./checkpoints/')
parser.add_argument("--tensorboard_dir",  type=str,   help="텐서보드 저장 경로", default='tensorboard')
parser.add_argument("--backbone_model", type=str,   help="EfficientNet 모델 설정", default='B0')
parser.add_argument("--train_dataset",  type=str,   help="학습에 사용할 dataset 설정 coco or voc", default='voc')
parser.add_argument("--use_weightDecay",  type=bool,  help="weightDecay 사용 유무", default=True)



args = parser.parse_args()
WEIGHT_DECAY = args.weight_decay
BATCH_SIZE = args.batch_size
EPOCHS = args.epoch
base_lr = args.lr
SAVE_MODEL_NAME = args.model_name
DATASET_DIR = args.dataset_dir
CHECKPOINT_DIR = args.checkpoint_dir
TENSORBOARD_DIR = args.tensorboard_dir
MODEL_NAME = args.backbone_model
TRAIN_MODE = args.train_dataset
IMAGE_SIZE = INPUT_SIZE
USE_WEIGHT_DECAY = args.use_weightDecay

os.makedirs(DATASET_DIR, exist_ok=True)
os.makedirs(CHECKPOINT_DIR, exist_ok=True)

specs = set_priorBox()

priors = create_priors_boxes(specs, IMAGE_SIZE[0])
TARGET_TRANSFORM = MatchingPriors(priors, center_variance, size_variance, iou_threshold)

# Create Dataset
dataset_config = GenerateDatasets(TRAIN_MODE, DATASET_DIR, IMAGE_SIZE, BATCH_SIZE, TARGET_TRANSFORM)

# Set loss function
loss = Total_loss(dataset_config.num_classes)

print("백본 EfficientNet{0} .".format(MODEL_NAME))

steps_per_epoch = dataset_config.number_train // BATCH_SIZE
validation_steps = dataset_config.number_test // BATCH_SIZE
print("학습 배치 개수:", steps_per_epoch)
print("검증 배치 개수:", validation_steps)

metrics = CreateMetrics(dataset_config.num_classes)
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.9, patience=3, min_lr=1e-5, verbose=1)

checkpoint = ModelCheckpoint(CHECKPOINT_DIR + TRAIN_MODE + '_' + SAVE_MODEL_NAME + '.h5',
                                 monitor='val_loss', save_best_only=True, save_weights_only=True, verbose=1)
testCallBack = Scalar_LR('test', TENSORBOARD_DIR)
tensorboard = tf.keras.callbacks.TensorBoard(log_dir=TENSORBOARD_DIR, write_graph=True, write_images=True)


polyDecay = tf.keras.optimizers.schedules.PolynomialDecay(initial_learning_rate=base_lr,
                                                          decay_steps=200,
                                                          end_learning_rate=0.0001, power=0.5)
lr_scheduler = tf.keras.callbacks.LearningRateScheduler(polyDecay)

optimizer = tf.keras.optimizers.SGD(learning_rate=base_lr, momentum=0.9)
optimizer = mixed_precision.LossScaleOptimizer(optimizer, loss_scale='dynamic')  # tf2.4.1 이전

callback = [checkpoint, reduce_lr , lr_scheduler, testCallBack, tensorboard]


mirrored_strategy = tf.distribute.MirroredStrategy(cross_device_ops=tf.distribute.HierarchicalCopyAllReduce())
print("Number of devices: {}".format(mirrored_strategy.num_replicas_in_sync))

with mirrored_strategy.scope(): # if use single gpu > with tf.device('/device:GPU:0'):
    model = model_build(TRAIN_MODE, train=True, image_size=IMAGE_SIZE)

    model.compile(
        optimizer=optimizer,
        loss=loss.total_loss,
        metrics=[metrics.precision, metrics.recall, metrics.cross_entropy, metrics.localization]
    )

    model.summary()

    history = model.fit(dataset_config.training_dataset,
            validation_data=dataset_config.validation_dataset,
            steps_per_epoch=steps_per_epoch,
            validation_steps=validation_steps,
            epochs=EPOCHS,
            callbacks=callback)

    model.save('./checkpoints/save_model.h5', True, True, 'h5')

