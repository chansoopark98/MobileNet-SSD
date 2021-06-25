from tensorflow.python.framework.convert_to_constants import convert_variables_to_constants_v2
from model.model_builder import model_build
from config import *
import argparse

tf.keras.backend.clear_session()

parser = argparse.ArgumentParser()
parser.add_argument("--checkpoint_dir", type=str,   help="모델 저장 디렉토리 설정", default='./checkpoints/voc_0621.h5')
args = parser.parse_args()

IMAGE_SIZE = INPUT_SIZE
TRAIN_MODE = 'voc'
MODEL_NAME = 'B0'
CHECKPOINT_DIR = args.checkpoint_dir

specs = set_priorBox()
priors = create_priors_boxes(specs, IMAGE_SIZE[0])
target_transform = MatchingPriors(priors, center_variance, size_variance, iou_threshold)


model = model_build(TRAIN_MODE, target_transform=target_transform, train=False, image_size=IMAGE_SIZE)

print("모델 가중치 로드...")
model.load_weights(CHECKPOINT_DIR)


#path of the directory where you want to save your model
frozen_out_path = './checkpoints/new_tfjs_frozen'
# name of the .pb file
frozen_graph_filename = "frozen_graph"
# Convert Keras model to ConcreteFunction
full_model = tf.function(lambda x: model(x))
full_model = full_model.get_concrete_function(
    tf.TensorSpec(model.inputs[0].shape, model.inputs[0].dtype))
# Get frozen ConcreteFunction
frozen_func = convert_variables_to_constants_v2(full_model)
frozen_func.graph.as_graph_def()
layers = [op.name for op in frozen_func.graph.get_operations()]
print("-" * 60)
print("Frozen model layers: ")
for layer in layers:
    print(layer)
print("-" * 60)
print("Frozen model inputs: ")
print(frozen_func.inputs)
print("Frozen model outputs: ")
print(frozen_func.outputs)
# Save frozen graph to disk
tf.io.write_graph(graph_or_graph_def=frozen_func.graph,
                  logdir=frozen_out_path,
                  name=f"{frozen_graph_filename}.pb",
                  as_text=False)
# Save its text representation
tf.io.write_graph(graph_or_graph_def=frozen_func.graph,
                  logdir=frozen_out_path,
                  name=f"{frozen_graph_filename}.pbtxt",
                  as_text=True)

