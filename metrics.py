import tensorflow as tf
import tensorflow.keras.backend as K
from tensorflow.keras.metrics import sparse_categorical_crossentropy


class CreateMetrics:
    def __init__(self, classes):
        self.num_classes = classes



    def recall(self, y_target, y_pred):
        y_target_yn = K.round(K.clip(y_target, 0, 1)) # 실제값을 0(Negative) 또는 1(Positive)로 설정한다
        y_pred_yn = K.round(K.clip(y_pred, 0, 1)) # 예측값을 0(Negative) 또는 1(Positive)로 설정한다

        # True Positive는 실제 값과 예측 값이 모두 1(Positive)인 경우이다
        count_true_positive = K.sum(y_target_yn * y_pred_yn)

        # (True Positive + False Negative) = 실제 값이 1(Positive) 전체
        count_true_positive_false_negative = K.sum(y_target_yn)

        # Recall =  (True Positive) / (True Positive + False Negative)
        # K.epsilon()는 'divide by zero error' 예방차원에서 작은 수를 더한다
        recall = count_true_positive / (count_true_positive_false_negative + K.epsilon())

        # return a single tensor value
        return recall

    def precision(self, y_target, y_pred):
        # clip(t, clip_value_min, clip_value_max) : clip_value_min~clip_value_max 이외 가장자리를 깎아 낸다
        # round : 반올림한다
        y_pred_yn = K.round(K.clip(y_pred, 0, 1)) # 예측값을 0(Negative) 또는 1(Positive)로 설정한다
        y_target_yn = K.round(K.clip(y_target, 0, 1)) # 실제값을 0(Negative) 또는 1(Positive)로 설정한다

        # True Positive는 실제 값과 예측 값이 모두 1(Positive)인 경우이다
        count_true_positive = K.sum(y_target_yn * y_pred_yn)

        # (True Positive + False Positive) = 예측 값이 1(Positive) 전체
        count_true_positive_false_positive = K.sum(y_pred_yn)

        # Precision = (True Positive) / (True Positive + False Positive)
        # K.epsilon()는 'divide by zero error' 예방차원에서 작은 수를 더한다
        precision = count_true_positive / (count_true_positive_false_positive + K.epsilon())

        # return a single tensor value
        return precision


    def f1score(self, y_target, y_pred):
        _recall = self.recall(y_target, y_pred)
        _precision = self.precision(y_target, y_pred)
        # K.epsilon()는 'divide by zero error' 예방차원에서 작은 수를 더한다
        _f1score = (2 * _recall * _precision) / (_recall + _precision + K.epsilon())

        # return a single tensor value
        return _f1score

    def cross_entropy(self, y_target, y_pred):
        return sparse_categorical_crossentropy(y_pred=y_pred[:,:,:self.num_classes], y_true=tf.argmax(y_target[:,:,:self.num_classes], axis=2))

    def smooth_l1(self, labels, scores, sigma=1.0):
        diff = scores - labels
        abs_diff = tf.abs(diff)
        return tf.where(tf.less(abs_diff, 1 / (sigma ** 2)), 0.5 * (sigma * diff) ** 2, abs_diff - 1 / (2 * sigma ** 2))

    def localization(self, y_true, y_pred):
        labels = tf.argmax(y_true[:, :, :self.num_classes], axis=2)  # batch, 13792
        predicted_locations = y_pred[:, :, self.num_classes:]  # None, None, 4
        gt_locations = y_true[:, :, self.num_classes:]  # None, 13792, None

        pos_mask = labels > 0

        predicted_locations = tf.reshape(tf.boolean_mask(predicted_locations, pos_mask), [-1, 4])

        gt_locations = tf.reshape(tf.boolean_mask(gt_locations, pos_mask), [-1, 4])

        num_pos = tf.cast(tf.shape(gt_locations)[0], tf.float32)
        return tf.math.reduce_sum(self.smooth_l1(scores=predicted_locations, labels=gt_locations))/num_pos
        # num_pos = tf.cast(tf.shape(gt_locations)[0], tf.float32)
        # loc_loss = smooth_l1_loss / num_pos

