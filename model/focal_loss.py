import sys

import tensorflow as tf
import numpy as np

def smooth_l1(labels, scores, sigma=1.0):
    diff = scores-labels
    abs_diff = tf.abs(diff)
    return tf.where(tf.less(abs_diff, 1/(sigma**2)), 0.5*(sigma*diff)**2, abs_diff-1/(2*sigma**2))

def hard_negative_mining(loss, labels, neg_pos_ratio):
    pos_mask = labels > 0
    num_pos = tf.math.reduce_sum(tf.cast(pos_mask, tf.float32), axis=1, keepdims=True)
    num_neg = num_pos * neg_pos_ratio

    loss = tf.where(pos_mask, tf.convert_to_tensor(np.NINF), loss)

    indexes = tf.argsort(loss, axis=1, direction='DESCENDING')
    orders = tf.argsort(indexes, axis=1)
    neg_mask = tf.cast(orders, tf.float32) < num_neg

    return tf.logical_or(pos_mask ,neg_mask)


def total_loss(y_true, y_pred, num_classes=21):
    labels = tf.argmax(y_true[:,:,:num_classes], axis=2)
    confidence = y_pred[:,:,:num_classes]
    predicted_locations = y_pred[:,:,num_classes:]
    gt_locations = y_true[:,:,num_classes:]
    neg_pos_ratio = 3.0
    loss = -tf.nn.log_softmax(confidence, axis=2)[:, :, 0]
    loss = tf.stop_gradient(loss)

    mask = hard_negative_mining(loss, labels, neg_pos_ratio)
    mask = tf.stop_gradient(mask) # neg sample 마스크

    confidence = tf.boolean_mask(confidence, mask)

    ce_logit = tf.reshape(confidence, [-1, num_classes])
    ce_label = tf.boolean_mask(labels, mask)
    weight_label = tf.ones_like(ce_label)
    tf.print("ce_label\n", ce_label, output_stream=sys.stdout, summarize=-1)
    # calc classification loss
    cls_loss = tf.nn.sparse_softmax_cross_entropy_with_logits(logits = ce_logit, labels = ce_label)
    tf.print("cls_loss\n", cls_loss, output_stream=sys.stdout, summarize=-1)
    classification_loss = tf.math.reduce_sum(cls_loss)
    pos_mask = labels > 0
    predicted_locations = tf.reshape(tf.boolean_mask(predicted_locations, pos_mask), [-1, 4])
    gt_locations = tf.reshape(tf.boolean_mask(gt_locations, pos_mask), [-1, 4])
    # calc localization loss
    smooth_l1_loss = tf.math.reduce_sum(smooth_l1(scores=predicted_locations,labels=gt_locations))
    num_pos = tf.cast(tf.shape(gt_locations)[0], tf.float32)
    # divide num_pos objects
    loc_loss = smooth_l1_loss / num_pos
    class_loss = classification_loss / num_pos
    mbox_loss = loc_loss + class_loss
    return mbox_loss