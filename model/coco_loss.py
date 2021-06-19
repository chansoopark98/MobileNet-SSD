import tensorflow as tf
import numpy as np

def smooth_l1(labels, scores, sigma=1.0):
    diff = scores-labels
    abs_diff = tf.abs(diff)
    return tf.where(tf.less(abs_diff, 1/(sigma**2)), 0.5*(sigma*diff)**2, abs_diff-1/(2*sigma**2))

def hard_negative_mining(loss, labels, neg_pos_factor):
    pos_mask = labels > 0
    num_pos = tf.math.reduce_sum(tf.cast(pos_mask, tf.float32), axis=1, keepdims=True)
    num_neg = num_pos * neg_pos_factor
    loss = tf.where(pos_mask, tf.convert_to_tensor(np.NINF), loss)
    indexes = tf.argsort(loss, axis=1, direction='DESCENDING')
    orders = tf.argsort(indexes, axis=1)
    neg_mask = tf.cast(orders, tf.float32) < num_neg
    return tf.logical_or(pos_mask ,neg_mask)

def total_loss(y_true, y_pred, num_classes=81):
    labels = tf.argmax(y_true[:,:,:num_classes], axis=2) # batch, 13792
    confidence = y_pred[:,:,:num_classes] # batch, None, 81
    predicted_locations = y_pred[:,:,num_classes:] # None, None, 4
    gt_locations = y_true[:,:,num_classes:] # None, 13792, None
    neg_pos_ratio = 3.0

    loss = -tf.nn.log_softmax(confidence, axis=2)[:, :, 0]
    loss = tf.stop_gradient(loss)

    mask = hard_negative_mining(loss, labels, neg_pos_ratio)
    mask = tf.stop_gradient(mask)

    confidence = tf.boolean_mask(confidence, mask)

    classification_loss = tf.math.reduce_sum(tf.nn.sparse_softmax_cross_entropy_with_logits(
        logits = tf.reshape(confidence, [-1, num_classes]),
        labels = tf.boolean_mask(labels, mask)))


    pos_mask = labels > 0

    predicted_locations = tf.reshape(tf.boolean_mask(predicted_locations, pos_mask), [-1, 4])

    gt_locations = tf.reshape(tf.boolean_mask(gt_locations, pos_mask), [-1, 4])

    smooth_l1_loss = tf.math.reduce_sum(smooth_l1(scores=predicted_locations,labels=gt_locations))

    num_pos = tf.cast(tf.shape(gt_locations)[0], tf.float32)

    loc_loss = smooth_l1_loss / num_pos
    class_loss = classification_loss / num_pos
    mbox_loss = loc_loss + class_loss

    return mbox_loss

# def total_loss(y_true, y_pred, num_classes=81):
#     labels = tf.argmax(y_true[:,:,:num_classes], axis=2) # batch, 13792
#     confidence = y_pred[:,:,:num_classes] # batch, None, 81
#     predicted_locations = y_pred[:,:,num_classes:] # None, None, 4
#     gt_locations = y_true[:,:,num_classes:] # None, 13792, None
#     neg_pos_ratio = 3.0
#     # derived from cross_entropy=sum(log(p))
#     loss = -tf.nn.log_softmax(confidence, axis=2)[:, :, 0] # (None, None)
#     loss = tf.stop_gradient(loss)
#     # print(loss)
#
#     mask = hard_negative_mining(loss, labels, neg_pos_ratio) #mask = (None, 13792)
#     mask = tf.stop_gradient(mask)
#     # return mask
#     confidence = tf.boolean_mask(confidence, mask)
#
#     cross_entropy_loss = tf.nn.sparse_softmax_cross_entropy_with_logits(
#         logits=tf.reshape(confidence, [-1, num_classes]),
#         labels=tf.boolean_mask(labels, mask))
#
#     #classification_loss = tf.math.reduce_sum(tf.clip_by_value(cross_entropy_loss, 1e-10, tf.reduce_max(cross_entropy_loss)))
#     classification_loss = tf.math.reduce_sum(cross_entropy_loss)
#
#
#     # return classification_loss
#     pos_mask = labels > 0
#     predicted_locations = tf.reshape(tf.boolean_mask(predicted_locations, pos_mask), [-1, 4])
#     gt_locations = tf.reshape(tf.boolean_mask(gt_locations, pos_mask), [-1, 4])
#
#     # smooth_l1_loss = tf.math.reduce_sum(smooth_l1(scores=predicted_locations,labels=gt_locations))
#     huber_loss = tf.keras.losses.Huber(reduction='sum')(y_true=gt_locations , y_pred=predicted_locations)
#     smooth_l1_loss = tf.where(tf.math.is_inf(huber_loss), tf.cast(0., dtype=tf.float32),huber_loss)
#
#     num_pos = tf.cast(tf.shape(gt_locations)[0], tf.float32)
#     #num_pos = tf.where(tf.equal(num_pos, tf.cast(0, tf.float32)), tf.cast(0.001, tf.float32), num_pos)  ## << add
#     loc_loss = smooth_l1_loss / num_pos
#
#     class_loss = classification_loss / num_pos
#     # print(num_pos)
#     mbox_loss = loc_loss + class_loss
#
#     # # num_pos = pos_mask true 개수
#     # tf.print("   //  num_pos: ", num_pos,
#     #          "   //  class_loss: ", classification_loss,
#     #          "   //  loc_loss: ", smooth_l1_loss,
#     #          "   //  gt_location: ", tf.reduce_sum(tf.where(tf.math.is_nan(gt_locations),1,0)),
#     #          "   //  predict_location: ", tf.reduce_sum(tf.where(tf.math.is_nan(predicted_locations),1,0)),
#     #          "   //  y_pred_nan: ", tf.reduce_sum(tf.where(tf.math.is_nan(y_pred),1,0)),
#     #          "   //  y_pred_inf: ", tf.reduce_sum(tf.where(tf.math.is_inf(y_pred),1,0)),
#     #          "   //  huber_loss: ", huber_loss,
#     #          output_stream = sys.stdout)
#
#     # tf.estimator.NanTensorHook(
#     #     mbox_loss, fail_on_nan_loss=True
#     # )
#
#     # tf.print("   //  num_pos: ", num_pos,
#     #          "   //  class_loss: ", classification_loss,
#     #          "   //  loc_loss: ", smooth_l1_loss,
#     #          "   //  pos_mask: ", pos_mask,
#     #          output_stream = sys.stdout)
#
#
#     # mbox_loss = tf.where(tf.math.is_nan(mbox_loss), tf.constant(1., dtype=tf.float32), mbox_loss)
#
#     return mbox_loss
