import tensorflow as tf


def extract_visibility_mask(true_heatmaps):
    """
    vis > 0인 keypoint heatmaps 살리기위한 mask 생성하는 함수

    heatmap 생성할 때 floor(keypoints)를 하지 않아 최대값이 1이 아님
    각 keypoint heatmap의 최대값이 0보다 크면 visible한 것으로 판단.
    """
    maxvals = tf.math.reduce_max(true_heatmaps, axis=(1, 2))
    visible_masks = tf.math.greater(maxvals, 0)
    visible_masks = tf.cast(visible_masks, tf.float32)
    return visible_masks


def reshape_and_transpose(heatmaps):
    B = tf.shape(heatmaps)[0]
    K = tf.shape(heatmaps)[-1]
    heatmaps = tf.reshape(heatmaps, [B, -1, K])
    heatmaps = tf.transpose(heatmaps, [0, 2, 1])
    return heatmaps


def keypoint_loss(
    true_hms: tf.Tensor,
    pred_hms: tf.Tensor,
):
    K = tf.shape(true_hms)[-1]

    visibility = extract_visibility_mask(true_hms)

    true_hms = reshape_and_transpose(true_hms)
    pred_hms = reshape_and_transpose(pred_hms)

    hm_size_splits = tf.ones([K], dtype=tf.int32)

    # list of (B, 1, N) shaped tensors
    true_hms = tf.split(true_hms, hm_size_splits, axis=1)
    pred_hms = tf.split(pred_hms, hm_size_splits, axis=1)

    visibility = tf.split(visibility, hm_size_splits, axis=1)  # (B, 1)

    loss = 0.0
    for (true_hm, pred_hm, vis) in zip(true_hms, pred_hms, visibility):

        loss += 0.5 * mse_loss(
            true_hm * tf.expand_dims(vis, axis=-1),
            pred_hm * tf.expand_dims(vis, axis=-1)
        )

    return loss / tf.cast(K, loss.dtype)


def mse_loss(y_true, y_pred):
    loss = tf.math.squared_difference(y_true, y_pred)
    loss = tf.math.reduce_mean(loss, axis=(1, 2))
    return loss


def pose_dual_loss_fn(true_hm, pred_hms):
    sup_pred_hm1, sup_pred_hm2, \
        unsup_true_hm1, unsup_true_hm2, \
        unsup_pred_hm1, unsup_pred_hm2 = pred_hms

    loss_sup = \
        0.5 * keypoint_loss(true_hm, sup_pred_hm1) + \
        0.5 * keypoint_loss(true_hm, sup_pred_hm2)

    loss_unsup = \
        keypoint_loss(unsup_true_hm1, unsup_pred_hm2) + \
        keypoint_loss(unsup_true_hm2, unsup_pred_hm1)

    loss = loss_sup + loss_unsup  # (B,)
    return loss
