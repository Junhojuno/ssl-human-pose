import tensorflow as tf

from src.decode import heatmap_to_coordinate


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count if self.count != 0 else 0


def accuracy(true_hms, pred_hms, thr: float = 0.5):
    H = tf.shape(true_hms)[1]
    W = tf.shape(true_hms)[2]

    norm = \
        tf.ones((tf.shape(pred_hms)[0], 2)) * \
        tf.convert_to_tensor([H, W], tf.float32) / 10
    norm = tf.expand_dims(norm, axis=1)

    true_kpts = heatmap_to_coordinate(true_hms)  # (B, K, 2)
    pred_kpts = heatmap_to_coordinate(pred_hms)

    true_kpts = tf.cast(true_kpts, tf.float32)
    pred_kpts = tf.cast(pred_kpts, tf.float32)

    dists, valid_mask, n_valids = cal_dist(true_kpts, pred_kpts, norm)
    n_corrects = tf.math.reduce_sum(
        tf.cast(dists < thr, tf.float32) * valid_mask
    )
    acc = tf.math.divide_no_nan(n_corrects, n_valids)
    return acc, n_valids


def cal_dist(target, pred, norm):
    valid_mask = tf.math.reduce_all(target > 1, axis=-1)
    dists = tf.math.reduce_euclidean_norm(
        (pred / norm) - (target / norm),
        axis=-1
    )  # (B, K)
    dists = tf.where(valid_mask, dists, -tf.ones_like(dists))
    valid_mask = tf.cast(valid_mask, tf.float32)
    n_valids = tf.math.reduce_sum(valid_mask)
    return dists, valid_mask, n_valids
