import tensorflow as tf


def heatmap_to_coordinate(heatmaps):
    B = tf.shape(heatmaps)[0]
    W = tf.shape(heatmaps)[2]
    K = tf.shape(heatmaps)[-1]
    kpt_idxs = tf.math.argmax(
        tf.reshape(heatmaps, [B, -1, K]),
        axis=1,
        output_type=tf.int32
    )  # (B, K)
    # div / mod by width
    kpt_y = tf.math.floordiv(kpt_idxs, W)  # (B, K)
    kpt_x = tf.math.floormod(kpt_idxs, W)  # (B, K)
    coordinates = tf.stack([kpt_x, kpt_y], axis=-1)
    return coordinates
