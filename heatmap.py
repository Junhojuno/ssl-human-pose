"""heatmap and offsetmap generation"""
from typing import Optional, List, Tuple
import tensorflow as tf


def extract_size(shape):
    """it is same with tf.reverse(shape, [0])"""
    wh = tf.convert_to_tensor([shape[1], shape[0]])
    wh = tf.cast(wh, tf.float32)
    return wh


def rescale_keypoints_to_heatmap_size(
    keypoints,
    input_shape: List,
    output_shape: List
):
    """return heatmap-sized keypoint coordinates"""
    input_wh = extract_size(input_shape)
    output_wh = extract_size(output_shape)

    kp_xy = keypoints[..., :2]
    vis = keypoints[..., 2:]

    kp_xy = kp_xy / input_wh * output_wh
    # kp_xy_int = tf.floor(kp_xy)
    keypoints = tf.concat([kp_xy, vis], axis=-1)
    return keypoints


def batch_gen_heatmap(
    keypoints: tf.Tensor,
    output_shape: List,
    num_keypoints: int = 17,
    sigma: float = 2.0,
) -> tf.Tensor:
    """heatmaps과 offsetmaps을 생성하는 함수

    keypoints를 따로 floor하지 않고 그대로 사용하여 heatmap을 만들어낸다.

    Args:
        keypoints (tf.Tensor): input H/W 크기의 값 범위를 가지고 있는 키포인트 (B, K, 3)
        output_shape (Optional[List], optional): heatmap H/W.
        num_keypoints (Optional[int], optional): 학습할 키포인트 개수.
            Defaults to 17.
        sigma (Optional[float], optional): heatmap 만들때 사용할 sigma.
            Defaults to 2.0.
        locref_stdev (Optional[float], optional): offsetmap에 나눠줄 값.
            Defaults to 1.0.
        kpd (Optional[float], optional):
            일정 거리 내의 heatmap 값들로만 제한하기 위한 threshold.
            Defaults to 4..

    Returns:
        [tf.Tensor, tf.Tensor, tf.Tensor]:
            heatmaps, offsetmaps, and binary mask
    """
    heatmap_height, heatmap_width = output_shape[0], output_shape[1]

    keypoints = rescale_keypoints_to_heatmap_size(
        keypoints,
        [output_shape[0] * 4, output_shape[1] * 4],
        output_shape
    )

    xx, yy = tf.meshgrid(tf.range(heatmap_width), tf.range(heatmap_height))
    grid_xy = tf.stack([xx, yy], axis=-1)
    grid_xy = tf.cast(grid_xy, tf.float32)  # (H/R, W/R, 2)
    grid_xy = tf.expand_dims(grid_xy, axis=2)

    dist = tf.reshape(
        keypoints[..., :2], [-1, 1, 1, num_keypoints, 2]
    ) - grid_xy
    heatmaps = tf.exp(
        -((dist[..., 0] / sigma) ** 2) / 2 - ((dist[..., 1] / sigma) ** 2) / 2
    )
    # np.finfo(np.float32).eps == 1.1920929e-07
    maxvals = tf.math.reduce_max(heatmaps)
    mask = tf.cast(heatmaps > (1.1920929e-07 * maxvals), tf.float32)
    heatmaps *= mask

    # invisible keypoints' heatmap should be zeros!
    valid = tf.cast(keypoints[:, :, -1] > 0, tf.float32)
    valid = tf.reshape(valid, [-1, 1, 1, num_keypoints])
    heatmaps *= valid
    return heatmaps


def batch_gen_heatmap_offsetmap(
    keypoints: tf.Tensor,
    output_shape: List,
    num_keypoints: int = 17,
    sigma: float = 2.0,
    locref_stdev: float = 1.0,
    kpd: float = 4.0
) -> Tuple[tf.Tensor, tf.Tensor, tf.Tensor]:
    """heatmaps과 offsetmaps을 생성하는 함수

    keypoints를 따로 floor하지 않고 그대로 사용하여 heatmap을 만들어낸다.

    Args:
        keypoints (tf.Tensor): input H/W 크기의 값 범위를 가지고 있는 키포인트 (B, K, 3)
        output_shape (Optional[List], optional): heatmap H/W.
        num_keypoints (Optional[int], optional): 학습할 키포인트 개수.
            Defaults to 17.
        sigma (Optional[float], optional): heatmap 만들때 사용할 sigma.
            Defaults to 2.0.
        locref_stdev (Optional[float], optional): offsetmap에 나눠줄 값.
            Defaults to 1.0.
        kpd (Optional[float], optional):
            일정 거리 내의 heatmap 값들로만 제한하기 위한 threshold.
            Defaults to 4..

    Returns:
        [tf.Tensor, tf.Tensor, tf.Tensor]:
            heatmaps, offsetmaps, and binary mask
    """
    heatmap_height, heatmap_width = output_shape[0], output_shape[1]

    keypoints = rescale_keypoints_to_heatmap_size(
        keypoints,
        [output_shape[0] * 4, output_shape[1] * 4],
        output_shape
    )

    xx, yy = tf.meshgrid(tf.range(heatmap_width), tf.range(heatmap_height))
    grid_xy = tf.stack([xx, yy], axis=-1)
    grid_xy = tf.cast(grid_xy, tf.float32)  # (H/R, W/R, 2)
    grid_xy = tf.expand_dims(grid_xy, axis=2)

    dist = tf.reshape(
        keypoints[..., :2], [-1, 1, 1, num_keypoints, 2]
    ) - grid_xy
    heatmaps = tf.exp(
        -((dist[..., 0] / sigma) ** 2) / 2 - ((dist[..., 1] / sigma) ** 2) / 2
    )
    # np.finfo(np.float32).eps == 1.1920929e-07
    maxvals = tf.math.reduce_max(heatmaps)
    mask = tf.cast(heatmaps > (1.1920929e-07 * maxvals), tf.float32)
    heatmaps *= mask

    # invisible keypoints' heatmap should be zeros!
    valid = tf.cast(keypoints[:, :, -1] > 0, tf.float32)
    valid = tf.reshape(valid, [-1, 1, 1, num_keypoints])
    heatmaps *= valid

    offsetmaps = dist / locref_stdev
    in_range =\
        tf.math.reduce_sum(
            dist**2, axis=-1, keepdims=True
        ) <= (kpd ** 2)
    mask_01 = tf.where(in_range, 1., 0.)
    offsetmaps *= mask_01

    return heatmaps, offsetmaps, mask_01
