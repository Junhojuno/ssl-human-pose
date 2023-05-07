from typing import List
import math

import tensorflow as tf
from tensorflow.python.keras import layers
from tensorflow.python.keras import Model, Input

from utils import generate_affine_matrix


def get_single_theta(
    rf: float,
    sf: float
):
    angle = tf.cond(
        tf.math.less_equal(tf.random.uniform([]), 0.8),
        lambda: tf.clip_by_value(
            tf.random.normal([]) * rf,
            -2 * rf,
            2 * rf
        ),
        lambda: 0.
    )

    radian = angle / 180 * tf.constant(math.pi, dtype=tf.float32)
    theta = [
        [sf * tf.math.cos(radian), sf * tf.math.sin(radian), 0.],
        [-sf * tf.math.sin(radian), sf * tf.math.cos(radian), 0.]
    ]
    return theta


def get_batch_theta(
    batch_size: int,
    rf: float,
    sf: float
):
    thetas = tf.stack(
        [get_single_theta(rf, sf) for _ in tf.range(batch_size)],
        axis=0
    )
    return thetas


def affine_grid(thetas, height, width):
    """
    https://github.com/kevinzakka/spatial-transformer-network/blob/master/stn/transformer.py
    """
    B = tf.shape(thetas)[0]

    # create normalized 2D grid
    x = tf.linspace(-1.0, 1.0, width)
    y = tf.linspace(-1.0, 1.0, height)
    x_t, y_t = tf.meshgrid(x, y)

    # flatten
    x_t_flat = tf.reshape(x_t, [-1])
    y_t_flat = tf.reshape(y_t, [-1])

    # reshape to [x_t, y_t , 1] - (homogeneous form)
    ones = tf.ones_like(x_t_flat)
    sampling_grid = tf.stack([x_t_flat, y_t_flat, ones])  # (3, h*w)

    # repeat grid num_batch times: (3, h*w) -> (B, 3, h*w)
    sampling_grid = tf.expand_dims(sampling_grid, axis=0)
    sampling_grid = tf.tile(sampling_grid, tf.stack([B, 1, 1]))

    # cast to float32 (required for matmul)
    thetas = tf.cast(thetas, tf.float32)
    sampling_grid = tf.cast(sampling_grid, tf.float32)

    # transform the sampling grid -> batch multiply
    # (B, 2, 3) @ (B, 3, h*w)
    # output shape : (B, 2, h*w)
    batch_grids = tf.matmul(thetas, sampling_grid)

    batch_grids = tf.reshape(batch_grids, [B, 2, height, width])
    return batch_grids


def get_pixel_value(img, x, y):
    """
    Utility function to get pixel value for coordinate
    vectors x and y from a  4D tensor image.

    Input
    -----
    - img: tensor of shape (B, H, W, C)
    - x: flattened tensor of shape (B*H*W,)
    - y: flattened tensor of shape (B*H*W,)

    Returns
    -------
    - output: tensor of shape (B, H, W, C)
    """
    shape = tf.shape(x)
    batch_size = shape[0]
    height = shape[1]
    width = shape[2]

    batch_idx = tf.range(0, batch_size)
    batch_idx = tf.reshape(batch_idx, (batch_size, 1, 1))
    b = tf.tile(batch_idx, (1, height, width))

    indices = tf.stack([b, y, x], 3)

    return tf.gather_nd(img, indices)


def bilinear_sampler(img, x, y):
    """
    Performs bilinear sampling of the input images according to the
    normalized coordinates provided by the sampling grid. Note that
    the sampling is done identically for each channel of the input.

    To test if the function works properly, output image should be
    identical to input image when theta is initialized to identity
    transform.

    Input
    -----
    - img: batch of images in (B, H, W, C) layout.
    - grid: x, y which is the output of affine_grid_generator.

    Returns
    -------
    - out: interpolated images according to grids. Same size as grid.
    """
    H = tf.shape(img)[1]
    W = tf.shape(img)[2]
    max_y = tf.cast(H - 1, 'int32')
    max_x = tf.cast(W - 1, 'int32')
    zero = tf.zeros([], dtype='int32')

    # rescale x and y to [0, W-1/H-1]
    x = tf.cast(x, 'float32')
    y = tf.cast(y, 'float32')
    x = 0.5 * ((x + 1.0) * tf.cast(max_x-1, 'float32'))
    y = 0.5 * ((y + 1.0) * tf.cast(max_y-1, 'float32'))

    # grab 4 nearest corner points for each (x_i, y_i)
    x0 = tf.cast(tf.floor(x), 'int32')
    x1 = x0 + 1
    y0 = tf.cast(tf.floor(y), 'int32')
    y1 = y0 + 1

    # clip to range [0, H-1/W-1] to not violate img boundaries
    x0 = tf.clip_by_value(x0, zero, max_x)
    x1 = tf.clip_by_value(x1, zero, max_x)
    y0 = tf.clip_by_value(y0, zero, max_y)
    y1 = tf.clip_by_value(y1, zero, max_y)

    # get pixel value at corner coords
    Ia = get_pixel_value(img, x0, y0)
    Ib = get_pixel_value(img, x0, y1)
    Ic = get_pixel_value(img, x1, y0)
    Id = get_pixel_value(img, x1, y1)

    # recast as float for delta calculation
    x0 = tf.cast(x0, 'float32')
    x1 = tf.cast(x1, 'float32')
    y0 = tf.cast(y0, 'float32')
    y1 = tf.cast(y1, 'float32')

    # calculate deltas
    wa = (x1-x) * (y1-y)
    wb = (x1-x) * (y-y0)
    wc = (x-x0) * (y1-y)
    wd = (x-x0) * (y-y0)

    # add dimension for addition
    wa = tf.expand_dims(wa, axis=3)
    wb = tf.expand_dims(wb, axis=3)
    wc = tf.expand_dims(wc, axis=3)
    wd = tf.expand_dims(wd, axis=3)

    # compute output
    out = tf.add_n([wa*Ia, wb*Ib, wc*Ic, wd*Id])

    return out


def mask_joint(image, joints, MASK_JOINT_NUM=4):
    # N,J,2 joints
    N, J = joints.shape[:2]
    _, _, width, height = image.shape
    re_joints = joints[:, :, :2] + torch.randn((N, J, 2)).cuda()*10
    re_joints = re_joints.int()
    size = torch.randint(10, 20, (N, J, 2)).int().cuda()

    x0 = re_joints[:, :, 0]-size[:, :, 0]
    y0 = re_joints[:, :, 1]-size[:, :, 1]

    x1 = re_joints[:, :, 0]+size[:, :, 0]
    y1 = re_joints[:, :, 1]+size[:, :, 1]

    torch.clamp_(x0, 0, width)
    torch.clamp_(x1, 0, width)
    torch.clamp_(y0, 0, height)
    torch.clamp_(y1, 0, height)

    for i in range(N):
        # num = np.random.randint(MASK_JOINT_NUM)
        # ind = np.random.choice(J, num)
        ind = np.random.choice(J, MASK_JOINT_NUM)
        for j in ind:
            image[i, :, y0[i, j]:y1[i, j], x0[i, j]:x1[i, j]] = 0
    return image


def mask_joints(image, keypoints, n_masks: int = 4):
    B = tf.shape(keypoints)[0]
    K = tf.shape(keypoints)[1]
    height = tf.shape(image)[1]
    width = tf.shape(image)[2]

    re_keypoints = keypoints[..., :2] + tf.random.normal([B, K, 2]) * 10.0
    re_keypoints = tf.cast(re_keypoints, tf.int32)
    size = tf.random.uniform([B, K, 2], 10, 20, tf.int32)

    tl = re_keypoints - size
    br = re_keypoints + size


class PoseCoTrain(Model):

    def __init__(
        self,
        heavy_model,
        lite_model,
        scale_factor,
        rotation_factor
    ):
        super().__init__()
        self.heavy_model = heavy_model
        self.lite_model = lite_model
        self.sf = scale_factor
        self.rf = rotation_factor

    def call(self, sup_x, unsup_x):
        B = tf.shape(sup_x)[0]
        H = tf.shape(sup_x)[1]
        W = tf.shape(sup_x)[2]

        sup_pred_hm1 = self.heavy_model(sup_x, training=True)
        sup_pred_hm2 = self.lite_model(sup_x, training=True)

        # Teachers
        # Easy Augmentation
        unsup_hm1 = self.heavy_model(unsup_x, training=False)
        unsup_hm2 = self.lite_model(unsup_x, training=False)

        # TODO: mask joints (joint cutout)

        # ----

        # Apply Affine Transformation again for hard augmentation
        # make easy prediction be hard one -> To use GT for unlabeled
        # unlabeled 데이터의 GT로 활용
        # NOTE: do not backprop (no gradients passing)
        thetas = get_batch_theta(B, self.rf, self.sf)
        grids = affine_grid(thetas, H, W)

        unsup_x_trans1 = bilinear_sampler(
            unsup_x,
            grids[:, 0, ...],
            grids[:, 1, ...]
        )
        unsup_x_trans2 = bilinear_sampler(
            unsup_x,
            grids[:, 0, ...],
            grids[:, 1, ...]
        )
        # transforms easy prediction to hard one
        hm_grids = affine_grid(
            thetas,
            tf.shape(sup_pred_hm1)[1],
            tf.shape(sup_pred_hm1)[2]
        )
        hm_grids = tf.cast(hm_grids, tf.float32)
        unsup_true_hm1 = bilinear_sampler(
            unsup_hm1,
            hm_grids[:, 0, ...],
            hm_grids[:, 1, ...]
        )
        unsup_true_hm2 = bilinear_sampler(
            unsup_hm2,
            hm_grids[:, 0, ...],
            hm_grids[:, 1, ...]
        )

        # Students
        # Hard Augmentation이 적용된 sample 예측
        # unlabeled 데이터의 prediction 활용
        unsup_pred_hm1 = self.heavy_model(unsup_x_trans1, training=True)
        unsup_pred_hm2 = self.lite_model(unsup_x_trans2, training=True)

        return (
            sup_pred_hm1, sup_pred_hm2,
            unsup_true_hm1, unsup_true_hm2,
            unsup_pred_hm1, unsup_pred_hm2
        )
