from typing import List
import math

import tensorflow as tf
from tensorflow.python.keras import Model

from src.decode import heatmap_to_coordinate


class PoseCoTrain(Model):

    def __init__(
        self,
        heavy_model,
        lite_model,
        scale_factor,
        rotation_factor,
        n_joints_mask,
        name: str = 'pose_cotrain'
    ):
        super().__init__(name=name)
        self.heavy_model = heavy_model
        self.lite_model = lite_model
        self.sf = scale_factor
        self.rf = rotation_factor
        self.n_joints_mask = n_joints_mask

    def build(self, input_shape):
        self.heavy_model.build(input_shape)
        self.lite_model.build(input_shape)

    def call(self, sup_x, unsup_x, training=True):
        B = tf.shape(sup_x)[0]
        H = tf.shape(sup_x)[1]
        W = tf.shape(sup_x)[2]

        sup_pred_hm1 = self.heavy_model(sup_x, training=training)
        sup_pred_hm2 = self.lite_model(sup_x, training=training)

        # Teachers
        # Easy Augmentation
        unsup_hm1 = self.heavy_model(unsup_x, training=False)
        unsup_hm2 = self.lite_model(unsup_x, training=False)

        unsup_x_trans = unsup_x
        unsup_x_trans_2 = unsup_x

        # Joint Cutout
        # Masking joints
        if self.n_joints_mask > 0:
            pred_1 = heatmap_to_coordinate(unsup_hm1)
            pred_2 = heatmap_to_coordinate(unsup_hm2)

            unsup_x_trans = self._mask_joints(
                unsup_x_trans, pred_1 * 4, self.n_joints_mask
            )
            unsup_x_trans_2 = self._mask_joints(
                unsup_x_trans_2, pred_2 * 4, self.n_joints_mask
            )

        # Apply Affine Transformation again for hard augmentation
        # make easy prediction be hard one -> To use GT for unlabeled
        # unlabeled 데이터의 GT로 활용
        # NOTE: do not backprop (no gradients passing)
        thetas = self._get_batch_theta(B, self.rf, self.sf)
        grids = self._affine_grid(thetas, H, W)

        unsup_x_trans1 = self._bilinear_sampler(
            unsup_x_trans,
            grids[:, 0, ...],
            grids[:, 1, ...]
        )
        unsup_x_trans2 = self._bilinear_sampler(
            unsup_x_trans_2,
            grids[:, 0, ...],
            grids[:, 1, ...]
        )
        # transforms easy prediction to hard one
        hm_grids = self._affine_grid(
            thetas,
            tf.shape(sup_pred_hm1)[1],
            tf.shape(sup_pred_hm1)[2]
        )
        hm_grids = tf.cast(hm_grids, tf.float32)
        unsup_true_hm1 = self._bilinear_sampler(
            unsup_hm1,
            hm_grids[:, 0, ...],
            hm_grids[:, 1, ...]
        )
        unsup_true_hm2 = self._bilinear_sampler(
            unsup_hm2,
            hm_grids[:, 0, ...],
            hm_grids[:, 1, ...]
        )

        # Students
        # Hard Augmentation이 적용된 sample 예측
        # unlabeled 데이터의 prediction 활용
        unsup_pred_hm1 = self.heavy_model(unsup_x_trans1, training=training)
        unsup_pred_hm2 = self.lite_model(unsup_x_trans2, training=training)

        return (
            sup_pred_hm1, sup_pred_hm2,
            unsup_true_hm1, unsup_true_hm2,
            unsup_pred_hm1, unsup_pred_hm2
        )

    def _get_single_theta(
        self,
        rf: float,
        sf: float
    ):
        """return 2x3 scale-rotation matrix"""
        scale = tf.clip_by_value(
            tf.random.normal([]) * sf + 1,
            1 - sf,
            1 + sf
        )
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
        theta = tf.convert_to_tensor(
            [
                [scale * tf.math.cos(radian), scale * tf.math.sin(radian), 0.],
                [-scale * tf.math.sin(radian), scale * tf.math.cos(radian), 0.]
            ],
            dtype=tf.float32
        )
        return theta  # (2, 3)

    def _get_batch_theta(
        self,
        batch_size: int,
        rf: float,
        sf: float
    ):
        thetas = tf.map_fn(
            lambda i: self._get_single_theta(rf, sf),
            elems=tf.range(batch_size),
            fn_output_signature=tf.float32
        )

        # thetas = tf.stack(
        #     [
        #         self._get_single_theta(rf, sf)
        #         for _ in tf.range(batch_size)
        #     ],
        #     axis=0
        # )
        return thetas

    def _affine_grid(self, thetas, height, width):
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

    def _get_pixel_value(self, img, x, y):
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

    def _bilinear_sampler(self, img, x, y):
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
        Ia = self._get_pixel_value(img, x0, y0)
        Ib = self._get_pixel_value(img, x0, y1)
        Ic = self._get_pixel_value(img, x1, y0)
        Id = self._get_pixel_value(img, x1, y1)

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

    def _mask_single_joint(self, image, xmin, ymin, xmax, ymax):
        """
        image: (B, H, W, 3)
        xmin ~ ymax: (B,)
        """
        xx, yy = tf.meshgrid(
            tf.range(image.shape[2]),
            tf.range(image.shape[1])
        )
        grid_xy = tf.stack([xx, yy], axis=-1)
        # grid_xy = tf.cast(grid_xy, tf.float32)  # (H, W, 2)
        grid_xy = tf.tile(
            tf.expand_dims(grid_xy, 0),
            [tf.shape(image)[0], 1, 1, 1]
        )  # (B, H, W, 2)

        # check occlusion range
        mask_x = tf.math.logical_and(
            grid_xy[..., 0] >= xmin,
            grid_xy[..., 0] <= xmax,
        )
        mask_y = tf.math.logical_and(
            grid_xy[..., 1] >= ymin,
            grid_xy[..., 1] <= ymax,
        )
        mask = tf.stack([mask_x, mask_y], axis=-1)
        mask = tf.math.reduce_all(mask, axis=-1, keepdims=True)
        mask = tf.cast(mask, tf.float32)

        gnoise = self._generate_random_noise(tf.shape(image))
        masked_image = mask * gnoise + (1 - mask) * image
        return masked_image

    def _generate_random_noise(
        self,
        image_shape: List,
        means: List = [0.485, 0.456, 0.406],
        stds: List = [0.229, 0.224, 0.225]
    ) -> tf.Tensor:
        return 255. * tf.random.normal(
            shape=image_shape,
            mean=means,
            stddev=stds,
            dtype=tf.float32
        )

    def _mask_joints(self, image, keypoints, n_masks: int = 4):
        """
        unlabeled image 예측 결과를 바탕으로 입력 이미지에 joints cutout 적용

        NOTE: while_loop 대신 matrix-broadcasting을 적용하여 속도 4배 향상

        Args:
            image (tf.Tensor): batch input images; (B, H, W, 3)
            keypoints (_type_): unlabeled image prediction; (B, K, 2)
            n_masks (int, optional): number of joints to be masked.
                Defaults to 4.

        Returns:
            tf.Tensor: same shape and dtype of input image
        """
        B = tf.shape(keypoints)[0]
        K = tf.shape(keypoints)[1]
        height = tf.shape(image)[1]
        width = tf.shape(image)[2]

        re_keypoints = \
            tf.cast(keypoints[..., :2], tf.float32) \
            + tf.random.normal([B, K, 2]) * 10.0
        re_keypoints = tf.cast(re_keypoints, tf.int32)
        size = tf.random.uniform([B, K, 2], 10, 20, tf.int32)

        tl = re_keypoints - size  # xmin, ymin
        br = re_keypoints + size  # xmax, ymax

        # (B, K) shape tensors
        xmin = tf.clip_by_value(tl[..., 0], 0, width)
        ymin = tf.clip_by_value(tl[..., 1], 0, height)
        xmax = tf.clip_by_value(br[..., 0], 0, width)
        ymax = tf.clip_by_value(br[..., 1], 0, height)

        mask_indices = tf.random.uniform(
            [B, n_masks], maxval=K, dtype=tf.int32)
        mask_indices = tf.stack(
            [
                tf.tile(tf.expand_dims(tf.range(B), 1),
                        multiples=[1, n_masks]),
                mask_indices
            ],
            axis=-1
        )
        xmin = tf.gather_nd(xmin, mask_indices)  # (B, n_masks)
        ymin = tf.gather_nd(ymin, mask_indices)
        xmax = tf.gather_nd(xmax, mask_indices)
        ymax = tf.gather_nd(ymax, mask_indices)

        # i = tf.constant(0)
        # _, image = tf.while_loop(
        #     cond=lambda i, _: tf.math.less(i, n_masks),
        #     body=lambda i, image: (
        #         i + 1,
        #         # image
        #         mask_single_joint(
        #             image,
        #             xmin[:, i], ymin[:, i], xmax[:, i], ymax[:, i]
        #         )
        #     ),
        #     loop_vars=[i, image]
        # )

        xx, yy = tf.meshgrid(
            tf.range(image.shape[2]),
            tf.range(image.shape[1])
        )
        grid_xy = tf.stack([xx, yy], axis=-1)
        grid_xy = tf.tile(
            tf.expand_dims(grid_xy, 0),
            [tf.shape(image)[0], 1, 1, 1]
        )  # (B, H, W, 2)
        grid_xy = tf.expand_dims(grid_xy, axis=3)  # (B, H, W, 1, 2)

        tl = tf.reshape(
            tf.stack([xmin, ymin], axis=-1),
            (-1, 1, 1, xmin.shape[-1], 2)
        )
        br = tf.reshape(
            tf.stack([xmax, ymax], axis=-1),
            (-1, 1, 1, xmax.shape[-1], 2)
        )
        mask = tf.math.logical_and(
            grid_xy >= tl,
            grid_xy <= br,
        )
        mask = tf.math.reduce_all(mask, axis=-1)
        mask = tf.math.reduce_any(mask, axis=-1, keepdims=True)
        mask = tf.cast(mask, tf.float32)

        gnoise = self._generate_random_noise(tf.shape(image))
        return mask * gnoise + (1 - mask) * image
