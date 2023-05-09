import math
import random
from typing import Tuple, List

import tensorflow as tf
import tensorflow_addons as tfa

import albumentations as A


def parse_example(record, num_keypoints):
    """decoding tf-example"""
    feature_description = {
        'image_raw': tf.io.FixedLenFeature([], tf.string),
        'bbox': tf.io.FixedLenFeature([4, ], tf.float32),
        'keypoints': tf.io.FixedLenFeature([num_keypoints*3, ], tf.float32),
    }
    example = tf.io.parse_example(record, features=feature_description)
    image = tf.io.decode_jpeg(
        example['image_raw'],
        channels=3
    )
    bbox = example['bbox']
    keypoints = tf.reshape(example['keypoints'], (-1, 3))
    return image, bbox, keypoints


def parse_example_unlabeled(record):
    """decoding tf-example"""
    feature_description = {
        'image_raw': tf.io.FixedLenFeature([], tf.string),
        'bbox': tf.io.FixedLenFeature([4, ], tf.float32),
    }
    example = tf.io.parse_example(record, features=feature_description)
    image = tf.io.decode_jpeg(
        example['image_raw'],
        channels=3
    )
    bbox = example['bbox']
    return image, bbox


def read_image(image_file):
    return tf.io.decode_image(tf.io.read_file(image_file), channels=3)


def transform(
    img,
    scale,
    angle,
    center,
    output_shape
) -> Tuple[tf.Tensor, tf.Tensor]:
    tx = center[0] - output_shape[1] * scale / 2
    ty = center[1] - output_shape[0] * scale / 2

    # for offsetting translations caused by rotation:
    # https://docs.opencv.org/2.4/modules/imgproc/doc/geometric_transformations.html
    rx = (1 - tf.cos(angle)) * \
        output_shape[1] * scale / 2 - \
        tf.sin(angle) * output_shape[0] * scale / 2
    ry = tf.sin(angle) * output_shape[1] * scale / 2 + \
        (1 - tf.cos(angle)) * output_shape[0] * scale / 2

    transform = [scale * tf.cos(angle), scale * tf.sin(angle), rx + tx,
                 -scale * tf.sin(angle), scale * tf.cos(angle), ry + ty,
                 0., 0.]

    img = tfa.image.transform(tf.expand_dims(img, axis=0),
                              tf.expand_dims(transform, axis=0),
                              fill_mode='constant',
                              output_shape=output_shape[:2])
    img = tf.squeeze(img)

    # transform for keypoints
    alpha = 1 / scale * tf.cos(-angle)
    beta = 1 / scale * tf.sin(-angle)

    rx_xy = (1 - alpha) * center[0] - beta * center[1]
    ry_xy = beta * center[0] + (1 - alpha) * center[1]

    transform_xy = [[alpha, beta],
                    [-beta, alpha]]

    tx_xy = center[0] - output_shape[1] / 2
    ty_xy = center[1] - output_shape[0] / 2

    M = tf.concat([transform_xy, [[rx_xy - tx_xy], [ry_xy - ty_xy]]], axis=1)
    return img, M


def affine_transform(
    image,
    bbox_center,
    angle: float,
    scale: float,
    input_shape: List,
):
    """
    return transformed image and Matrix

    the reason of using (height - 1), (width - 1)
    results from https://arxiv.org/pdf/1911.07524.pdf
    (The Devil is in the Details: Delving into Unbiased
     Data Processing for Human Pose Estimation)
    """
    M = generate_affine_matrix(
        bbox_center,
        angle,
        scale,
        [input_shape[0] - 1, input_shape[1] - 1],
        inv=False
    )
    M = tf.reshape(M[:6], [2, 3])

    transforms = generate_affine_matrix(
        bbox_center,
        angle,
        scale,
        [input_shape[0] - 1, input_shape[1] - 1],
        inv=True
    )
    transformed_image = tfa.image.transform(
        tf.expand_dims(image, 0),
        tf.expand_dims(transforms, 0),
        output_shape=input_shape[:2]
    )
    transformed_image = tf.squeeze(transformed_image, 0)
    return transformed_image, M


def generate_affine_matrix(
    bbox_center,
    angle: float,
    scale: float,
    input_shape: List,
    inv: bool = False
):
    crop_mat = generate_crop_matrix(bbox_center, scale, input_shape, inv=inv)
    resize_mat = generate_resize_matrix(scale, inv=inv)
    rot_mat = generate_rotation_matrix(angle, input_shape, inv=inv)

    if inv:
        transform = crop_mat @ resize_mat @ rot_mat
    else:
        transform = rot_mat @ resize_mat @ crop_mat

    transform = tf.reshape(transform, [-1])[:-1]
    return transform


def generate_crop_matrix(bbox_center, scale, input_shape, inv: bool = False):
    crop_x = bbox_center[0] - (input_shape[1] * scale) / 2
    crop_y = bbox_center[1] - (input_shape[0] * scale) / 2

    crop_mat = tf.cond(
        tf.math.equal(inv, True),
        lambda: tf.reshape(
            [
                1., 0., crop_x,
                0., 1., crop_y,
                0., 0., 1.
            ],
            shape=[3, 3]
        ),
        lambda: tf.reshape(
            [
                1., 0., -crop_x,
                0., 1., -crop_y,
                0., 0., 1.
            ],
            shape=[3, 3]
        )
    )
    return tf.cast(crop_mat, tf.float32)


def generate_resize_matrix(scale, inv: bool = False):
    resize_mat = tf.cond(
        tf.math.equal(inv, True),
        lambda: tf.reshape(
            [
                scale, 0., 0.,
                0., scale, 0.,
                0., 0., 1.
            ],
            shape=[3, 3]
        ),
        lambda: tf.reshape(
            [
                1. / scale, 0., 0.,
                0., 1. / scale, 0.,
                0., 0., 1.
            ],
            shape=[3, 3]
        ),
    )
    return tf.cast(resize_mat, tf.float32)


def generate_rotation_matrix(angle, input_shape, inv: bool = False):
    radian = angle / 180 * tf.constant(math.pi, dtype=tf.float32)

    # move center to origin
    # 이미지 중심을 원점으로 이동
    translation1 = tf.reshape(
        tf.convert_to_tensor(
            [
                1., 0., (input_shape[1] / 2),
                0., 1., (input_shape[0] / 2),
                0., 0., 1.
            ],
            dtype=tf.float32
        ),
        shape=[3, 3]
    )

    # move back to center
    # 다시 이미지 중심으로 이동
    translation2 = tf.reshape(
        tf.convert_to_tensor(
            [
                1., 0., -(input_shape[1] / 2),
                0., 1., -(input_shape[0] / 2),
                0., 0., 1.
            ],
            dtype=tf.float32
        ),
        shape=[3, 3]
    )

    rotation_mat = tf.cond(
        tf.math.equal(inv, True),
        lambda: tf.reshape(
            tf.convert_to_tensor(
                [
                    tf.math.cos(radian), tf.math.sin(radian), 0.,
                    -tf.math.sin(radian), tf.math.cos(radian), 0.,
                    0., 0., 1.
                ],
                dtype=tf.float32
            ),
            shape=[3, 3]
        ),
        lambda: tf.reshape(
            tf.convert_to_tensor(
                [
                    tf.math.cos(radian), -tf.math.sin(radian), 0.,
                    tf.math.sin(radian), tf.math.cos(radian), 0.,
                    0., 0., 1.
                ],
                dtype=tf.float32
            ),
            shape=[3, 3]
        )
    )
    return tf.cast(translation1 @ rotation_mat @ translation2, tf.float32)


def normalize_image(image, means, stds):
    image /= 255.
    image -= [[means]]
    image /= [[stds]]
    return image


def half_body_transform(joints, center, scale, kpt_upper, input_shape):
    K = tf.shape(joints)[0]
    num_upper = tf.shape(kpt_upper)[0]
    vis_mask = joints[:, 2] > 0
    kpt_upper = tf.reshape(kpt_upper, (-1, 1))
    upper_body_mask = tf.scatter_nd(kpt_upper, tf.ones(num_upper,), shape=(K,))
    upper_body_mask = tf.cast(upper_body_mask, tf.bool)
    lower_body_mask = tf.math.logical_not(upper_body_mask)
    lower_body_mask = tf.math.logical_and(lower_body_mask, vis_mask)
    upper_body_mask = tf.math.logical_and(upper_body_mask, vis_mask)
    upper = tf.boolean_mask(joints, upper_body_mask)
    lower = tf.boolean_mask(joints, lower_body_mask)

    selected_joints = tf.cond(
        tf.math.less(tf.random.uniform([]), 0.5)
        & tf.math.greater(tf.shape(upper)[0], 2),
        lambda: upper,
        lambda: lower
    )
    center, scale = tf.cond(
        tf.math.greater_equal(tf.shape(selected_joints)[0], 2),
        lambda: _half_body_transform(selected_joints, input_shape),
        lambda: (center, scale)
    )
    return center, scale


def _half_body_transform(selected_joints, input_shape):
    center = tf.math.reduce_mean(selected_joints[:, :2], axis=0)
    left_top = tf.math.reduce_min(selected_joints[:, :2], axis=0)
    right_bottom = tf.math.reduce_max(selected_joints[:, :2], axis=0)
    w = right_bottom[0] - left_top[0]
    h = right_bottom[1] - left_top[1]
    aspect_ratio = input_shape[1] / input_shape[0]
    h = tf.cond(
        tf.math.greater(w, aspect_ratio * h),
        lambda: w * 1.0 / aspect_ratio,
        lambda: h
    )
    scale = (h * 1.25) / input_shape[0]
    scale = scale * 1.5
    scale = h / input_shape[0]
    return center, scale


def horizontal_flip(img, center, kp, flip_kp_indices):
    img_w = tf.cast(tf.shape(img)[1], tf.float32)
    img = img[:, ::-1, :]
    center_x = img_w - 1 - center[0]
    kp_x = img_w - 1 - kp[:, 0]
    kp = tf.concat([tf.expand_dims(kp_x, axis=1), kp[:, 1:]], axis=-1)
    kp = tf.gather(kp, flip_kp_indices, axis=0)
    center = tf.cast([center_x, center[1]], tf.float32)
    return img, center, kp


def random_src_color() -> Tuple[int, int, int]:
    return (
        random.randint(0, 255),  # Red
        random.randint(0, 255),  # Green
        random.randint(0, 255)   # Blue
    )


def get_albumentation():
    a_transform = A.Compose([
        A.MotionBlur(p=0.25, blur_limit=7, allow_shifted=False),
        A.HueSaturationValue(p=0.25),
        A.RandomBrightnessContrast(
            brightness_limit=(-0.3, 0.5),
            contrast_limit=(-0.3, 0.5),
            p=0.25
        ),
        A.RandomGamma(p=0.25),
        A.GaussNoise(p=0.25),
        A.CoarseDropout(
            p=0.25,
            max_height=30,
            max_width=30,
            max_holes=3,
            min_holes=1,
            min_height=10,
            min_width=10,
            fill_value=random_src_color()
        ),
        A.ISONoise(p=0.25, color_shift=(0.01, 0.15)),
        A.ImageCompression(quality_lower=50, p=0.25),
        A.OneOf([
            A.MotionBlur(blur_limit=7, p=1),
            A.Blur(blur_limit=3, p=1),
        ], p=0.25)
    ])
    return a_transform


def aug_fn(img):
    """augmentation using albumentations library"""
    data = {"image": img}
    aug_data = get_albumentation()(**data)
    return aug_data["image"]


@tf.function(input_signature=[tf.TensorSpec([None, None, 3], tf.uint8)])
def apply_aug(img) -> tf.Tensor:
    """apply augmentation using albumentations library"""
    img = tf.numpy_function(func=aug_fn, inp=[img], Tout=tf.uint8)
    return img


def preprocess(
    img, bbox, kp,
    use_image_norm: bool = True,
    means: List = [0.485, 0.456, 0.406],
    stds: List = [0.229, 0.224, 0.225],
    scale_factor: float = 0.3,
    rotation_prob: float = 0.6,
    rotation_factor: int = 40,
    flip_prob: float = 0.5,
    flip_kp_indices: List = [0, 2, 1, 4, 3, 6,
                             5, 8, 7, 10, 9, 12, 11, 14, 13, 16, 15],
    half_body_prob: float = 1.0,
    half_body_min_kp: int = 8,
    kpt_upper: List = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
    input_shape: List = [256, 192, 3],
    use_aug: bool = True,
    use_album_aug: bool = False
):
    """example별로 적용되는 전처리 함수. spatial/color augmentation"""
    kp = tf.cast(kp, tf.float32)

    x1, y1, w, h = bbox[0], bbox[1], bbox[2], bbox[3]
    bbox = tf.cast([x1, y1, x1 + w, y1 + h], tf.float32)
    bbox_center = tf.cast([x1 + w / 2., y1 + h / 2.], tf.float32)

    aspect_ratio = input_shape[1] / input_shape[0]
    h = tf.cond(
        tf.math.greater(w, aspect_ratio * h),
        lambda: w * 1.0 / aspect_ratio,
        lambda: h
    )
    scale = (h * 1.25) / input_shape[0]  # scale with bbox
    angle = 0.

    # augmentation
    joint_vis = tf.math.reduce_sum(tf.cast(kp[:, 2] > 0, tf.int32))
    bbox_center, scale = tf.cond(
        tf.math.equal(use_aug, True)
        & tf.math.greater(joint_vis, half_body_min_kp)
        & tf.math.less(tf.random.uniform([]), half_body_prob),
        lambda: half_body_transform(
            kp, bbox_center, scale, kpt_upper, input_shape
        ),
        lambda: (bbox_center, scale)
    )

    # 1. scale
    scale *= tf.cond(
        tf.math.equal(use_aug, True),
        lambda: tf.clip_by_value(tf.random.normal([]) * scale_factor + 1,
                                 1 - scale_factor,
                                 1 + scale_factor),
        lambda: 1.0
    )
    # 2. rotation, not radian
    angle = tf.cond(
        tf.math.equal(use_aug, True)
        & tf.math.less_equal(tf.random.uniform([]), rotation_prob),
        lambda: tf.clip_by_value(
            tf.random.normal([]) * rotation_factor,
            -2 * rotation_factor,
            2 * rotation_factor
        ),
        lambda: angle
    )
    # 3. horizontal flip
    img, bbox_center, kp = tf.cond(
        tf.math.equal(use_aug, True)
        & tf.math.less_equal(tf.random.uniform([]), flip_prob),
        lambda: horizontal_flip(img, bbox_center, kp, flip_kp_indices),
        lambda: (img, bbox_center, kp)
    )
    # transform to the object's center
    img, M = affine_transform(img, bbox_center, angle, scale, input_shape[:2])

    xy = kp[:, :2]
    xy = tf.transpose(tf.matmul(M[:, :2], xy, transpose_b=True)) + M[:, -1]

    # input 이미지 내에 있지 않은 키포인트의 visibility를 0으로 설정
    vis = tf.cast(kp[:, 2] > 0, tf.float32)  # vis==2인 경우 처리하기 위함
    vis *= tf.cast((
        (xy[:, 0] >= 0) &
        (xy[:, 0] < input_shape[1]) &
        (xy[:, 1] >= 0) &
        (xy[:, 1] < input_shape[0])), tf.float32)

    kp = tf.concat([xy, tf.expand_dims(vis, axis=1)], axis=1)

    # augmentation using albumentations
    img = tf.cond(
        tf.math.equal(use_album_aug, True),
        lambda: apply_aug(img),
        lambda: img
    )

    img = tf.cast(img, tf.float32)

    img = tf.cond(
        tf.math.equal(use_image_norm, True),
        lambda: normalize_image(img, means, stds),
        lambda: img
    )
    return img, kp


def preprocess_unlabeled(
    img, bbox,
    use_image_norm: bool = True,
    means: List = [0.485, 0.456, 0.406],
    stds: List = [0.229, 0.224, 0.225],
    scale_factor: float = 0.3,
    rotation_prob: float = 0.6,
    rotation_factor: int = 40,
    flip_prob: float = 0.5,
    flip_kp_indices: List = [0, 2, 1, 4, 3, 6,
                             5, 8, 7, 10, 9, 12, 11, 14, 13, 16, 15],
    input_shape: List = [256, 192, 3],
    use_aug: bool = True,
    use_album_aug: bool = False
):
    """unlabeled image에 적용되는 전처리 함수"""

    x1, y1, w, h = bbox[0], bbox[1], bbox[2], bbox[3]
    bbox = tf.cast([x1, y1, x1 + w, y1 + h], tf.float32)
    bbox_center = tf.cast([x1 + w / 2., y1 + h / 2.], tf.float32)

    aspect_ratio = input_shape[1] / input_shape[0]
    h = tf.cond(
        tf.math.greater(w, aspect_ratio * h),
        lambda: w * 1.0 / aspect_ratio,
        lambda: h
    )
    scale = (h * 1.25) / input_shape[0]  # scale with bbox
    angle = 0.

    # augmentation
    # 1. scale
    scale *= tf.cond(
        tf.math.equal(use_aug, True),
        lambda: tf.clip_by_value(tf.random.normal([]) * scale_factor + 1,
                                 1 - scale_factor,
                                 1 + scale_factor),
        lambda: 1.0
    )
    # 2. rotation, not radian
    angle = tf.cond(
        tf.math.equal(use_aug, True)
        & tf.math.less_equal(tf.random.uniform([]), rotation_prob),
        lambda: tf.clip_by_value(
            tf.random.normal([]) * rotation_factor,
            -2 * rotation_factor,
            2 * rotation_factor
        ),
        lambda: angle
    )
    # 3. horizontal flip
    # kp = tf.zeros([17, 3], tf.float32)
    img, bbox_center = tf.cond(
        tf.math.equal(use_aug, True)
        & tf.math.less_equal(tf.random.uniform([]), flip_prob),
        lambda: horizontal_flip(
            img,
            bbox_center,
            tf.zeros([17, 3], tf.float32),
            flip_kp_indices
        )[:-1],
        lambda: (img, bbox_center)
    )
    # transform to the object's center
    img, _ = affine_transform(img, bbox_center, angle, scale, input_shape[:2])

    # augmentation using albumentations
    img = tf.cond(
        tf.math.equal(use_album_aug, True),
        lambda: apply_aug(img),
        lambda: img
    )

    img = tf.cast(img, tf.float32)

    img = tf.cond(
        tf.math.equal(use_image_norm, True),
        lambda: normalize_image(img, means, stds),
        lambda: img
    )
    return img
