"""HRNet with two heads"""
from typing import List, Optional, Union
import tensorflow as tf
from tensorflow.keras import Model, Sequential
from tensorflow.keras import layers

from src.model.common import ConvBN, BasicBlock, Bottleneck


class Stem(layers.Layer):
    """
    first step of HRNet.
    it consists of 2 convolutions(stride=2) decreasing resolution.
    """

    def __init__(
        self,
        filters: int,
        momentum: Optional[float] = 0.9,
        name: Optional[str] = 'stem'
    ):
        super().__init__(name=name)
        self.conv1 = layers.Conv2D(
            filters=filters,
            kernel_size=3,
            strides=2,
            padding='same',
            use_bias=False,
            name=f'{name}/conv1'
        )
        self.bn1 = layers.BatchNormalization(
            momentum=momentum,
            name=f'{name}/bn1'
        )
        self.relu1 = layers.Activation(
            'relu',
            name=f'{name}/relu1'
        )
        self.conv2 = layers.Conv2D(
            filters=filters,
            kernel_size=3,
            strides=2,
            padding='same',
            use_bias=False,
            name=f'{name}/conv2'
        )
        self.bn2 = layers.BatchNormalization(
            momentum=momentum,
            name=f'{name}/bn2'
        )
        self.relu2 = layers.Activation(
            'relu',
            name=f'{name}/relu2'
        )

    def call(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu1(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu2(x)
        return x


class FirstTransition(layers.Layer):

    def __init__(
        self,
        filters: Optional[int] = 64,
        name: Optional[str] = 'transition_1'
    ):
        super(FirstTransition, self).__init__(name=name)
        self.conv1 = ConvBN(filters // 2, 3, 1, 'same')
        self.transition = ConvBN(filters, 3, 2, 'same')

    def call(self, x):
        x_4 = self.conv1(x)
        x_8 = self.transition(x)
        return [x_4, x_8]


class Transition(layers.Layer):

    def __init__(self, filters, name='transition'):
        super(Transition, self).__init__(name=name)
        self.transition = ConvBN(filters, 3, 2, 'same')

    def call(self, branches):
        branches.extend([self.transition(tf.identity(branches[-1]))])
        return branches


class Stage(layers.Layer):
    """Iteratively apply block modules and fuse their outputs.

    Its outputs are same with the inputs (inputs' shapes are not changed).
    Last iteration of final stage(stage4) make only one output
    with lowest resolution feature map.
    """

    def __init__(
        self,
        num_branches: int,
        num_iter: int,
        block_module: Union[BasicBlock, Bottleneck],
        num_blocks_list: List,
        filters_list: List,
        momentum: Optional[int] = 0.9,
        is_multi_output: Optional[bool] = True,
        name: Optional[str] = 'stage'
    ):
        super(Stage, self).__init__(name=name)
        self.num_branches = num_branches
        self.block_module = block_module
        self.num_blocks_list = num_blocks_list
        self.filters_list = filters_list
        self.momentum = momentum
        self.branches_blocks = [self.__set_blocks_for_branches()
                                for _ in range(num_iter)]
        self.fusion_layers = []
        for i in range(num_iter):
            if not is_multi_output and i == num_iter - 1:
                self.is_multi_output = False
            else:
                self.is_multi_output = True
            self.fusion_layers.append(self.__set_fusion_layers())
        self.num_iter = num_iter
        self.relu = layers.ReLU()

    def call(self, inputs):
        # iterative call 'call_once' method by num_iter
        for i in range(self.num_iter):
            inputs = self.call_once(
                inputs, self.branches_blocks[i], self.fusion_layers[i])
        return inputs  # return same shape of inputs

    def call_once(self, inputs, branches_blocks, fusion_layers):
        # iteratively applying blocks to each branch
        x_list = [branches_blocks[i](inputs[i])
                  for i in range(self.num_branches)]
        x_fuses = []
        for i in range(len(fusion_layers)):
            y = x_list[0] if i == 0 else fusion_layers[i][0](x_list[0])
            for j in range(1, self.num_branches):
                if i == j:
                    y += x_list[j]
                else:
                    y += fusion_layers[i][j](x_list[j])
            x_fuses.append(self.relu(y))

        return x_fuses

    def __set_blocks_for_branches(self):
        # apply all branches
        branches_blocks = [
            self.__set_blocks_for_branch(self.num_blocks_list[branch_index],
                                         self.filters_list[branch_index])
            for branch_index in range(self.num_branches)
        ]
        return branches_blocks

    def __set_blocks_for_branch(self, num_blocks, filters):
        # apply each branch
        branch_blocks = Sequential(
            [self.block_module(filters) for _ in range(num_blocks)])
        return branch_blocks

    def __set_fusion_layers(self):
        fusion_layers = []
        for i in range(self.num_branches if self.is_multi_output else 1):
            fusion_layer = []
            for j in range(self.num_branches):
                if j == i:  # just as it is
                    fusion_layer.append(None)
                elif j > i:
                    fusion_layer.append(
                        Sequential(
                            [
                                layers.Conv2D(
                                    self.filters_list[i], 1, 1, use_bias=False
                                ),
                                layers.BatchNormalization(
                                    momentum=self.momentum
                                ),
                                layers.UpSampling2D(2**(j-i))
                            ]
                        )
                    )
                else:  # j < i, downsampling
                    downsamplings = []
                    for k in range(i-j):
                        if k == (i - j - 1):
                            downsamplings.append(
                                layers.Conv2D(
                                    self.filters_list[i], 3, 2, 'same',
                                    use_bias=False
                                )
                            )
                            downsamplings.append(
                                layers.BatchNormalization(
                                    momentum=self.momentum
                                )
                            )
                        else:
                            downsamplings.append(
                                ConvBN(self.filters_list[j], 3, 2, 'same')
                            )
                    fusion_layer.append(
                        Sequential(downsamplings)
                    )
            fusion_layers.append(fusion_layer)

        return fusion_layers


class HRNet(Model):
    momentum = 0.9

    def __init__(
        self,
        filter_list: Optional[List] = [32, 64, 128, 256],
        num_keypoints: Optional[int] = 17,
        name: Optional[str] = 'hrnet'
    ):
        super(HRNet, self).__init__(name=name)
        self.stem = Stem(64)  # return [H/4, W/4] feature maps

        # main body : stage 1
        downsample = Sequential([
            layers.Conv2D(
                64 * Bottleneck.expansion,
                kernel_size=1,
                strides=1,
                use_bias=False
            ),
            layers.BatchNormalization(momentum=self.momentum)
        ])  # is it downsampled?

        self.stage1 = Sequential(
            [
                Bottleneck(64, downsample=downsample),
                Bottleneck(64),
                Bottleneck(64),
                Bottleneck(64),
            ],
            name='stage1'
        )

        self.transition1 = FirstTransition(
            filters=filter_list[-3], name='transition_1'
        )
        self.stage2 = Stage(
            num_branches=2,
            num_iter=1,
            block_module=BasicBlock,
            num_blocks_list=[4, 4],
            filters_list=filter_list[:-2],
            is_multi_output=True,
            name='stage_2'
        )

        self.transition2 = Transition(
            filters=filter_list[-2], name='transition_2'
        )
        self.stage3 = Stage(
            num_branches=3,
            num_iter=4,
            block_module=BasicBlock,
            num_blocks_list=[4, 4, 4],
            filters_list=filter_list[:-1],
            is_multi_output=True,
            name='stage_3'
        )

        self.transition3 = Transition(
            filters=filter_list[-1], name='transition_3'
        )
        self.stage4 = Stage(
            num_branches=4,
            num_iter=3,
            block_module=BasicBlock,
            num_blocks_list=[4, 4, 4, 4],
            filters_list=filter_list,
            is_multi_output=False,
            name='stage_4'
        )

        self.regressor = layers.Conv2D(num_keypoints, 1, 1, name='regressor')

    def call(self, x):
        # stem
        x = self.stem(x)

        # main body
        x = self.stage1(x)  # (B, H/4, W/4, 256)

        # (B, H/4, W/4, 32), (B, H/8, W/8, 64)
        x_4, x_8 = self.transition1(x)
        x_4, x_8 = self.stage2([x_4, x_8])

        # (B, H/4, W/4, 32), (B, H/8, W/8, 64), (B, H/16, W/16, 128)
        x_4, x_8, x_16 = self.transition2([x_4, x_8])
        x_4, x_8, x_16 = self.stage3([x_4, x_8, x_16])

        # x_4_hm -> (B, H/4, W/4, 32),
        # x_8_hm -> (B, H/8, W/8, 64),
        # x_16_hm -> (B, H/16, W/16, 128),
        # x_32_hm -> (B, H/32, W/32, 256)
        x_4, x_8, x_16, x_32 = self.transition3([x_4, x_8, x_16])
        x_4 = self.stage4([x_4, x_8, x_16, x_32])[0]

        # regressor
        out = self.regressor(x_4)  # (B, H/4, W/4, num_keypoints)
        return out


def HRNetW32(
    filter_list: Optional[List] = [32, 64, 128, 256],
    num_keypoints: Optional[int] = 17
):
    return HRNet(filter_list, num_keypoints, name='hrnet-w32')


def HRNetW48(
    filter_list: Optional[List] = [48, 96, 192, 384],
    num_keypoints: Optional[int] = 17
):
    return HRNet(filter_list, num_keypoints, name='hrnet-w48')
