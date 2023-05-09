from tensorflow.keras import layers


class ConvBN(layers.Layer):
    """conv2d -> batch normalization -> activation"""

    def __init__(
        self,
        filters,
        kernel_size=3,
        strides=1,
        padding='same',
        momentum=0.9,
        activation='relu'
    ):
        super(ConvBN, self).__init__()
        self.conv = layers.Conv2D(
            filters, kernel_size, strides, padding, use_bias=False)
        self.bn = layers.BatchNormalization(momentum=momentum)

        if activation == 'relu':
            self.activation = layers.ReLU()
        else:
            self.activation = layers.LeakyReLU()

    def call(self, x):
        out = self.conv(x)
        out = self.bn(out)
        out = self.activation(out)
        return out


class BasicBlock(layers.Layer):

    def __init__(self, filters, strides=1, momentum=0.9, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = layers.Conv2D(filters,
                                   kernel_size=3,
                                   strides=strides,
                                   padding='same',
                                   use_bias=False)
        self.bn1 = layers.BatchNormalization(momentum=momentum)
        self.relu = layers.ReLU()
        self.conv2 = layers.Conv2D(
            filters,
            kernel_size=3,
            strides=1,
            padding='same',
            use_bias=False
        )
        self.bn2 = layers.BatchNormalization(momentum=momentum)
        self.downsample = downsample

    def call(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class Bottleneck(layers.Layer):
    """
    1x1 -> 3x3 -> 1x1 -> residual connection
    if you use stride=1(default), resolution doesn't change.
        --> just input channels are getting `expansion` times bigger than before.
    if you use stride=2, input resolution become half.
        --> in this case, you should give the `downsample` argument.
    """
    expansion = 4

    def __init__(self, filters, strides=1, momentum=0.9, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = layers.Conv2D(filters,
                                   kernel_size=1,
                                   use_bias=False)
        self.bn1 = layers.BatchNormalization(momentum=momentum)
        self.conv2 = layers.Conv2D(filters,
                                   kernel_size=3,
                                   strides=strides,
                                   padding='same',
                                   use_bias=False)
        self.bn2 = layers.BatchNormalization(momentum=momentum)
        self.conv3 = layers.Conv2D(filters * self.expansion,
                                   kernel_size=1,
                                   use_bias=False)
        self.bn3 = layers.BatchNormalization(momentum=momentum)
        self.relu = layers.ReLU()
        self.downsample = downsample

    def call(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)
        return out
