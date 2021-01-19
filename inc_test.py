# inc, just with batch normalization removed and 2 prediction units

INC_HW = 299
CHANNEL_AXIS = 3

def assemble_layers():
    from tensorflow.keras.layers import (
        Dense,
        MaxPooling2D,
        Conv2D,
        Activation,
        AveragePooling2D,
        # BatchNormalization,
        Concatenate,
        GlobalAveragePooling2D,
        Lambda
    )
    def conv2d_bn(x,
                  filters,
                  kernel_size,
                  strides=1,
                  padding='same',
                  activation='relu',
                  use_bias=False,
                  name=None):
        """Utility function to apply conv + BN.

        # Arguments
            x: input tensor.
            filters: filters in `Conv2D`.
            kernel_size: kernel size as in `Conv2D`.
            strides: strides in `Conv2D`.
            padding: padding mode in `Conv2D`.
            activation: activation in `Conv2D`.
            use_bias: whether to use a bias in `Conv2D`.
            name: name of the ops; will become `name + '_ac'` for the activation
                and `name + '_bn'` for the batch norm layer.

        # Returns
            Output tensor after applying `Conv2D` and `BatchNormalization`.
        """
        x = Conv2D(filters,
                   kernel_size,
                   strides=strides,
                   padding=padding,
                   use_bias=use_bias,
                   name=name)(x)
        # if not use_bias:
        #     bn_name = None if name is None else name + '_bn'
        #     x = BatchNormalization(axis=CHANNEL_AXIS,
        #                            scale=False,
        #                            name=bn_name,
        #
        #                            # DEBUG
        #                            # trainable=True
        #
        #
        #                            )(x)
        if activation is not None:
            ac_name = None if name is None else name + '_ac'
            x = Activation(activation, name=ac_name)(x)
        return x

    def inception_resnet_block(x, scale, block_type, block_idx, activation='relu'):
        """Adds a Inception-ResNet block.

        This function builds 3 types of Inception-ResNet blocks mentioned
        in the paper, controlled by the `block_type` argument (which is the
        block name used in the official TF-slim implementation):
            - Inception-ResNet-A: `block_type='block35'`
            - Inception-ResNet-B: `block_type='block17'`
            - Inception-ResNet-C: `block_type='block8'`

        # Arguments
            x: input tensor.
            scale: scaling factor to scale the residuals (i.e., the output of
                passing `x` through an inception module) before adding them
                to the shortcut branch.
                Let `r` be the output from the residual branch,
                the output of this block will be `x + scale * r`.
            block_type: `'block35'`, `'block17'` or `'block8'`, determines
                the network structure in the residual branch.
            block_idx: an `int` used for generating layer names.
                The Inception-ResNet blocks
                are repeated many times in this network.
                We use `block_idx` to identify
                each of the repetitions. For example,
                the first Inception-ResNet-A block
                will have `block_type='block35', block_idx=0`,
                and the layer names will have
                a common prefix `'block35_0'`.
            activation: activation function to use at the end of the block
                (see [activations](../activations.md)).
                When `activation=None`, no activation is applied
                (i.e., "linear" activation: `a(x) = x`).

        # Returns
            Output tensor for the block.

        # Raises
            ValueError: if `block_type` is not one of `'block35'`,
                `'block17'` or `'block8'`.
        """
        if block_type == 'block35':
            branch_0 = conv2d_bn(x, 32, 1)
            branch_1 = conv2d_bn(x, 32, 1)
            branch_1 = conv2d_bn(branch_1, 32, 3)
            branch_2 = conv2d_bn(x, 32, 1)
            branch_2 = conv2d_bn(branch_2, 48, 3)
            branch_2 = conv2d_bn(branch_2, 64, 3)
            branches = [branch_0, branch_1, branch_2]
        elif block_type == 'block17':
            branch_0 = conv2d_bn(x, 192, 1)
            branch_1 = conv2d_bn(x, 128, 1)
            branch_1 = conv2d_bn(branch_1, 160, [1, 7])
            branch_1 = conv2d_bn(branch_1, 192, [7, 1])
            branches = [branch_0, branch_1]
        elif block_type == 'block8':
            branch_0 = conv2d_bn(x, 192, 1)
            branch_1 = conv2d_bn(x, 192, 1)
            branch_1 = conv2d_bn(branch_1, 224, [1, 3])
            branch_1 = conv2d_bn(branch_1, 256, [3, 1])
            branches = [branch_0, branch_1]
        else:
            raise ValueError('Unknown Inception-ResNet block type. '
                             'Expects "block35", "block17" or "block8", '
                             'but got: ' + str(block_type))

        block_name = block_type + '_' + str(block_idx)
        channel_axis = CHANNEL_AXIS
        mixed = Concatenate(
            axis=channel_axis, name=block_name + '_mixed')(branches)
        up = conv2d_bn(mixed,
                       tuple(x.shape.as_list())[channel_axis],
                       1,
                       activation=None,
                       use_bias=True,
                       name=block_name + '_conv')

        x = Lambda(lambda inputs, the_scale: inputs[0] + inputs[1] * the_scale,
                   output_shape=tuple(x.shape.as_list())[1:],
                   arguments={'the_scale': scale},
                   name=block_name)([x, up])
        if activation is not None:
            x = Activation(activation, name=block_name + '_ac')(x)
        return x







    dims = [INC_HW, INC_HW, INC_HW]
    dims[CHANNEL_AXIS - 1] = 3
    from tensorflow.python.keras import Input
    inputs = Input(tuple(dims))

    # Stem block: 35 x 35 x 192
    x = MaxPooling2D(3, strides=2)(
        conv2d_bn(
            conv2d_bn(
                MaxPooling2D(
                    3,
                    strides=2
                )(
                    conv2d_bn(
                        conv2d_bn(
                            conv2d_bn(
                                inputs,
                                32,
                                3,
                                strides=2,
                                padding='valid'
                            ),
                            32,
                            3,
                            padding='valid'
                        ),
                        64,
                        3
                    )
                ),
                80,
                1,
                padding='valid'
            ),
            192,
            3,
            padding='valid'
        )
    )

    # Mixed 5b (Inception-A block): 35 x 35 x 320
    x = Concatenate(
        axis=CHANNEL_AXIS,
        name='mixed_5b'
    )([
        # branches
        conv2d_bn(x, 96, 1),
        conv2d_bn(conv2d_bn(x, 48, 1), 64, 5),
        conv2d_bn(
            conv2d_bn(
                conv2d_bn(x, 64, 1), 96, 3
            ),
            96,
            3
        ),
        # branch_pool
        conv2d_bn(
            AveragePooling2D(3, strides=1, padding='same')(x),
            64,
            1
        )
    ])

    # 10x block35 (Inception-ResNet-A block): 35 x 35 x 320
    for block_idx in range(1, 11):
        x = inception_resnet_block(
            x,
            scale=0.17,
            block_type='block35',
            block_idx=block_idx
        )

    # Mixed 6a (Reduction-A block): 17 x 17 x 1088
    x = Concatenate(
        axis=CHANNEL_AXIS,
        name='mixed_6a'
    )([
        # branches
        conv2d_bn(x, 384, 3, strides=2, padding='valid'),
        conv2d_bn(
            conv2d_bn(
                conv2d_bn(x, 256, 1),
                256,
                3
            ),
            384,
            3,
            strides=2,
            padding='valid'
        ),

        # branch_pool
        MaxPooling2D(3, strides=2, padding='valid')(x)
    ])

    # 20x block17 (Inception-ResNet-B block): 17 x 17 x 1088
    for block_idx in range(1, 21):
        x = inception_resnet_block(
            x,
            scale=0.1,
            block_type='block17',
            block_idx=block_idx
        )

    # Mixed 7a (Reduction-B block): 8 x 8 x 2080
    x = Concatenate(axis=CHANNEL_AXIS, name='mixed_7a')(
        [
            # branches
            conv2d_bn(
                conv2d_bn(x, 256, 1),
                384,
                3,
                strides=2,
                padding='valid'
            ),
            conv2d_bn(
                conv2d_bn(x, 256, 1),
                288,
                3,
                strides=2,
                padding='valid'
            ),
            conv2d_bn(
                conv2d_bn(
                    conv2d_bn(x, 256, 1),
                    288,
                    3
                ),
                320,
                3,
                strides=2,
                padding='valid'
            ),

            # branch pool
            MaxPooling2D(3, strides=2, padding='valid')(x)
        ]
    )

    # 10x block8 (Inception-ResNet-C block): 8 x 8 x 2080
    for block_idx in range(1, 10):
        x = inception_resnet_block(
            x,
            scale=0.2,
            block_type='block8',
            block_idx=block_idx
        )
    x = inception_resnet_block(
        x,
        scale=1.,
        activation=None,
        block_type='block8',
        block_idx=10
    )

    return Dense(
        2,
        activation='softmax',
        name='predictions'
    )(GlobalAveragePooling2D(
        # Classification block
        name='avg_pool'
    )(conv2d_bn(
        # Final convolution block: 8 x 8 x 1536
        x,
        1536,
        1,
        name='conv_7b'
    )))
