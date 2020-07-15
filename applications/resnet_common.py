"""
References:
https://github.com/keras-team/keras-applications/blob/1.0.8/keras_applications/resnet_common.py
"""

from collections import OrderedDict

from tensorflow.python.keras import backend as K
from tensorflow.python.keras.applications.imagenet_utils import preprocess_input
from tensorflow.python.keras.applications.resnet import (
    BASE_WEIGHTS_PATH,
    WEIGHTS_HASHES,
    ResNet,
    stack1,
    stack2,
    stack3,
)
from tensorflow.python.keras.layers import Input
from tensorflow.python.keras.models import Model
from tensorflow.python.keras.utils import data_utils


def init_resnet(
    stack_fn,
    preact,
    use_bias,
    model_name,
    input_shape,
    block_name_to_hyperparameters_dict,
    preprocess_input_mode,
):
    # Downloads the weights
    file_name = model_name + "_weights_tf_dim_ordering_tf_kernels_notop.h5"
    file_hash = WEIGHTS_HASHES[model_name][1]
    weights_path = data_utils.get_file(
        file_name,
        BASE_WEIGHTS_PATH + file_name,
        cache_subdir="models",
        file_hash=file_hash,
    )

    # Define and initialize the first block
    first_block = ResNet(
        stack_fn=lambda x: x,
        preact=preact,
        use_bias=use_bias,
        include_top=False,
        weights=None,
        input_shape=input_shape,
    )
    first_block.load_weights(weights_path, by_name=True)
    submodel_list = [first_block]

    # Define and initialize each block
    for block_name, (
        filters,
        blocks,
        stride1,
    ) in block_name_to_hyperparameters_dict.items():
        input_tensor = Input(shape=K.int_shape(submodel_list[-1].output)[1:])
        output_tensor = stack_fn(
            input_tensor,
            filters=filters,
            blocks=blocks,
            stride1=stride1,
            name=block_name,
        )
        submodel = Model(
            inputs=input_tensor,
            outputs=output_tensor,
            name="{}_block".format(block_name),
        )
        submodel.load_weights(weights_path, by_name=True)
        submodel_list.append(submodel)

    return submodel_list, lambda x: preprocess_input(x, mode=preprocess_input_mode)


ResNet50 = lambda input_shape, last_stride1=1: init_resnet(
    stack_fn=stack1,
    preact=False,
    use_bias=True,
    model_name="resnet50",
    input_shape=input_shape,
    block_name_to_hyperparameters_dict=OrderedDict(
        [
            ("conv2", (64, 3, 1)),
            ("conv3", (128, 4, 2)),
            ("conv4", (256, 6, 2)),
            ("conv5", (512, 3, last_stride1)),
        ]
    ),
    preprocess_input_mode="caffe",
)

ResNet101 = lambda input_shape, last_stride1=1: init_resnet(
    stack_fn=stack1,
    preact=False,
    use_bias=True,
    model_name="resnet101",
    input_shape=input_shape,
    block_name_to_hyperparameters_dict=OrderedDict(
        [
            ("conv2", (64, 3, 1)),
            ("conv3", (128, 4, 2)),
            ("conv4", (256, 23, 2)),
            ("conv5", (512, 3, last_stride1)),
        ]
    ),
    preprocess_input_mode="caffe",
)

ResNet152 = lambda input_shape, last_stride1=1: init_resnet(
    stack_fn=stack1,
    preact=False,
    use_bias=True,
    model_name="resnet152",
    input_shape=input_shape,
    block_name_to_hyperparameters_dict=OrderedDict(
        [
            ("conv2", (64, 3, 1)),
            ("conv3", (128, 8, 2)),
            ("conv4", (256, 36, 2)),
            ("conv5", (512, 3, last_stride1)),
        ]
    ),
    preprocess_input_mode="caffe",
)

ResNet50V2 = lambda input_shape: init_resnet(
    stack_fn=stack2,
    preact=False,
    use_bias=True,
    model_name="resnet50v2",
    input_shape=input_shape,
    block_name_to_hyperparameters_dict=OrderedDict(
        [
            ("conv2", (64, 3, 2)),
            ("conv3", (128, 4, 2)),
            ("conv4", (256, 6, 2)),
            ("conv5", (512, 3, 1)),
        ]
    ),
    preprocess_input_mode="tf",
)

ResNet101V2 = lambda input_shape: init_resnet(
    stack_fn=stack2,
    preact=False,
    use_bias=True,
    model_name="resnet101v2",
    input_shape=input_shape,
    block_name_to_hyperparameters_dict=OrderedDict(
        [
            ("conv2", (64, 3, 2)),
            ("conv3", (128, 4, 2)),
            ("conv4", (256, 23, 2)),
            ("conv5", (512, 3, 1)),
        ]
    ),
    preprocess_input_mode="tf",
)

ResNet152V2 = lambda input_shape: init_resnet(
    stack_fn=stack2,
    preact=False,
    use_bias=True,
    model_name="resnet152v2",
    input_shape=input_shape,
    block_name_to_hyperparameters_dict=OrderedDict(
        [
            ("conv2", (64, 3, 2)),
            ("conv3", (128, 8, 2)),
            ("conv4", (256, 36, 2)),
            ("conv5", (512, 3, 1)),
        ]
    ),
    preprocess_input_mode="tf",
)

ResNeXt50 = lambda input_shape: init_resnet(
    stack_fn=stack3,
    preact=False,
    use_bias=False,
    model_name="resnext50",
    input_shape=input_shape,
    block_name_to_hyperparameters_dict=OrderedDict(
        [
            ("conv2", (128, 3, 1)),
            ("conv3", (256, 4, 2)),
            ("conv4", (512, 6, 2)),
            ("conv5", (1024, 3, 2)),
        ]
    ),
    preprocess_input_mode="torch",
)

ResNeXt101 = lambda input_shape: init_resnet(
    stack_fn=stack3,
    preact=False,
    use_bias=False,
    model_name="resnext101",
    input_shape=input_shape,
    block_name_to_hyperparameters_dict=OrderedDict(
        [
            ("conv2", (128, 3, 1)),
            ("conv3", (256, 4, 2)),
            ("conv4", (512, 23, 2)),
            ("conv5", (1024, 3, 2)),
        ]
    ),
    preprocess_input_mode="torch",
)
