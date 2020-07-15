import os
import shutil
import sys
import time
from datetime import datetime

import cv2
import numpy as np
import pandas as pd
import tensorflow as tf
from absl import app, flags
from sklearn.metrics import pairwise_distances
from sklearn.model_selection import GroupShuffleSplit, StratifiedShuffleSplit
from tensorflow.keras import backend as K
from tensorflow.keras.callbacks import Callback, LearningRateScheduler, ModelCheckpoint
from tensorflow.keras.initializers import RandomNormal
from tensorflow.keras.layers import (
    Activation,
    BatchNormalization,
    Concatenate,
    Conv2D,
    Dense,
    GlobalAveragePooling2D,
    Input,
    Lambda,
)
from tensorflow.keras.losses import categorical_crossentropy
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import Sequence

import applications
import image_augmentation
from callbacks import HistoryLogger
from datasets import load_accumulated_info_of_dataset
from evaluation.metrics import compute_CMC_mAP
from evaluation.post_processing.re_ranking_ranklist import re_ranking
from metric_learning.triplet_hermans import batch_hard, cdist
from regularizers.adaptation import InspectRegularizationFactors
from utils.model_utils import replicate_model, specify_regularizers
from utils.vis_utils import summarize_model, visualize_model

flags.DEFINE_string("root_folder_path", "", "Folder path of the dataset.")
flags.DEFINE_string("dataset_name", "Market1501", "Name of the dataset.")
# ["Market1501", "DukeMTMC_reID", "MSMT17"]
flags.DEFINE_string("backbone_model_name", "ResNet50", "Name of the backbone model.")
# ["ResNet50", "ResNet101", "ResNet152",
# "ResNet50V2", "ResNet101V2", "ResNet152V2",
# "ResNeXt50", "ResNeXt101"]
flags.DEFINE_integer(
    "freeze_backbone_for_N_epochs",
    20,
    "Freeze layers in the backbone model for N epochs.",
)
flags.DEFINE_integer("image_width", 128, "Width of the images.")
flags.DEFINE_integer("image_height", 384, "Height of the images.")
flags.DEFINE_integer("region_num", 2, "Number of regions in the regional branch.")
flags.DEFINE_float(
    "kernel_regularization_factor", 0.005, "Regularization factor of kernel."
)
flags.DEFINE_float(
    "bias_regularization_factor", 0.005, "Regularization factor of bias."
)
flags.DEFINE_float(
    "gamma_regularization_factor", 0.005, "Regularization factor of gamma."
)
flags.DEFINE_float(
    "beta_regularization_factor", 0.005, "Regularization factor of beta."
)
flags.DEFINE_bool(
    "use_adaptive_l1_l2_regularizer", True, "Use the adaptive L1L2 regularizer."
)
flags.DEFINE_float(
    "min_value_in_clipping", 0.0, "Minimum value when using the clipping function."
)
flags.DEFINE_float(
    "max_value_in_clipping", 1.0, "Maximum value when using the clipping function."
)
flags.DEFINE_float(
    "validation_size", 0.0, "Proportion or absolute number of validation samples."
)
flags.DEFINE_float(
    "testing_size", 1.0, "Proportion or absolute number of testing groups."
)
flags.DEFINE_integer(
    "evaluate_validation_every_N_epochs",
    1,
    "Evaluate the performance on validation samples every N epochs.",
)
flags.DEFINE_integer(
    "evaluate_testing_every_N_epochs",
    10,
    "Evaluate the performance on testing samples every N epochs.",
)
flags.DEFINE_integer("identity_num_per_batch", 16, "Number of identities in one batch.")
flags.DEFINE_integer("image_num_per_identity", 4, "Number of images of one identity.")
flags.DEFINE_string(
    "learning_rate_mode", "default", "Mode of the learning rate scheduler."
)
# ["constant", "linear", "cosine", "warmup", "default"]
flags.DEFINE_float("learning_rate_start", 2e-4, "Starting learning rate.")
flags.DEFINE_float("learning_rate_end", 2e-4, "Ending learning rate.")
flags.DEFINE_float("learning_rate_base", 2e-4, "Base learning rate.")
flags.DEFINE_integer(
    "learning_rate_warmup_epochs", 10, "Number of epochs to warmup the learning rate."
)
flags.DEFINE_integer(
    "learning_rate_steady_epochs",
    30,
    "Number of epochs to keep the learning rate steady.",
)
flags.DEFINE_float(
    "learning_rate_drop_factor", 10, "Factor to decrease the learning rate."
)
flags.DEFINE_float(
    "learning_rate_lower_bound", 2e-6, "Lower bound of the learning rate."
)
flags.DEFINE_integer("steps_per_epoch", 200, "Number of steps per epoch.")
flags.DEFINE_integer("epoch_num", 200, "Number of epochs.")
flags.DEFINE_integer("workers", 5, "Number of processes to spin up for data generator.")
flags.DEFINE_string(
    "image_augmentor_name", "RandomErasingImageAugmentor", "Name of image augmentor."
)
# ["BaseImageAugmentor", "RandomErasingImageAugmentor"]
flags.DEFINE_bool(
    "use_data_augmentation_in_training", True, "Use data augmentation in training."
)
flags.DEFINE_bool(
    "use_data_augmentation_in_evaluation", False, "Use data augmentation in evaluation."
)
flags.DEFINE_integer(
    "augmentation_num", 1, "Number of augmented samples to use in evaluation."
)
flags.DEFINE_bool(
    "use_horizontal_flipping_in_evaluation",
    True,
    "Use horizontal flipping in evaluation.",
)
flags.DEFINE_bool(
    "use_identity_balancing_in_training", False, "Use identity balancing in training."
)
flags.DEFINE_bool("use_re_ranking", False, "Use the re-ranking method.")
flags.DEFINE_bool("evaluation_only", False, "Only perform evaluation.")
flags.DEFINE_bool(
    "save_data_to_disk",
    False,
    "Save image features, identity ID and camera ID to disk.",
)
flags.DEFINE_string(
    "pretrained_model_file_path", "", "File path of the pretrained model."
)
flags.DEFINE_string(
    "output_folder_path",
    os.path.abspath(
        os.path.join(
            __file__, "../output_{}".format(datetime.now().strftime("%Y_%m_%d"))
        )
    ),
    "Path to directory to output files.",
)
FLAGS = flags.FLAGS


def apply_stratifiedshufflesplit(y, test_size, random_state=0):
    if test_size == 1:
        train_indexes = np.arange(len(y))  # Hacky snippet
        test_indexes = np.arange(len(y))
    else:
        shufflesplit_instance = StratifiedShuffleSplit(
            n_splits=1, test_size=test_size, random_state=random_state
        )
        train_indexes, test_indexes = next(
            shufflesplit_instance.split(np.arange(len(y)), y=y)
        )
    return train_indexes, test_indexes


def apply_groupshufflesplit(groups, test_size, random_state=0):
    groupshufflesplit_instance = GroupShuffleSplit(
        n_splits=1, test_size=test_size, random_state=random_state
    )
    train_indexes, test_indexes = next(
        groupshufflesplit_instance.split(np.arange(len(groups)), groups=groups)
    )
    return train_indexes, test_indexes


def init_model(
    backbone_model_name,
    freeze_backbone_for_N_epochs,
    input_shape,
    region_num,
    attribute_name_to_label_encoder_dict,
    kernel_regularization_factor,
    bias_regularization_factor,
    gamma_regularization_factor,
    beta_regularization_factor,
    use_adaptive_l1_l2_regularizer,
    min_value_in_clipping,
    max_value_in_clipping,
    share_last_block=False,
):

    def _add_objective_module(input_tensor):
        # Add a pooling layer if needed
        if len(K.int_shape(input_tensor)) == 4:
            global_pooling_tensor = GlobalAveragePooling2D()(input_tensor)
        else:
            global_pooling_tensor = input_tensor
        if min_value_in_clipping is not None and max_value_in_clipping is not None:
            global_pooling_tensor = Lambda(
                lambda x: K.clip(
                    x, min_value=min_value_in_clipping, max_value=max_value_in_clipping
                )
            )(global_pooling_tensor)

        # https://arxiv.org/abs/1801.07698v1 Section 3.2.2 Output setting
        # https://arxiv.org/abs/1807.11042
        classification_input_tensor = global_pooling_tensor
        classification_embedding_tensor = BatchNormalization(scale=True, epsilon=2e-5)(
            classification_input_tensor
        )

        # Add categorical crossentropy loss
        label_encoder = attribute_name_to_label_encoder_dict["identity_ID"]
        class_num = len(label_encoder.classes_)
        classification_output_tensor = Dense(
            units=class_num,
            use_bias=False,
            kernel_initializer=RandomNormal(mean=0.0, stddev=0.001),
        )(classification_embedding_tensor)
        classification_output_tensor = Activation("softmax")(
            classification_output_tensor
        )

        # Add miscellaneous loss
        miscellaneous_input_tensor = global_pooling_tensor
        miscellaneous_embedding_tensor = miscellaneous_input_tensor
        miscellaneous_output_tensor = miscellaneous_input_tensor

        return (
            classification_output_tensor,
            classification_embedding_tensor,
            miscellaneous_output_tensor,
            miscellaneous_embedding_tensor,
        )

    def _apply_concatenation(tensor_list):
        if len(tensor_list) == 1:
            return tensor_list[0]
        else:
            return Concatenate()(tensor_list)

    def _triplet_hermans_loss(y_true, y_pred, metric="euclidean", margin="soft"):
        # Create the loss in two steps:
        # 1. Compute all pairwise distances according to the specified metric.
        # 2. For each anchor along the first dimension, compute its loss.
        dists = cdist(y_pred, y_pred, metric=metric)
        loss = batch_hard(dists=dists, pids=tf.argmax(y_true, axis=-1), margin=margin)
        return loss

    # Initiation
    classification_output_tensor_list = []
    classification_embedding_tensor_list = []
    miscellaneous_output_tensor_list = []
    miscellaneous_embedding_tensor_list = []

    # Initiate the early blocks
    model_instantiation = getattr(applications, backbone_model_name, None)
    assert model_instantiation is not None, "Backbone {} is not supported.".format(
        backbone_model_name
    )
    submodel_list, preprocess_input = model_instantiation(input_shape=input_shape)
    vanilla_input_tensor = Input(shape=K.int_shape(submodel_list[0].input)[1:])
    intermediate_output_tensor = vanilla_input_tensor
    for submodel in submodel_list[:-1]:
        if freeze_backbone_for_N_epochs > 0:
            submodel.trainable = False
        intermediate_output_tensor = submodel(intermediate_output_tensor)

    # Initiate the last blocks
    last_block = submodel_list[-1]
    last_block_for_global_branch_model = replicate_model(
        last_block, name="last_block_for_global_branch"
    )
    if freeze_backbone_for_N_epochs > 0:
        last_block_for_global_branch_model.trainable = False
    if share_last_block:
        last_block_for_regional_branch_model = last_block_for_global_branch_model
    else:
        last_block_for_regional_branch_model = replicate_model(
            last_block, name="last_block_for_regional_branch"
        )
        if freeze_backbone_for_N_epochs > 0:
            last_block_for_regional_branch_model.trainable = False

    # Add the global branch
    (
        classification_output_tensor,
        classification_embedding_tensor,
        miscellaneous_output_tensor,
        miscellaneous_embedding_tensor,
    ) = _add_objective_module(
        last_block_for_global_branch_model(intermediate_output_tensor)
    )
    classification_output_tensor_list.append(classification_output_tensor)
    classification_embedding_tensor_list.append(classification_embedding_tensor)
    miscellaneous_output_tensor_list.append(miscellaneous_output_tensor)
    miscellaneous_embedding_tensor_list.append(miscellaneous_embedding_tensor)

    # Add the regional branch
    if region_num > 0:
        # Process each region
        regional_branch_output_tensor = last_block_for_regional_branch_model(
            intermediate_output_tensor
        )
        total_height = K.int_shape(regional_branch_output_tensor)[1]
        region_size = total_height // region_num
        for region_index in np.arange(region_num):
            # Get a slice of feature maps
            start_index = region_index * region_size
            end_index = (region_index + 1) * region_size
            if region_index == region_num - 1:
                end_index = total_height
            sliced_regional_branch_output_tensor = Lambda(
                lambda x, start_index=start_index, end_index=end_index: x[
                    :, start_index:end_index
                ]
            )(regional_branch_output_tensor)

            # Downsampling
            sliced_regional_branch_output_tensor = Conv2D(
                filters=K.int_shape(sliced_regional_branch_output_tensor)[-1]
                // region_num,
                kernel_size=3,
                padding="same",
            )(sliced_regional_branch_output_tensor)
            sliced_regional_branch_output_tensor = Activation("relu")(
                sliced_regional_branch_output_tensor
            )

            # Add the regional branch
            (
                classification_output_tensor,
                classification_embedding_tensor,
                miscellaneous_output_tensor,
                miscellaneous_embedding_tensor,
            ) = _add_objective_module(sliced_regional_branch_output_tensor)
            classification_output_tensor_list.append(classification_output_tensor)
            classification_embedding_tensor_list.append(classification_embedding_tensor)
            miscellaneous_output_tensor_list.append(miscellaneous_output_tensor)
            miscellaneous_embedding_tensor_list.append(miscellaneous_embedding_tensor)

    # Define the merged model
    embedding_tensor_list = [_apply_concatenation(miscellaneous_embedding_tensor_list)]
    embedding_size_list = [
        K.int_shape(embedding_tensor)[1] for embedding_tensor in embedding_tensor_list
    ]
    merged_embedding_tensor = _apply_concatenation(embedding_tensor_list)
    merged_model = Model(
        inputs=[vanilla_input_tensor],
        outputs=classification_output_tensor_list
        + miscellaneous_output_tensor_list
        + [merged_embedding_tensor],
    )
    merged_model = specify_regularizers(
        merged_model,
        kernel_regularization_factor,
        bias_regularization_factor,
        gamma_regularization_factor,
        beta_regularization_factor,
        use_adaptive_l1_l2_regularizer,
    )

    # Define the models for training/inference
    training_model = Model(
        inputs=[merged_model.input],
        outputs=merged_model.output[:-1],
        name="training_model",
    )
    inference_model = Model(
        inputs=[merged_model.input],
        outputs=[merged_model.output[-1]],
        name="inference_model",
    )
    inference_model.embedding_size_list = embedding_size_list

    # Compile the model
    categorical_crossentropy_loss_function = (
        lambda y_true, y_pred: 1.0
        * categorical_crossentropy(
            y_true, y_pred, from_logits=False, label_smoothing=0.1
        )
    )
    classification_loss_function_list = [categorical_crossentropy_loss_function] * len(
        classification_output_tensor_list
    )
    triplet_hermans_loss_function = lambda y_true, y_pred: 1.0 * _triplet_hermans_loss(
        y_true, y_pred
    )
    miscellaneous_loss_function_list = [triplet_hermans_loss_function] * len(
        miscellaneous_output_tensor_list
    )
    training_model.compile_kwargs = {
        "optimizer": Adam(),
        "loss": classification_loss_function_list + miscellaneous_loss_function_list,
    }
    training_model.compile(**training_model.compile_kwargs)

    # Print the summary of the models
    summarize_model(training_model)
    summarize_model(inference_model)

    return training_model, inference_model, preprocess_input


def read_image_file(image_file_path, input_shape):
    # Read image file
    image_content = cv2.imread(image_file_path)

    # Resize the image
    image_content = cv2.resize(image_content, input_shape[:2][::-1])

    # Convert from BGR to RGB
    image_content = cv2.cvtColor(image_content, cv2.COLOR_BGR2RGB)

    return image_content


class TrainDataSequence(Sequence):

    def __init__(
        self,
        accumulated_info_dataframe,
        attribute_name_to_label_encoder_dict,
        preprocess_input,
        input_shape,
        image_augmentor,
        use_data_augmentation,
        use_identity_balancing,
        label_repetition_num,
        identity_num_per_batch,
        image_num_per_identity,
        steps_per_epoch,
    ):
        super(TrainDataSequence, self).__init__()

        # Save as variables
        (
            self.accumulated_info_dataframe,
            self.attribute_name_to_label_encoder_dict,
            self.preprocess_input,
            self.input_shape,
        ) = (
            accumulated_info_dataframe,
            attribute_name_to_label_encoder_dict,
            preprocess_input,
            input_shape,
        )
        (
            self.image_augmentor,
            self.use_data_augmentation,
            self.use_identity_balancing,
        ) = (image_augmentor, use_data_augmentation, use_identity_balancing)
        self.label_repetition_num = label_repetition_num
        (
            self.identity_num_per_batch,
            self.image_num_per_identity,
            self.steps_per_epoch,
        ) = (identity_num_per_batch, image_num_per_identity, steps_per_epoch)

        # Unpack image_file_path and identity_ID
        self.image_file_path_array, self.identity_ID_array = (
            self.accumulated_info_dataframe[
                ["image_file_path", "identity_ID"]
            ].values.transpose()
        )
        self.image_file_path_to_record_index_dict = dict(
            [
                (image_file_path, record_index)
                for record_index, image_file_path in enumerate(
                    self.image_file_path_array
                )
            ]
        )
        self.batch_size = identity_num_per_batch * image_num_per_identity
        self.image_num_per_epoch = self.batch_size * steps_per_epoch

        # Initiation
        self.image_file_path_list_generator = self._get_image_file_path_list_generator()
        self.image_file_path_list = next(self.image_file_path_list_generator)

    def _get_image_file_path_list_generator(self):
        # Map identity ID to image file paths
        identity_ID_to_image_file_paths_dict = {}
        for image_file_path, identity_ID in zip(
            self.image_file_path_array, self.identity_ID_array
        ):
            if identity_ID not in identity_ID_to_image_file_paths_dict:
                identity_ID_to_image_file_paths_dict[identity_ID] = []
            identity_ID_to_image_file_paths_dict[identity_ID].append(image_file_path)

        image_file_path_list = []
        while True:
            # Split image file paths into multiple sections
            identity_ID_to_image_file_paths_in_sections_dict = {}
            for identity_ID in identity_ID_to_image_file_paths_dict:
                image_file_paths = np.array(
                    identity_ID_to_image_file_paths_dict[identity_ID]
                )
                if len(image_file_paths) < self.image_num_per_identity:
                    continue
                np.random.shuffle(image_file_paths)
                section_num = int(len(image_file_paths) / self.image_num_per_identity)
                image_file_paths = image_file_paths[
                    : section_num * self.image_num_per_identity
                ]
                image_file_paths_in_sections = np.split(image_file_paths, section_num)
                identity_ID_to_image_file_paths_in_sections_dict[identity_ID] = (
                    image_file_paths_in_sections
                )

            while (
                len(identity_ID_to_image_file_paths_in_sections_dict)
                >= self.identity_num_per_batch
            ):
                # Choose identity_num_per_batch identity_IDs
                identity_IDs = np.random.choice(
                    list(identity_ID_to_image_file_paths_in_sections_dict.keys()),
                    size=self.identity_num_per_batch,
                    replace=False,
                )
                for identity_ID in identity_IDs:
                    # Get one section
                    image_file_paths_in_sections = (
                        identity_ID_to_image_file_paths_in_sections_dict[identity_ID]
                    )
                    image_file_paths = image_file_paths_in_sections.pop(-1)
                    if (
                        self.use_identity_balancing
                        or len(image_file_paths_in_sections) == 0
                    ):
                        del identity_ID_to_image_file_paths_in_sections_dict[
                            identity_ID
                        ]

                    # Add the entries
                    image_file_path_list += image_file_paths.tolist()

                if len(image_file_path_list) == self.image_num_per_epoch:
                    yield image_file_path_list
                    image_file_path_list = []

    def __len__(self):
        return self.steps_per_epoch

    def __getitem__(self, index):
        label_encoder = self.attribute_name_to_label_encoder_dict["identity_ID"]
        image_content_list, one_hot_encoding_list = [], []
        image_file_path_list = self.image_file_path_list[
            index * self.batch_size : (index + 1) * self.batch_size
        ]
        for image_file_path in image_file_path_list:
            # Read image
            image_content = read_image_file(image_file_path, self.input_shape)
            image_content_list.append(image_content)

            # Get current record from accumulated_info_dataframe
            record_index = self.image_file_path_to_record_index_dict[image_file_path]
            accumulated_info = self.accumulated_info_dataframe.iloc[record_index]
            assert image_file_path == accumulated_info["image_file_path"]

            # Get the one hot encoding vector
            identity_ID = accumulated_info["identity_ID"]
            one_hot_encoding = np.zeros(len(label_encoder.classes_))
            one_hot_encoding[label_encoder.transform([identity_ID])[0]] = 1
            one_hot_encoding_list.append(one_hot_encoding)

        # Construct image_content_array
        image_content_array = np.array(image_content_list)
        if self.use_data_augmentation:
            # Apply data augmentation
            image_content_array = self.image_augmentor.apply_augmentation(
                image_content_array
            )
        # Apply preprocess_input function
        image_content_array = self.preprocess_input(image_content_array)

        # Construct one_hot_encoding_array_list
        one_hot_encoding_array = np.array(one_hot_encoding_list)
        one_hot_encoding_array_list = [
            one_hot_encoding_array
        ] * self.label_repetition_num

        return image_content_array, one_hot_encoding_array_list

    def on_epoch_end(self):
        self.image_file_path_list = next(self.image_file_path_list_generator)


class TestDataSequence(Sequence):

    def __init__(
        self,
        accumulated_info_dataframe,
        preprocess_input,
        input_shape,
        image_augmentor,
        use_data_augmentation,
        batch_size,
    ):
        super(TestDataSequence, self).__init__()

        # Save as variables
        self.accumulated_info_dataframe, self.preprocess_input, self.input_shape = (
            accumulated_info_dataframe,
            preprocess_input,
            input_shape,
        )
        self.image_augmentor, self.use_data_augmentation = (
            image_augmentor,
            use_data_augmentation,
        )

        # Unpack image_file_path and identity_ID
        self.image_file_path_array = self.accumulated_info_dataframe[
            "image_file_path"
        ].values
        self.batch_size = batch_size
        self.steps_per_epoch = int(
            np.ceil(len(self.image_file_path_array) / self.batch_size)
        )

        # Initiation
        self.image_file_path_list = self.image_file_path_array.tolist()
        self.use_horizontal_flipping = False

    def enable_horizontal_flipping(self):
        self.use_horizontal_flipping = True

    def disable_horizontal_flipping(self):
        self.use_horizontal_flipping = False

    def __len__(self):
        return self.steps_per_epoch

    def __getitem__(self, index):
        image_content_list = []
        image_file_path_list = self.image_file_path_list[
            index * self.batch_size : (index + 1) * self.batch_size
        ]
        for image_file_path in image_file_path_list:
            # Read image
            image_content = read_image_file(image_file_path, self.input_shape)
            if self.use_horizontal_flipping:
                image_content = cv2.flip(image_content, 1)
            image_content_list.append(image_content)

        # Construct image_content_array
        image_content_array = np.array(image_content_list)
        if self.use_data_augmentation:
            # Apply data augmentation
            image_content_array = self.image_augmentor.apply_augmentation(
                image_content_array
            )
        # Apply preprocess_input function
        image_content_array = self.preprocess_input(image_content_array)

        return image_content_array


class Evaluator(Callback):

    def __init__(
        self,
        inference_model,
        split_name,
        query_accumulated_info_dataframe,
        gallery_accumulated_info_dataframe,
        preprocess_input,
        input_shape,
        image_augmentor,
        use_data_augmentation,
        augmentation_num,
        use_horizontal_flipping,
        use_re_ranking,
        batch_size,
        workers,
        use_multiprocessing,
        rank_list=(1, 5, 10, 20),
        every_N_epochs=1,
        output_folder_path=None,
    ):
        super(Evaluator, self).__init__()
        if hasattr(self, "_supports_tf_logs"):
            self._supports_tf_logs = True

        self.callback_disabled = (
            query_accumulated_info_dataframe is None
            or gallery_accumulated_info_dataframe is None
        )
        if self.callback_disabled:
            return

        self.inference_model = inference_model
        self.split_name = split_name
        self.query_generator = TestDataSequence(
            query_accumulated_info_dataframe,
            preprocess_input,
            input_shape,
            image_augmentor,
            use_data_augmentation,
            batch_size,
        )
        self.gallery_generator = TestDataSequence(
            gallery_accumulated_info_dataframe,
            preprocess_input,
            input_shape,
            image_augmentor,
            use_data_augmentation,
            batch_size,
        )
        self.query_identity_ID_array, self.query_camera_ID_array = (
            query_accumulated_info_dataframe[
                ["identity_ID", "camera_ID"]
            ].values.transpose()
        )
        self.gallery_identity_ID_array, self.gallery_camera_ID_array = (
            gallery_accumulated_info_dataframe[
                ["identity_ID", "camera_ID"]
            ].values.transpose()
        )
        (
            self.preprocess_input,
            self.input_shape,
            self.image_augmentor,
            self.use_data_augmentation,
            self.augmentation_num,
            self.use_horizontal_flipping,
        ) = (
            preprocess_input,
            input_shape,
            image_augmentor,
            use_data_augmentation,
            augmentation_num,
            use_horizontal_flipping,
        )
        self.use_re_ranking, self.batch_size = use_re_ranking, batch_size
        self.workers, self.use_multiprocessing = workers, use_multiprocessing
        self.rank_list, self.every_N_epochs = rank_list, every_N_epochs
        self.output_file_path = (
            None
            if output_folder_path is None
            else os.path.join(output_folder_path, "{}.npz".format(split_name))
        )

        if not use_data_augmentation and augmentation_num != 1:
            print("Set augmentation_num to 1 since use_data_augmentation is False.")
            self.augmentation_num = 1

        self.metrics = ["cosine"]

    def extract_features(self, data_generator):
        # Extract the accumulated_feature_array
        accumulated_feature_array = None
        for _ in np.arange(self.augmentation_num):
            data_generator.disable_horizontal_flipping()
            feature_array = self.inference_model.predict(
                x=data_generator,
                workers=self.workers,
                use_multiprocessing=self.use_multiprocessing,
            )
            if self.use_horizontal_flipping:
                data_generator.enable_horizontal_flipping()
                feature_array += self.inference_model.predict(
                    x=data_generator,
                    workers=self.workers,
                    use_multiprocessing=self.use_multiprocessing,
                )
                feature_array /= 2
            if accumulated_feature_array is None:
                accumulated_feature_array = feature_array / self.augmentation_num
            else:
                accumulated_feature_array += feature_array / self.augmentation_num
        return accumulated_feature_array

    def split_features(self, accumulated_feature_array):
        # Split the accumulated_feature_array into separate slices
        feature_array_list = []
        for embedding_size_index in np.arange(
            len(self.inference_model.embedding_size_list)
        ):
            if embedding_size_index == 0:
                start_index = 0
                end_index = self.inference_model.embedding_size_list[0]
            else:
                start_index = np.sum(
                    self.inference_model.embedding_size_list[:embedding_size_index]
                )
                end_index = np.sum(
                    self.inference_model.embedding_size_list[: embedding_size_index + 1]
                )
            feature_array = accumulated_feature_array[:, start_index:end_index]
            feature_array_list.append(feature_array)
        return feature_array_list

    def compute_distance_matrix(
        self, query_image_features, gallery_image_features, metric, use_re_ranking
    ):
        # Compute the distance matrix
        query_gallery_distance = pairwise_distances(
            query_image_features, gallery_image_features, metric=metric
        )
        distance_matrix = query_gallery_distance

        # Use the re-ranking method
        if use_re_ranking:
            query_query_distance = pairwise_distances(
                query_image_features, query_image_features, metric=metric
            )
            gallery_gallery_distance = pairwise_distances(
                gallery_image_features, gallery_image_features, metric=metric
            )
            distance_matrix = re_ranking(
                query_gallery_distance, query_query_distance, gallery_gallery_distance
            )

        return distance_matrix

    def on_epoch_end(self, epoch, logs=None):
        if self.callback_disabled or (epoch + 1) % self.every_N_epochs != 0:
            return

        # Extract features
        feature_extraction_start = time.time()
        query_image_features_array = self.extract_features(self.query_generator)
        gallery_image_features_array = self.extract_features(self.gallery_generator)
        feature_extraction_end = time.time()
        feature_extraction_speed = (
            len(query_image_features_array) + len(gallery_image_features_array)
        ) / (feature_extraction_end - feature_extraction_start)
        print(
            "Speed of feature extraction: {:.2f} images per second.".format(
                feature_extraction_speed
            )
        )

        # Check unique values in the features array
        query_unique_values = np.unique(query_image_features_array)
        additional_metrics = []
        if len(query_unique_values) <= 2**8:
            print(
                "Unique values in query_image_features_array: {}".format(
                    query_unique_values
                )
            )
            additional_metrics.append("hamming")

        # Save image features, identity ID and camera ID to disk
        if self.output_file_path is not None:
            np.savez(
                self.output_file_path,
                query_image_features_array=query_image_features_array,
                gallery_image_features_array=gallery_image_features_array,
                query_identity_ID_array=self.query_identity_ID_array,
                gallery_identity_ID_array=self.gallery_identity_ID_array,
                query_camera_ID_array=self.query_camera_ID_array,
                gallery_camera_ID_array=self.gallery_camera_ID_array,
            )

        # Split features
        print("embedding_size_list:", self.inference_model.embedding_size_list)
        query_image_features_list = self.split_features(query_image_features_array)
        gallery_image_features_list = self.split_features(gallery_image_features_array)

        for metric in self.metrics + additional_metrics:
            distance_matrix_list = []
            for query_image_features, gallery_image_features in zip(
                query_image_features_list, gallery_image_features_list
            ):
                distance_matrix = self.compute_distance_matrix(
                    query_image_features,
                    gallery_image_features,
                    metric,
                    self.use_re_ranking,
                )
                distance_matrix_list.append(distance_matrix)

            method_name_list = (np.arange(len(distance_matrix_list)) + 1).tolist()
            for distance_matrix, method_name in zip(
                distance_matrix_list, method_name_list
            ):
                # Compute the CMC and mAP scores
                CMC_score_array, mAP_score = compute_CMC_mAP(
                    distmat=distance_matrix,
                    q_pids=self.query_identity_ID_array,
                    g_pids=self.gallery_identity_ID_array,
                    q_camids=self.query_camera_ID_array,
                    g_camids=self.gallery_camera_ID_array,
                )

                # Append the CMC and mAP scores
                logs[
                    "{}_{}_{}_{}_rank_to_accuracy_dict".format(
                        self.split_name, metric, self.use_re_ranking, method_name
                    )
                ] = dict(
                    [
                        ("rank-{} accuracy".format(rank), CMC_score_array[rank - 1])
                        for rank in self.rank_list
                    ]
                )
                logs[
                    "{}_{}_{}_{}_mAP_score".format(
                        self.split_name, metric, self.use_re_ranking, method_name
                    )
                ] = mAP_score


def learning_rate_scheduler(
    epoch_index,
    epoch_num,
    learning_rate_mode,
    learning_rate_start,
    learning_rate_end,
    learning_rate_base,
    learning_rate_warmup_epochs,
    learning_rate_steady_epochs,
    learning_rate_drop_factor,
    learning_rate_lower_bound,
):
    learning_rate = None
    if learning_rate_mode == "constant":
        assert (
            learning_rate_start == learning_rate_end
        ), "starting and ending learning rates should be equal!"
        learning_rate = learning_rate_start
    elif learning_rate_mode == "linear":
        learning_rate = (learning_rate_end - learning_rate_start) / (
            epoch_num - 1
        ) * epoch_index + learning_rate_start
    elif learning_rate_mode == "cosine":
        assert (
            learning_rate_start > learning_rate_end
        ), "starting learning rate should be higher than ending learning rate!"
        learning_rate = (learning_rate_start - learning_rate_end) / 2 * np.cos(
            np.pi * epoch_index / (epoch_num - 1)
        ) + (learning_rate_start + learning_rate_end) / 2
    elif learning_rate_mode == "warmup":
        learning_rate = (learning_rate_end - learning_rate_start) / (
            learning_rate_warmup_epochs - 1
        ) * epoch_index + learning_rate_start
        learning_rate = np.min((learning_rate, learning_rate_end))
    elif learning_rate_mode == "default":
        if epoch_index < learning_rate_warmup_epochs:
            learning_rate = (learning_rate_base - learning_rate_lower_bound) / (
                learning_rate_warmup_epochs - 1
            ) * epoch_index + learning_rate_lower_bound
        else:
            if learning_rate_drop_factor == 0:
                learning_rate_drop_factor = np.exp(
                    learning_rate_steady_epochs
                    / (epoch_num - learning_rate_warmup_epochs * 2)
                    * np.log(learning_rate_base / learning_rate_lower_bound)
                )
            learning_rate = learning_rate_base / np.power(
                learning_rate_drop_factor,
                int(
                    (epoch_index - learning_rate_warmup_epochs)
                    / learning_rate_steady_epochs
                ),
            )
    else:
        assert False, "{} is an invalid argument!".format(learning_rate_mode)
    learning_rate = np.max((learning_rate, learning_rate_lower_bound))
    return learning_rate


def main(_):
    print("Getting hyperparameters ...")
    print("Using command {}".format(" ".join(sys.argv)))
    flag_values_dict = FLAGS.flag_values_dict()
    for flag_name in sorted(flag_values_dict.keys()):
        flag_value = flag_values_dict[flag_name]
        print(flag_name, flag_value)
    root_folder_path, dataset_name = FLAGS.root_folder_path, FLAGS.dataset_name
    backbone_model_name, freeze_backbone_for_N_epochs = (
        FLAGS.backbone_model_name,
        FLAGS.freeze_backbone_for_N_epochs,
    )
    image_height, image_width = FLAGS.image_height, FLAGS.image_width
    input_shape = (image_height, image_width, 3)
    region_num = FLAGS.region_num
    kernel_regularization_factor = FLAGS.kernel_regularization_factor
    bias_regularization_factor = FLAGS.bias_regularization_factor
    gamma_regularization_factor = FLAGS.gamma_regularization_factor
    beta_regularization_factor = FLAGS.beta_regularization_factor
    use_adaptive_l1_l2_regularizer = FLAGS.use_adaptive_l1_l2_regularizer
    min_value_in_clipping, max_value_in_clipping = (
        FLAGS.min_value_in_clipping,
        FLAGS.max_value_in_clipping,
    )
    validation_size = FLAGS.validation_size
    validation_size = int(validation_size) if validation_size > 1 else validation_size
    use_validation = validation_size != 0
    testing_size = FLAGS.testing_size
    testing_size = int(testing_size) if testing_size > 1 else testing_size
    use_testing = testing_size != 0
    evaluate_validation_every_N_epochs = FLAGS.evaluate_validation_every_N_epochs
    evaluate_testing_every_N_epochs = FLAGS.evaluate_testing_every_N_epochs
    identity_num_per_batch, image_num_per_identity = (
        FLAGS.identity_num_per_batch,
        FLAGS.image_num_per_identity,
    )
    batch_size = identity_num_per_batch * image_num_per_identity
    learning_rate_mode, learning_rate_start, learning_rate_end = (
        FLAGS.learning_rate_mode,
        FLAGS.learning_rate_start,
        FLAGS.learning_rate_end,
    )
    learning_rate_base, learning_rate_warmup_epochs, learning_rate_steady_epochs = (
        FLAGS.learning_rate_base,
        FLAGS.learning_rate_warmup_epochs,
        FLAGS.learning_rate_steady_epochs,
    )
    learning_rate_drop_factor, learning_rate_lower_bound = (
        FLAGS.learning_rate_drop_factor,
        FLAGS.learning_rate_lower_bound,
    )
    steps_per_epoch = FLAGS.steps_per_epoch
    epoch_num = FLAGS.epoch_num
    workers = FLAGS.workers
    use_multiprocessing = workers > 1
    image_augmentor_name = FLAGS.image_augmentor_name
    use_data_augmentation_in_training = FLAGS.use_data_augmentation_in_training
    use_data_augmentation_in_evaluation = FLAGS.use_data_augmentation_in_evaluation
    augmentation_num = FLAGS.augmentation_num
    use_horizontal_flipping_in_evaluation = FLAGS.use_horizontal_flipping_in_evaluation
    use_identity_balancing_in_training = FLAGS.use_identity_balancing_in_training
    use_re_ranking = FLAGS.use_re_ranking
    evaluation_only, save_data_to_disk = FLAGS.evaluation_only, FLAGS.save_data_to_disk
    pretrained_model_file_path = FLAGS.pretrained_model_file_path

    output_folder_path = os.path.abspath(
        os.path.join(
            FLAGS.output_folder_path,
            "{}_{}x{}".format(dataset_name, input_shape[0], input_shape[1]),
            "{}_{}_{}".format(
                backbone_model_name, identity_num_per_batch, image_num_per_identity
            ),
        )
    )
    shutil.rmtree(output_folder_path, ignore_errors=True)
    os.makedirs(output_folder_path)
    print("Recreating the output folder at {} ...".format(output_folder_path))

    print("Loading the annotations of the {} dataset ...".format(dataset_name))
    (
        train_and_valid_accumulated_info_dataframe,
        test_query_accumulated_info_dataframe,
        test_gallery_accumulated_info_dataframe,
        train_and_valid_attribute_name_to_label_encoder_dict,
    ) = load_accumulated_info_of_dataset(
        root_folder_path=root_folder_path, dataset_name=dataset_name
    )

    if use_validation:
        print("Using customized cross validation splits ...")
        train_and_valid_identity_ID_array = train_and_valid_accumulated_info_dataframe[
            "identity_ID"
        ].values
        train_indexes, valid_indexes = apply_stratifiedshufflesplit(
            y=train_and_valid_identity_ID_array, test_size=validation_size
        )
        train_accumulated_info_dataframe = (
            train_and_valid_accumulated_info_dataframe.iloc[train_indexes]
        )
        valid_accumulated_info_dataframe = (
            train_and_valid_accumulated_info_dataframe.iloc[valid_indexes]
        )

        print("Splitting the validation dataset ...")
        valid_identity_ID_array = valid_accumulated_info_dataframe["identity_ID"].values
        gallery_size = len(test_gallery_accumulated_info_dataframe) / (
            len(test_query_accumulated_info_dataframe)
            + len(test_gallery_accumulated_info_dataframe)
        )
        valid_query_indexes, valid_gallery_indexes = apply_stratifiedshufflesplit(
            y=valid_identity_ID_array, test_size=gallery_size
        )
        valid_query_accumulated_info_dataframe = valid_accumulated_info_dataframe.iloc[
            valid_query_indexes
        ]
        valid_gallery_accumulated_info_dataframe = (
            valid_accumulated_info_dataframe.iloc[valid_gallery_indexes]
        )
    else:
        train_accumulated_info_dataframe = train_and_valid_accumulated_info_dataframe
        (
            valid_query_accumulated_info_dataframe,
            valid_gallery_accumulated_info_dataframe,
        ) = (None, None)

    if use_testing:
        if testing_size != 1:
            print("Using a subset from the testing dataset ...")
            test_accumulated_info_dataframe = pd.concat(
                [
                    test_query_accumulated_info_dataframe,
                    test_gallery_accumulated_info_dataframe,
                ],
                ignore_index=True,
            )
            test_identity_ID_array = test_accumulated_info_dataframe[
                "identity_ID"
            ].values
            _, test_query_and_gallery_indexes = apply_groupshufflesplit(
                groups=test_identity_ID_array, test_size=testing_size
            )
            test_query_mask = test_query_and_gallery_indexes < len(
                test_query_accumulated_info_dataframe
            )
            test_gallery_mask = np.logical_not(test_query_mask)
            test_query_indexes, test_gallery_indexes = (
                test_query_and_gallery_indexes[test_query_mask],
                test_query_and_gallery_indexes[test_gallery_mask],
            )
            test_query_accumulated_info_dataframe = (
                test_accumulated_info_dataframe.iloc[test_query_indexes]
            )
            test_gallery_accumulated_info_dataframe = (
                test_accumulated_info_dataframe.iloc[test_gallery_indexes]
            )
    else:
        (
            test_query_accumulated_info_dataframe,
            test_gallery_accumulated_info_dataframe,
        ) = (None, None)

    print("Initiating the model ...")
    training_model, inference_model, preprocess_input = init_model(
        backbone_model_name=backbone_model_name,
        freeze_backbone_for_N_epochs=freeze_backbone_for_N_epochs,
        input_shape=input_shape,
        region_num=region_num,
        attribute_name_to_label_encoder_dict=train_and_valid_attribute_name_to_label_encoder_dict,
        kernel_regularization_factor=kernel_regularization_factor,
        bias_regularization_factor=bias_regularization_factor,
        gamma_regularization_factor=gamma_regularization_factor,
        beta_regularization_factor=beta_regularization_factor,
        use_adaptive_l1_l2_regularizer=use_adaptive_l1_l2_regularizer,
        min_value_in_clipping=min_value_in_clipping,
        max_value_in_clipping=max_value_in_clipping,
    )
    visualize_model(model=training_model, output_folder_path=output_folder_path)

    print("Initiating the image augmentor {} ...".format(image_augmentor_name))
    image_augmentor = getattr(image_augmentation, image_augmentor_name)(
        image_height=image_height, image_width=image_width
    )
    image_augmentor.compose_transforms()

    print("Perform training ...")
    train_generator = TrainDataSequence(
        accumulated_info_dataframe=train_accumulated_info_dataframe,
        attribute_name_to_label_encoder_dict=train_and_valid_attribute_name_to_label_encoder_dict,
        preprocess_input=preprocess_input,
        input_shape=input_shape,
        image_augmentor=image_augmentor,
        use_data_augmentation=use_data_augmentation_in_training,
        use_identity_balancing=use_identity_balancing_in_training,
        label_repetition_num=len(training_model.outputs),
        identity_num_per_batch=identity_num_per_batch,
        image_num_per_identity=image_num_per_identity,
        steps_per_epoch=steps_per_epoch,
    )
    valid_evaluator_callback = Evaluator(
        inference_model=inference_model,
        split_name="valid",
        query_accumulated_info_dataframe=valid_query_accumulated_info_dataframe,
        gallery_accumulated_info_dataframe=valid_gallery_accumulated_info_dataframe,
        preprocess_input=preprocess_input,
        input_shape=input_shape,
        image_augmentor=image_augmentor,
        use_data_augmentation=use_data_augmentation_in_evaluation,
        augmentation_num=augmentation_num,
        use_horizontal_flipping=use_horizontal_flipping_in_evaluation,
        use_re_ranking=use_re_ranking,
        batch_size=batch_size,
        workers=workers,
        use_multiprocessing=use_multiprocessing,
        every_N_epochs=evaluate_validation_every_N_epochs,
        output_folder_path=output_folder_path if save_data_to_disk else None,
    )
    test_evaluator_callback = Evaluator(
        inference_model=inference_model,
        split_name="test",
        query_accumulated_info_dataframe=test_query_accumulated_info_dataframe,
        gallery_accumulated_info_dataframe=test_gallery_accumulated_info_dataframe,
        preprocess_input=preprocess_input,
        input_shape=input_shape,
        image_augmentor=image_augmentor,
        use_data_augmentation=use_data_augmentation_in_evaluation,
        augmentation_num=augmentation_num,
        use_horizontal_flipping=use_horizontal_flipping_in_evaluation,
        use_re_ranking=use_re_ranking,
        batch_size=batch_size,
        workers=workers,
        use_multiprocessing=use_multiprocessing,
        every_N_epochs=evaluate_testing_every_N_epochs,
        output_folder_path=output_folder_path if save_data_to_disk else None,
    )
    inspect_regularization_factors_callback = InspectRegularizationFactors()
    optimal_model_file_path = os.path.join(output_folder_path, "training_model.h5")
    modelcheckpoint_monitor = (
        "test_cosine_False_1_mAP_score"
        if use_testing
        else "valid_cosine_False_1_mAP_score"
    )
    modelcheckpoint_callback = ModelCheckpoint(
        filepath=optimal_model_file_path,
        monitor=modelcheckpoint_monitor,
        mode="max",
        save_best_only=True,
        save_weights_only=False,
        verbose=1,
    )
    learningratescheduler_callback = LearningRateScheduler(
        schedule=lambda epoch_index: learning_rate_scheduler(
            epoch_index=epoch_index,
            epoch_num=epoch_num,
            learning_rate_mode=learning_rate_mode,
            learning_rate_start=learning_rate_start,
            learning_rate_end=learning_rate_end,
            learning_rate_base=learning_rate_base,
            learning_rate_warmup_epochs=learning_rate_warmup_epochs,
            learning_rate_steady_epochs=learning_rate_steady_epochs,
            learning_rate_drop_factor=learning_rate_drop_factor,
            learning_rate_lower_bound=learning_rate_lower_bound,
        ),
        verbose=1,
    )
    if len(pretrained_model_file_path) > 0:
        assert os.path.isfile(pretrained_model_file_path)
        print("Loading weights from {} ...".format(pretrained_model_file_path))
        # Hacky workaround for the issue with "load_weights"
        if use_adaptive_l1_l2_regularizer:
            _ = training_model.test_on_batch(train_generator[0])
        # Load weights from the pretrained model
        training_model.load_weights(pretrained_model_file_path)
    if evaluation_only:
        print("Freezing the whole model in the evaluation_only mode ...")
        training_model.trainable = False
        training_model.compile(**training_model.compile_kwargs)

        assert testing_size == 1, "Use all testing samples for evaluation!"
        historylogger_callback = HistoryLogger(
            output_folder_path=os.path.join(output_folder_path, "evaluation")
        )
        training_model.fit(
            x=train_generator,
            steps_per_epoch=1,
            callbacks=[
                inspect_regularization_factors_callback,
                valid_evaluator_callback,
                test_evaluator_callback,
                historylogger_callback,
            ],
            epochs=1,
            workers=workers,
            use_multiprocessing=use_multiprocessing,
            verbose=2,
        )
    else:
        if freeze_backbone_for_N_epochs > 0:
            print(
                "Freeze layers in the backbone model for {} epochs.".format(
                    freeze_backbone_for_N_epochs
                )
            )
            historylogger_callback = HistoryLogger(
                output_folder_path=os.path.join(output_folder_path, "training_A")
            )
            training_model.fit(
                x=train_generator,
                steps_per_epoch=steps_per_epoch,
                callbacks=[
                    valid_evaluator_callback,
                    test_evaluator_callback,
                    learningratescheduler_callback,
                    historylogger_callback,
                ],
                epochs=freeze_backbone_for_N_epochs,
                workers=workers,
                use_multiprocessing=use_multiprocessing,
                verbose=2,
            )

            print("Unfreeze layers in the backbone model.")
            for item in training_model.layers:
                item.trainable = True
            training_model.compile(**training_model.compile_kwargs)

        print("Perform conventional training for {} epochs.".format(epoch_num))
        historylogger_callback = HistoryLogger(
            output_folder_path=os.path.join(output_folder_path, "training_B")
        )
        training_model.fit(
            x=train_generator,
            steps_per_epoch=steps_per_epoch,
            callbacks=[
                inspect_regularization_factors_callback,
                valid_evaluator_callback,
                test_evaluator_callback,
                modelcheckpoint_callback,
                learningratescheduler_callback,
                historylogger_callback,
            ],
            epochs=epoch_num,
            workers=workers,
            use_multiprocessing=use_multiprocessing,
            verbose=2,
        )

        if not os.path.isfile(optimal_model_file_path):
            print("Saving model to {} ...".format(optimal_model_file_path))
            training_model.save(optimal_model_file_path)

    print("All done!")


if __name__ == "__main__":
    app.run(main)
