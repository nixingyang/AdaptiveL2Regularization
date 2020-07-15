import os
import sys

from tensorflow.keras.layers import BatchNormalization, Conv1D, Conv2D, Dense
from tensorflow.keras.models import Model, clone_model, model_from_json
from tensorflow.keras.regularizers import l2

sys.path.append(os.path.abspath(os.path.join(__file__, "../..")))
from regularizers.adaptation import AdaptiveL1L2


def replicate_model(model, name):
    vanilla_weights = model.get_weights()
    model = clone_model(model)
    model = Model(inputs=model.input, outputs=model.output, name=name)
    model.set_weights(vanilla_weights)
    return model


def specify_regularizers(
    model,
    kernel_regularization_factor=0,
    bias_regularization_factor=0,
    gamma_regularization_factor=0,
    beta_regularization_factor=0,
    use_adaptive_l1_l2_regularizer=False,
    omitted_layer_name_prefix_tuple=(),
):

    def _init_regularizers(
        model,
        kernel_regularization_factor,
        bias_regularization_factor,
        gamma_regularization_factor,
        beta_regularization_factor,
        use_adaptive_l1_l2_regularizer,
        omitted_layer_name_prefix_tuple,
    ):
        print("Initializing regularizers for {} ...".format(model.name))
        for item in model.layers:
            if isinstance(item, Model):
                _init_regularizers(
                    item,
                    kernel_regularization_factor,
                    bias_regularization_factor,
                    gamma_regularization_factor,
                    beta_regularization_factor,
                    use_adaptive_l1_l2_regularizer,
                    omitted_layer_name_prefix_tuple,
                )
            omit_layer = False
            for omitted_layer_name_prefix in omitted_layer_name_prefix_tuple:
                if item.name.startswith(omitted_layer_name_prefix):
                    omit_layer = True
                    break
            if omit_layer:
                continue
            if isinstance(item, (Conv1D, Conv2D, Dense)):
                if kernel_regularization_factor >= 0 and hasattr(
                    item, "kernel_regularizer"
                ):
                    item.kernel_regularizer = (
                        AdaptiveL1L2(amplitude_l2=kernel_regularization_factor)
                        if use_adaptive_l1_l2_regularizer
                        else l2(l=kernel_regularization_factor)
                    )
                if bias_regularization_factor >= 0 and hasattr(
                    item, "bias_regularizer"
                ):
                    if item.use_bias:
                        item.bias_regularizer = (
                            AdaptiveL1L2(amplitude_l2=bias_regularization_factor)
                            if use_adaptive_l1_l2_regularizer
                            else l2(l=bias_regularization_factor)
                        )
            elif isinstance(item, BatchNormalization):
                if gamma_regularization_factor >= 0 and hasattr(
                    item, "gamma_regularizer"
                ):
                    if item.scale:
                        item.gamma_regularizer = (
                            AdaptiveL1L2(amplitude_l2=gamma_regularization_factor)
                            if use_adaptive_l1_l2_regularizer
                            else l2(l=gamma_regularization_factor)
                        )
                if beta_regularization_factor >= 0 and hasattr(
                    item, "beta_regularizer"
                ):
                    if item.center:
                        item.beta_regularizer = (
                            AdaptiveL1L2(amplitude_l2=beta_regularization_factor)
                            if use_adaptive_l1_l2_regularizer
                            else l2(l=beta_regularization_factor)
                        )

    # Initialize regularizers
    _init_regularizers(
        model,
        kernel_regularization_factor,
        bias_regularization_factor,
        gamma_regularization_factor,
        beta_regularization_factor,
        use_adaptive_l1_l2_regularizer,
        omitted_layer_name_prefix_tuple,
    )

    # Reload the model
    # https://github.com/keras-team/keras/issues/2717#issuecomment-447570737
    vanilla_weights = model.get_weights()
    model = model_from_json(
        json_string=model.to_json(), custom_objects={"AdaptiveL1L2": AdaptiveL1L2}
    )
    model.set_weights(vanilla_weights)

    return model
