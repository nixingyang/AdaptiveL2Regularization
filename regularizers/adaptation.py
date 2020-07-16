import numpy as np
import tensorflow as tf
from tensorflow.keras import backend as K
from tensorflow.keras.callbacks import Callback
from tensorflow.keras.initializers import Constant
from tensorflow.keras.layers import Layer
from tensorflow.keras.models import Model


class AdaptiveL1L2(Layer):

    def __init__(
        self, amplitude_l1=None, amplitude_l2=None, omega_l1=1, omega_l2=1, **kwargs
    ):
        super(AdaptiveL1L2, self).__init__(**kwargs)
        self.amplitude_l1 = amplitude_l1
        self.amplitude_l2 = amplitude_l2
        self.omega_l1 = omega_l1
        self.omega_l2 = omega_l2
        self.preprocess_function = lambda x: amplitude_l2 * np.clip(
            0.2 * omega_l2 * x + 0.5, 0.0, 1.0
        )  # NB: This function is only applicable to L2.

    def build(self, input_shape):
        self.l1_regularization_factor = (
            None  # pylint: disable=attribute-defined-outside-init
        )
        if self.amplitude_l1 is not None:
            self.l1_regularization_factor = (
                self.add_weight(  # pylint: disable=attribute-defined-outside-init
                    name="l1_regularization_factor", initializer=Constant(0)
                )
            )
        self.l2_regularization_factor = (
            None  # pylint: disable=attribute-defined-outside-init
        )
        if self.amplitude_l2 is not None:
            self.l2_regularization_factor = (
                self.add_weight(  # pylint: disable=attribute-defined-outside-init
                    name="l2_regularization_factor", initializer=Constant(0)
                )
            )
        super(AdaptiveL1L2, self).build(input_shape)

    def call(self, inputs):  # pylint: disable=arguments-differ
        regularization = 0.0
        if self.l1_regularization_factor is not None:
            regularization += (
                self.amplitude_l1
                * K.hard_sigmoid(self.omega_l1 * self.l1_regularization_factor)
                * tf.reduce_sum(tf.abs(inputs))
            )
        if self.l2_regularization_factor is not None:
            regularization += (
                self.amplitude_l2
                * K.hard_sigmoid(self.omega_l2 * self.l2_regularization_factor)
                * tf.reduce_sum(tf.square(inputs))
            )
        return regularization

    def get_config(self):
        config = {
            "amplitude_l1": self.amplitude_l1,
            "amplitude_l2": self.amplitude_l2,
            "omega_l1": self.omega_l1,
            "omega_l2": self.omega_l2,
        }
        base_config = super(AdaptiveL1L2, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


class InspectRegularizationFactors(Callback):

    def forward(self, model, logs):
        for item in model.layers:
            if isinstance(item, Model):
                self.forward(item, logs)
            for regularizer_name in [
                "kernel_regularizer",
                "bias_regularizer",
                "gamma_regularizer",
                "beta_regularizer",
            ]:
                if not hasattr(item, regularizer_name):
                    continue
                regularizer = getattr(item, regularizer_name)
                if not isinstance(regularizer, AdaptiveL1L2):
                    continue
                preprocess_function = regularizer.preprocess_function
                for variable_name in [
                    "l1_regularization_factor",
                    "l2_regularization_factor",
                ]:
                    regularization_factor = getattr(regularizer, variable_name)
                    if regularization_factor is not None:
                        logs[
                            "{}_{}_{}".format(
                                item.name, regularizer_name, variable_name
                            )
                        ] = preprocess_function(K.get_value(regularization_factor))

    def on_epoch_end(self, epoch, logs=None):  # @UnusedVariable
        self.forward(self.model, logs)
