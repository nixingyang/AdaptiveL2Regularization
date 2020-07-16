import os

from tensorflow.keras.layers import Layer
from tensorflow.keras.models import Model
from tensorflow.keras.utils import plot_model


def summarize_model(model):
    # Summarize the model at hand
    identifier = "{}_{}".format(model.name, id(model))
    print("Summarizing {} ...".format(identifier))
    model.summary()

    # Summarize submodels
    for item in model.layers:
        if isinstance(item, Model):
            summarize_model(item)


def visualize_model(model, output_folder_path):
    # Visualize the model at hand
    identifier = "{}_{}".format(model.name, id(model))
    print("Visualizing {} ...".format(identifier))
    try:
        # TODO: Wait for patches from upstream.
        # https://github.com/tensorflow/tensorflow/issues/38988
        model._layers = [  # pylint: disable=protected-access
            item
            for item in model._layers  # pylint: disable=protected-access
            if isinstance(item, Layer)
        ]
        plot_model(
            model,
            show_shapes=True,
            show_layer_names=True,
            to_file=os.path.join(output_folder_path, "{}.png".format(identifier)),
        )
    except Exception as exception:  # pylint: disable=broad-except
        print(exception)
        print("Failed to plot {}.".format(identifier))

    # Visualize submodels
    for item in model.layers:
        if isinstance(item, Model):
            visualize_model(item, output_folder_path)
