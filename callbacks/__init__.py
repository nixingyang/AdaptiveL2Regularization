import os
import pickle
from itertools import product

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from tensorflow.keras.callbacks import Callback

# Specify the backend of matplotlib
matplotlib.use("Agg")


class HistoryLogger(Callback):

    def __init__(self, output_folder_path):
        super(HistoryLogger, self).__init__()

        self.accumulated_logs_dict = {}
        self.output_folder_path = output_folder_path

        if not os.path.isdir(output_folder_path):
            os.makedirs(output_folder_path)

    def visualize(self, loss_name):
        # Unpack the values
        epoch_to_loss_value_dict = self.accumulated_logs_dict[loss_name]
        epoch_list = sorted(epoch_to_loss_value_dict.keys())
        loss_value_list = [epoch_to_loss_value_dict[epoch] for epoch in epoch_list]
        epoch_list = (np.array(epoch_list) + 1).tolist()

        # Save the figure to disk
        figure = plt.figure()
        if isinstance(loss_value_list[0], dict):
            for metric_name in loss_value_list[0].keys():
                metric_value_list = [
                    loss_value[metric_name] for loss_value in loss_value_list
                ]
                print(
                    "{} {} {:.6f}".format(loss_name, metric_name, metric_value_list[-1])
                )
                plt.plot(
                    epoch_list,
                    metric_value_list,
                    label="{} {:.6f}".format(metric_name, metric_value_list[-1]),
                )
        else:
            print("{} {:.6f}".format(loss_name, loss_value_list[-1]))
            plt.plot(
                epoch_list,
                loss_value_list,
                label="{} {:.6f}".format(loss_name, loss_value_list[-1]),
            )
            plt.ylabel(loss_name)
        plt.xlabel("Epoch")
        plt.grid(True)
        plt.legend(loc="best")
        plt.savefig(os.path.join(self.output_folder_path, "{}.png".format(loss_name)))
        plt.close(figure)

    def on_epoch_end(self, epoch, logs=None):
        # Visualize each figure
        for loss_name, loss_value in logs.items():
            if loss_name not in self.accumulated_logs_dict:
                self.accumulated_logs_dict[loss_name] = {}
            self.accumulated_logs_dict[loss_name][epoch] = loss_value
            self.visualize(loss_name)

        # Save the accumulated_logs_dict to disk
        with open(
            os.path.join(self.output_folder_path, "accumulated_logs_dict.pkl"), "wb"
        ) as file_object:
            pickle.dump(
                self.accumulated_logs_dict, file_object, pickle.HIGHEST_PROTOCOL
            )

        # Delete extra keys due to changes in ProgbarLogger
        loss_name_list = list(logs.keys())
        split_name_list = ["valid", "test"]
        for loss_name, split_name in product(loss_name_list, split_name_list):
            if loss_name.startswith(split_name):
                _ = logs.pop(loss_name)
