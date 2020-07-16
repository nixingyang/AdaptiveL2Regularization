import os
from collections import OrderedDict

from sklearn.preprocessing import LabelEncoder

from .dukemtmc_reid import load_DukeMTMC_reID
from .market1501 import load_Market1501
from .msmt17 import load_MSMT17


def _get_root_folder_path():
    root_folder_path_list = [
        os.path.expanduser("~/Documents/Local Storage/Dataset"),
        "/sgn-data/MLG/nixingyang/Dataset",
    ]
    root_folder_path_mask = [
        os.path.isdir(folder_path) for folder_path in root_folder_path_list
    ]
    root_folder_path = root_folder_path_list[root_folder_path_mask.index(True)]
    return root_folder_path


def _get_attribute_name_to_label_encoder_dict(accumulated_info_dataframe):
    attribute_name_to_label_encoder_dict = OrderedDict({})
    accumulated_info_dataframe = accumulated_info_dataframe.drop(
        columns=["image_file_path", "camera_ID"]
    )
    for attribute_name in accumulated_info_dataframe.columns:
        label_encoder = LabelEncoder()
        label_encoder.fit(accumulated_info_dataframe[attribute_name].values)
        attribute_name_to_label_encoder_dict[attribute_name] = label_encoder
    return attribute_name_to_label_encoder_dict


def load_accumulated_info_of_dataset(root_folder_path, dataset_name):
    if not os.path.isdir(root_folder_path):
        root_folder_path = _get_root_folder_path()
        print("Use {} as root_folder_path ...".format(root_folder_path))

    dataset_name_to_load_function_dict = {
        "Market1501": load_Market1501,
        "DukeMTMC_reID": load_DukeMTMC_reID,
        "MSMT17": load_MSMT17,
    }
    assert dataset_name in dataset_name_to_load_function_dict
    load_function = dataset_name_to_load_function_dict[dataset_name]
    (
        train_and_valid_accumulated_info_dataframe,
        test_query_accumulated_info_dataframe,
        test_gallery_accumulated_info_dataframe,
    ) = load_function(root_folder_path=root_folder_path)

    assert (
        not train_and_valid_accumulated_info_dataframe.isnull().values.any()
    )  # All fields contain value
    train_and_valid_attribute_name_to_label_encoder_dict = (
        _get_attribute_name_to_label_encoder_dict(
            train_and_valid_accumulated_info_dataframe
        )
    )

    return (
        train_and_valid_accumulated_info_dataframe,
        test_query_accumulated_info_dataframe,
        test_gallery_accumulated_info_dataframe,
        train_and_valid_attribute_name_to_label_encoder_dict,
    )
