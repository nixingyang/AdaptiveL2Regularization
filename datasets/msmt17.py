import os

import pandas as pd


def _load_accumulated_info(
    root_folder_path,
    dataset_folder_name="MSMT17_V2",
    image_folder_name="mask_train_v2",
    list_file_name="list_train.txt",
):
    # https://www.pkuvmc.com/publications/msmt17.html
    dataset_folder_path = os.path.join(root_folder_path, dataset_folder_name)
    image_folder_path = os.path.join(dataset_folder_path, image_folder_name)

    accumulated_info_list = []
    list_file_path = os.path.join(dataset_folder_path, list_file_name)
    with open(list_file_path) as file_object:
        for line_content in file_object:
            image_file_name, identity_ID = line_content.split(" ")
            identity_ID = int(identity_ID)
            image_file_path = os.path.join(image_folder_path, image_file_name)
            assert os.path.isfile(image_file_path), "{} does not exists!".format(
                image_file_path
            )
            assert identity_ID == int(image_file_name.split(os.sep)[0])
            camera_ID = int(image_file_name.split(os.sep)[1].split("_")[2])

            # Append the records
            accumulated_info = {
                "image_file_path": image_file_path,
                "identity_ID": identity_ID,
                "camera_ID": camera_ID,
            }
            accumulated_info_list.append(accumulated_info)

    # Convert list to data frame
    accumulated_info_dataframe = pd.DataFrame(accumulated_info_list)
    return accumulated_info_dataframe


def load_MSMT17(root_folder_path):
    train_accumulated_info_dataframe = _load_accumulated_info(
        root_folder_path=root_folder_path,
        image_folder_name="mask_train_v2",
        list_file_name="list_train.txt",
    )
    valid_accumulated_info_dataframe = _load_accumulated_info(
        root_folder_path=root_folder_path,
        image_folder_name="mask_train_v2",
        list_file_name="list_val.txt",
    )
    train_and_valid_accumulated_info_dataframe = pd.concat(
        [train_accumulated_info_dataframe, valid_accumulated_info_dataframe],
        ignore_index=True,
    )
    test_query_accumulated_info_dataframe = _load_accumulated_info(
        root_folder_path=root_folder_path,
        image_folder_name="mask_test_v2",
        list_file_name="list_query.txt",
    )
    test_gallery_accumulated_info_dataframe = _load_accumulated_info(
        root_folder_path=root_folder_path,
        image_folder_name="mask_test_v2",
        list_file_name="list_gallery.txt",
    )
    return (
        train_and_valid_accumulated_info_dataframe,
        test_query_accumulated_info_dataframe,
        test_gallery_accumulated_info_dataframe,
    )
