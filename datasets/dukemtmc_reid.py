import glob
import os

import pandas as pd


def _load_accumulated_info(
    root_folder_path,
    dataset_folder_name="DukeMTMC-reID",
    image_folder_name="bounding_box_train",
):
    """
    References:
    https://drive.google.com/file/d/1jjE85dRCMOgRtvJ5RQV9-Afs-2_5dY3O/view
    https://stackoverflow.com/questions/25010369/wget-curl-large-file-from-google-drive
    gdrive download 1jjE85dRCMOgRtvJ5RQV9-Afs-2_5dY3O
    7za x DukeMTMC-reID.zip
    sha256sum DukeMTMC-reID.zip
    932ae18937b6a77bc59846d4fb00da4ee02cdda93329ca0537ad899a569e3505  DukeMTMC-reID.zip
    """
    dataset_folder_path = os.path.join(root_folder_path, dataset_folder_name)
    image_folder_path = os.path.join(dataset_folder_path, image_folder_name)

    image_file_path_list = sorted(glob.glob(os.path.join(image_folder_path, "*.jpg")))
    if image_folder_name == "bounding_box_train":
        assert len(image_file_path_list) == 16522
    elif image_folder_name == "bounding_box_test":
        assert len(image_file_path_list) == 17661
    elif image_folder_name == "query":
        assert len(image_file_path_list) == 2228
    else:
        assert False, "{} is an invalid argument!".format(image_folder_name)

    accumulated_info_list = []
    for image_file_path in image_file_path_list:
        image_file_name = image_file_path.split(os.sep)[-1]
        identity_ID = int(image_file_name.split("_")[0])
        camera_ID = int(image_file_name.split("_")[1][1])
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


def load_DukeMTMC_reID(root_folder_path):
    train_and_valid_accumulated_info_dataframe = _load_accumulated_info(
        root_folder_path=root_folder_path, image_folder_name="bounding_box_train"
    )
    test_gallery_accumulated_info_dataframe = _load_accumulated_info(
        root_folder_path=root_folder_path, image_folder_name="bounding_box_test"
    )
    test_query_accumulated_info_dataframe = _load_accumulated_info(
        root_folder_path=root_folder_path, image_folder_name="query"
    )
    return (
        train_and_valid_accumulated_info_dataframe,
        test_query_accumulated_info_dataframe,
        test_gallery_accumulated_info_dataframe,
    )
