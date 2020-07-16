from urllib.request import urlopen

import cv2
import numpy as np
from albumentations import Compose, HorizontalFlip, PadIfNeeded, RandomCrop, Rotate

from .random_erasing import RandomErasing


class BaseImageAugmentor(object):

    def __init__(
        self,
        horizontal_flip_probability=0.5,
        rotate_limit=0,
        image_height=224,
        image_width=224,
        padding_length=20,
        padding_ratio=0,
    ):
        # Initiation
        self.transforms = []
        self.transformer = None

        # Flip the input horizontally
        if horizontal_flip_probability > 0:
            self.transforms.append(HorizontalFlip(p=horizontal_flip_probability))

        # Rotate the input by an angle selected randomly from the uniform distribution
        if rotate_limit > 0:
            self.transforms.append(
                Rotate(
                    limit=rotate_limit, border_mode=cv2.BORDER_CONSTANT, value=0, p=1.0
                )
            )

        # Pad side of the image and crop a random part of it
        if padding_length > 0 or padding_ratio > 0:
            min_height = image_height + int(
                max(padding_length, image_height * padding_ratio)
            )
            min_width = image_width + int(
                max(padding_length, image_width * padding_ratio)
            )
            self.transforms.append(
                PadIfNeeded(
                    min_height=min_height,
                    min_width=min_width,
                    border_mode=cv2.BORDER_CONSTANT,
                    value=0,
                )
            )
            self.transforms.append(RandomCrop(height=image_height, width=image_width))

    def add_transforms(self, additional_transforms):
        self.transforms += additional_transforms

    def compose_transforms(self):
        self.transformer = Compose(transforms=self.transforms)

    def apply_augmentation(self, image_content_array):
        transformed_image_content_list = []
        for image_content in image_content_array:
            transformed_image_content = self.transformer(image=image_content)["image"]
            transformed_image_content_list.append(transformed_image_content)
        return np.array(transformed_image_content_list, dtype=np.uint8)


class RandomErasingImageAugmentor(BaseImageAugmentor):

    def __init__(self, **kwargs):
        super(RandomErasingImageAugmentor, self).__init__(**kwargs)
        additional_transforms = [RandomErasing()]
        self.add_transforms(additional_transforms)


def example():
    print("Loading the image content ...")
    raw_data = urlopen(url="https://avatars3.githubusercontent.com/u/15064790").read()
    raw_data = np.frombuffer(raw_data, np.uint8)
    image_content = cv2.imdecode(raw_data, cv2.IMREAD_COLOR)
    image_content = cv2.cvtColor(image_content, cv2.COLOR_BGR2RGB)
    image_height, image_width = image_content.shape[:2]

    print("Initiating the image augmentor ...")
    image_augmentor = RandomErasingImageAugmentor(
        image_height=image_height, image_width=image_width
    )
    image_augmentor.compose_transforms()

    print("Generating the batch ...")
    image_content_list = [image_content] * 8
    image_content_array = np.array(image_content_list)

    print("Applying data augmentation ...")
    image_content_array = image_augmentor.apply_augmentation(image_content_array)

    print("Visualization ...")
    for image_index, image_content in enumerate(image_content_array, start=1):
        image_content = cv2.cvtColor(image_content, cv2.COLOR_RGB2BGR)
        cv2.imshow("image {}".format(image_index), image_content)
        cv2.waitKey(0)
    cv2.destroyAllWindows()

    print("All done!")


if __name__ == "__main__":
    example()
