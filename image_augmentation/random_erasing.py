import numpy as np
from albumentations import ImageOnlyTransform


def apply_random_erasing(image_content, sl, sh, r1, mean, max_attempt_num):
    # Make a copy of the input image since we don't want to modify it directly
    image_content = image_content.copy()
    image_height, image_width = image_content.shape[:-1]
    image_area = image_height * image_width
    for _ in range(max_attempt_num):
        target_area = np.random.uniform(sl, sh) * image_area
        aspect_ratio = np.random.uniform(r1, 1 / r1)
        erasing_height = int(np.round(np.sqrt(target_area * aspect_ratio)))
        erasing_width = int(np.round(np.sqrt(target_area / aspect_ratio)))
        if erasing_width < image_width and erasing_height < image_height:
            starting_height = np.random.randint(0, image_height - erasing_height)
            starting_width = np.random.randint(0, image_width - erasing_width)
            image_content[
                starting_height : starting_height + erasing_height,
                starting_width : starting_width + erasing_width,
            ] = (
                np.array(mean, dtype=np.float32) * 255
            )
            break
    return image_content


class RandomErasing(ImageOnlyTransform):
    """
    References:
    https://arxiv.org/abs/1708.04896
    https://github.com/zhunzhong07/Random-Erasing/blob/master/transforms.py
    https://github.com/albumentations-team/albumentations/blob/0.4.0/albumentations/augmentations/transforms.py#L1492-L1569
    """

    def __init__(
        self,
        sl=0.02,
        sh=0.4,
        r1=0.3,
        mean=(0.4914, 0.4822, 0.4465),
        max_attempt_num=100,
        always_apply=False,
        p=0.5,
    ):
        super(RandomErasing, self).__init__(always_apply, p)
        self.sl = sl
        self.sh = sh
        self.r1 = r1
        self.mean = mean
        self.max_attempt_num = max_attempt_num

    def apply(
        self, image, sl, sh, r1, mean, max_attempt_num, **params
    ):  # pylint: disable=arguments-differ
        return apply_random_erasing(image, sl, sh, r1, mean, max_attempt_num)

    def get_params_dependent_on_targets(self, params):
        return {
            "sl": self.sl,
            "sh": self.sh,
            "r1": self.r1,
            "mean": self.mean,
            "max_attempt_num": self.max_attempt_num,
        }

    @property
    def targets_as_params(self):
        return ["image"]

    def get_transform_init_args_names(self):
        return ("sl", "sh", "r1", "mean", "max_attempt_num")
