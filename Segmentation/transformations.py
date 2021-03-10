import numpy as np


def normalize_uint8(inp: np.ndarray):
    output = inp / 255.
    return output


def unsqueeze(inp: np.ndarray):
    output = np.expand_dims(inp, axis=0)
    return output


class Normalize:
    """Normalize uint type images"""

    def __init__(self,
              transform_input=True,
              transform_target=False):
        self.transform_input = transform_input
        self.transform_target = transform_target

    def __call__(self, input, target):
        output = normalize_uint8(input)
        return output, target

    def __rep__(self):
        return str({self.__class__.__name__: self.__dict__})


class Unsqueeze:
    """Unsqueeze uint type images"""

    def __init__(self,
              transform_input=True,
              transform_target=False):
        self.transform_input = transform_input
        self.transform_target = transform_target

    def __call__(self, input, target):
        output = unsqueeze(input)
        return output, target

    def __rep__(self):
        return str({self.__class__.__name__: self.__dict__})


class Compose:
    """Compose several transforms together"""

    def __init__(self, transforms: list):
        self.transforms = transforms

    def __call__(self, inp, target):
        for t in self.transforms:
            inp, target = t(inp, target)
        return inp, target

    def __repr__(self): return str([transform for transform in self.transforms])

