from dl_toolbox.augmentations import *

def d4_transformation():

    out = OneOf(
        transforms=[
            NoOp(),
            D4(p=1)
        ],
        transforms_ps=[1, 1]
    )

    return out


def color_transformation():

    out = OneOf(
        transforms=[
            NoOp(),
            OneOf(
                transforms=[
                    Gamma(p=1),
                    Contrast(p=1),
                    Saturation(p=1),
                    Brightness(p=1)
                ],
                transforms_ps=[1, 1, 1, 1])
        ],
        transforms_ps=[1, 1]
    )

    return out

def d4_color_transformation():

    T = Compose([
        D4(p=1),
        OneOf(
            transforms=[
                Gamma(p=1),
                Contrast(p=1),
                Saturation(p=1),
                Brightness(p=1)
            ],
            transforms_ps=[1,1,1,1])
    ])

    out = OneOf(
        transforms= [
            NoOp(),
            T
        ],
        transforms_ps=[1,1]
    )

    return out

def get_transforms(name: str):

    if name == 'd4_color':
        return d4_color_transformation()
    elif name == 'd4':
        return d4_transformation()
    elif name == 'color':
        return color_transformation()
    return NoOp()
