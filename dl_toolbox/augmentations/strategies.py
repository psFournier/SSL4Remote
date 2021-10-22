from dl_toolbox.augmentations import *

def hard_transformation():

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

    if name == 'hard':
        return hard_transformation()
    if name == 'mixup':
        return Mixup()
    return NoOp()
