import dl_toolbox.augmentations as aug 

aug_dict = {
    'no': aug.NoOp,
    'd4': aug.D4,
    'hflip': aug.Hflip,
    'vflip': aug.Vflip,
    'd1flip': aug.Transpose1,
    'd2flip': aug.Transpose2,
    'rot90': aug.Rot90,
    'rot180': aug.Rot180,
    'rot270': aug.Rot270,
    'saturation': aug.Saturation,
    'sharpness': aug.Sharpness,
    'contrast': aug.Contrast,
    'gamma': aug.Gamma,
    'brightness': aug.Brightness,
    'color': aug.Color
}

#def d4_transformation():
#
#    out = OneOf(
#        transforms=[
#            D4(p=1)
#        ],
#        transforms_ps=[1]
#    )
#
#    return out
#
#
#def color_transformation():
#
#    out = OneOf(
#        transforms=[
#            NoOp(),
#            OneOf(
#                transforms=[
#                    Gamma(p=1),
#                    Contrast(p=1),
#                    Saturation(p=1),
#                    Brightness(p=1)
#                ],
#                transforms_ps=[1, 1, 1, 1])
#        ],
#        transforms_ps=[1, 1]
#    )
#
#    return out
#
#def d4_color_transformation():
#
#    T = Compose([
#        D4(p=1),
#        OneOf(
#            transforms=[
#                Gamma(p=1),
#                Contrast(p=1),
#                Saturation(p=1),
#                Brightness(p=1)
#            ],
#            transforms_ps=[1,1,1,1])
#    ])
#
#    out = OneOf(
#        transforms= [
#            NoOp(),
#            T
#        ],
#        transforms_ps=[1,1]
#    )
#
#    return out

def get_transforms(name: str):
    
    parts = name.split('_')
    aug_list = []
    for part in parts:
        if part.startswith('color'):
            bounds = part.split('-')[-1]
            augment = aug.Color(bound=0.1*int(bounds))
        else:
            augment = aug_dict[part]()
        aug_list.append(augment)
    return aug.Compose(aug_list)
#
#    if parts[-1]=='color':
#        bounds=
#    if name == 'mixup':
#        return Mixup()
#    if name == 'cutmix':
#        return Cutmix()
#    if name == 'd4_color':
#        return d4_color_transformation()
#    elif name == 'd4':
#        return d4_transformation()
#    elif name == 'color':
#        return color_transformation()
#    return NoOp()
