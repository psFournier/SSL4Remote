import numpy as np
from torch import nn
from torch_datasets import IsprsVaihingen
import os
import rasterio

# nb_bat, nb_non_bat = 0, 0
# for path in IsprsVaihingen.label_paths:
#     with rasterio.open(
#         os.path.join(
#             '/home/pierre/Documents/ONERA/ai4geo/ISPRS_VAIHINGEN', path
#         )
#     ) as label_file:
#         label = label_file.read(out_dtype=np.uint8)
#         label = label.transpose(1, 2, 0)
#         label = IsprsVaihingen.colors_to_labels(label)
#         nb_bat += np.sum((label == 1).astype(int))
#         nb_non_bat += np.sum((label == 0).astype(int))
#
# print(nb_bat, nb_non_bat)
l = [(1172392, 3867608), (28523547, 151476453), (39993440, 140006560),
     (7147644, 172852356), (8904818, 171095182), (57070159, 122929841),
     (148316583, 1481907417), (76940893, 309545207), (7714337, 107425363),
     (57437323, 402513077), (17442114, 84008586), (4574576, 19617040),
     (295497, 3704503), (89008522, 726308904), (2183814, 15816186),
     (3982071, 17230251), (906476, 1653524), (945851, 3554149), (4807477, 13192523),
     (627806, 3692194), (1044655, 7955345), (1320554, 7679446), (406526, 17593474)]
A, B = 0, 0
for a, b in l:
    A+=a
    B+=b
print(A, B)