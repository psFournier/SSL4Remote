import os
import numpy as np
import json
import PIL
from PIL import Image, ImageDraw
from rasterio.enums import Resampling
import rasterio
import random
import csv


def resizefile(root, XY, output, nativeresolution, outputresolution=50.0):

    i = 0
    upscale_factor = nativeresolution / outputresolution

    for name in XY:

        x, y = XY[name]

        with rasterio.open(root + "/" + x) as imagefile:

            out_profile = imagefile.profile
            out_height = int(imagefile.height * upscale_factor)
            out_width = int(imagefile.width * upscale_factor)

            # resample data to target shape
            data = imagefile.read(
                out_shape=(
                    imagefile.count,
                    out_height,
                    out_width
                ),
                resampling=Resampling.bilinear
            )

            # scale image transform
            transform = imagefile.transform * imagefile.transform.scale(
                (imagefile.width / data.shape[-1]),
                (imagefile.height / data.shape[-2])
            )

            out_profile.update(
                transform=transform,
                height=out_height,
                width=out_width
            )

        with rasterio.open(output + "/" + str(i) + "_x.tif", 'w', **out_profile) as dstimage:

            dstimage.write(data)

        with rasterio.open(root + "/" + y) as labelfile:

            out_profile = labelfile.profile
            out_height = int(labelfile.height * upscale_factor)
            out_width = int(labelfile.width * upscale_factor)

            # resample data to target shape
            data = labelfile.read(
                out_shape=(
                    labelfile.count,
                    out_height,
                    out_width
                ),
                resampling=Resampling.bilinear
            )

            transform = labelfile.transform * labelfile.transform.scale(
                (labelfile.width / data.shape[-1]),
                (labelfile.height / data.shape[-2])
            )

            out_profile.update(
                transform=transform,
                height=out_height,
                width=out_width
            )

        with rasterio.open(output + "/" + str(i) + "_y.tif", 'w', **out_profile) as dst_label:

            dst_label.write(data)

        i+=1


# XY = {
#     0: ("austin/train/15_x.tif", "austin/train/15_y.tif")
# }
# resizefile('/home/pierre/Documents/ONERA/ai4geo/miniworld_tif', XY, '/home/pierre/Documents/ONERA/ai4geo', 30)

availabledata = [
    # "isprs",
    # "airs",
    "inria",
    # "semcity"
]
root = "/scratch_ai4geo/DATASETS/"
rootminiworld = "/scratch_ai4geo/miniworld_tif/"



def makepath(name):
    os.makedirs(rootminiworld + name)
    os.makedirs(rootminiworld + name + "/train")
    os.makedirs(rootminiworld + name + "/test")

if "inria" in availabledata:
    print("export inria")
    # towns = ["austin", "chicago", "kitsap", "tyrol-w", "vienna"]
    towns = ["bellingham", "bloomington", "innsbruck", "sfo", "tyrol-e"]
    for town in towns:
        makepath(town)

        XY = {}
        for i in range(20):
            XY[i] = (
                "images/" + town + str(1 + i) + ".tif",
                "gt/" + town + str(1 + i) + ".tif",
            )
        resizefile(
            root + "INRIA/AerialImageDataset/train/",
            XY,
            rootminiworld + town + "/train/",
            30,
            )

        XY = {}
        for i in range(16):
            XY[i] = (
                "images/" + town + str(21 + i) + ".tif",
                "gt/" + town + str(21 + i) + ".tif",
            )
        resizefile(
            root + "INRIA/AerialImageDataset/train/",
            XY,
            rootminiworld + town + "/test/",
            30,
            )

if "airs" in availabledata:
    print("export airs")
    makepath("christchurch")

    hack = "trainval/"

    for flag, flag2 in [("test", "val"), ("train", "train")]:
        XY = {}
        allname = os.listdir(root + "AIRS/" + hack + flag2 + "/image")
        for name in allname:
            XY[name] = (
                "image/" + name[0:-4] + ".tif",
                "label/" + name[0:-4] + "_vis.tif",
            )
        resizefile(
            root + "AIRS/" + hack + flag2,
            XY,
            rootminiworld + "christchurch/" + flag + "/",
            7.5,
            )

if "isprs" in availabledata:

    print("export isprs potsdam")
    makepath("potsdam")

    names = {}
    names["train"] = [
        "top_potsdam_2_10_",
        "top_potsdam_2_11_",
        "top_potsdam_2_12_",
        "top_potsdam_3_10_",
        "top_potsdam_3_11_",
        "top_potsdam_3_12_",
        "top_potsdam_4_10_",
        "top_potsdam_4_11_",
        "top_potsdam_4_12_",
        "top_potsdam_5_10_",
        "top_potsdam_5_11_",
        "top_potsdam_5_12_",
        "top_potsdam_6_7_",
        "top_potsdam_6_8_",
    ]
    names["test"] = [
        "top_potsdam_6_9_",
        "top_potsdam_6_10_",
        "top_potsdam_6_11_",
        "top_potsdam_6_12_",
        "top_potsdam_7_7_",
        "top_potsdam_7_8_",
        "top_potsdam_7_9_",
        "top_potsdam_7_10_",
        "top_potsdam_7_11_",
        "top_potsdam_7_12_",
    ]

    for flag in ["train", "test"]:
        XY = {}
        for name in names[flag]:
            XY[name] = (
                "2_Ortho_RGB/" + name + "RGB.tif",
                "5_Labels_for_participants/" + name + "label.tif"
            )
        resizefile(root + "ISPRS_POTSDAM/", XY, rootminiworld + "potsdam/" + flag, 5)

if "semcity" in availabledata:
    print("export toulouse")
    makepath("toulouse")

    hack = "../"

    names = {}
    names["train"] = ["04", "08"]
    names["test"] = ["03", "07"]

    for flag in ["train", "test"]:
        XY = {}
        for name in names[flag]:
            XY[name] = (
                "TLS_BDSD_M_" + name + ".tif",
                "TLS_GT_" + name + ".tif"
            )
        resizefile(root + hack + "SEMCITY_TOULOUSE/", XY, rootminiworld + "toulouse/" + flag, 50)