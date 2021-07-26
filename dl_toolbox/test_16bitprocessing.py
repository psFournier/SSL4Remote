import rasterio
import numpy as np
import matplotlib.pyplot as plt
import albumentations as A
import cv2

src = rasterio.open('/home/pierre/Documents/ONERA/ai4geo/miniworld_tif/kitsap/train/10_x.tif')
img = src.read(out_dtype=np.float32)
# img = np.uint8(img[[4,3,2], :, :]/16).transpose(1,2,0)
img = np.uint8(img).transpose(1,2,0)
# img1 = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
# m, M = np.percentile(img1[:, :, 2], [0, 100])
# out = np.copy(img1).astype(np.float32)
# out[:, :, 2] = np.clip(((img1[:, :, 2] - m) / (M - m)), 0, 1) * 255
# img1 = cv2.cvtColor(np.uint8(out), cv2.COLOR_HSV2RGB)
# img1 = A.functional.adjust_brightness_torchvision(img1, 2)
# plt.imshow(np.uint8(img1))

img2 = A.functional.clahe(img)
f, ax = plt.subplots(1, 2, figsize=(15, 15))
ax[0].imshow(img)
ax[1].imshow(img2)


def histogramnormalization(
        im, removecentiles=2, tile=0, stride=0, vmin=1, vmax=-1, verbose=True, pivot=None
):
    if pivot is None:
        if verbose:
            print("extracting pivot")
        if tile <= 0 or stride <= 0 or tile > stride:
            allvalues = list(im.flatten())
        else:
            allvalues = []
            for row in range(0, im.shape[0] - tile, stride):
                for col in range(0, im.shape[1] - tile, stride):
                    allvalues += list(im[row : row + tile, col : col + tile].flatten())

        ## remove "no data"
        if vmin < vmax:
            allvalues = [v for v in allvalues if vmin <= v and v <= vmax]

        if verbose:
            print("sorting pivot")
        allvalues = sorted(allvalues)
        n = len(allvalues)
        allvalues = allvalues[0 : int((100 - removecentiles) * n / 100)]
        allvalues = allvalues[int(removecentiles * n / 100) :]

        n = len(allvalues)
        k = n // 255

        pivot = [0] + [allvalues[i] for i in range(0, n, k)]

    assert len(pivot) >= 255

    if verbose:
        print("normalization")
    out = np.uint8(np.zeros(im.shape, dtype=int))
    for i in range(1, 255):
        if i % 10 == 0 and verbose:
            print("normalization in progress", i, "/255")
        out = np.maximum(out, np.uint8(im > pivot[i]) * i)

    if verbose:
        print("normalization succeed")
    return np.uint8(out)
#
# r = histogramnormalization(np.int16(src.read(4)*16))
# g = histogramnormalization(np.int16(src.read(3)*16))
# b = histogramnormalization(np.int16(src.read(2)*16))
# x = np.stack([r, g, b], axis=2)
# plt.imshow(x)


# flat = img.flatten()
# hist = get_histogram(flat,65536)
# #plt.plot(hist)
#
# cs = cumsum(hist)
# # re-normalize cumsum values to be between 0-255
#
# # numerator & denomenator
# nj = (cs - cs.min()) * 65535
# N = cs.max() - cs.min()
#
# # re-normalize the cdf
# cs = nj / N
# cs = cs.astype('uint16')
# img_new = cs[flat]
# #plt.hist(img_new, bins=65536)
# #plt.show(block=True)
# img_new = np.reshape(img_new, img.shape)
# cv2.imwrite("contrast.tif",img_new)
# eq = A.Equalize(p=1)
# equalized = eq(image=rgb)['image']

# out = np.zeros_like(hsv).astype(np.float32)
# for i in range(3):
#     m, M = np.percentile(rgb[i, :, :], [2, 98])
#     m = rgb[i, :, :].min()
#     M = rgb[i, :, :].max()
#
#     t = np.clip((rgb[i, :, :] - m) / (M - m), 0, 1)
#     out[i, :, :] = t
# out = out*(2**8-1)
#
# out = rgb_t / 255
# hsv
# out = A.functional._equalize_cv(np.uint8(out))
# plt.imshow(img)
plt.show()