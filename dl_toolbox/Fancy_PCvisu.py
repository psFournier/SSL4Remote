from torch_datasets import IsprsVLabeled
from torch.utils.data import DataLoader
import albumentations as A
from matplotlib import pyplot as plt
from albumentations.core.transforms_interface import ImageOnlyTransform
import random
from albumentations import functional as F
import numpy as np

class MistakenFancyPCA(ImageOnlyTransform):

    def __init__(self, alpha=0.1, always_apply=False, p=0.5):
        super(MistakenFancyPCA, self).__init__(always_apply=always_apply, p=p)
        self.alpha = alpha

    def apply(self, img, alpha=0.1, **params):
        if img.dtype != np.uint8:
            raise TypeError("Image must be RGB image in uint8 format.")

        orig_img = img.astype(float).copy()

        img = img / 255.0  # rescale to 0 to 1 range

        # flatten image to columns of RGB
        img_rs = img.reshape(-1, 3)
        # img_rs shape (640000, 3)

        # center mean
        img_centered = img_rs - np.mean(img_rs, axis=0)

        # paper says 3x3 covariance matrix
        img_cov = np.cov(img_centered, rowvar=False)

        # eigen values and eigen vectors
        eig_vals, eig_vecs = np.linalg.eigh(img_cov)

        # sort values and vector
        sort_perm = eig_vals[::-1].argsort()
        eig_vals[::-1].sort()
        eig_vecs = eig_vecs[:, sort_perm]

        # get [p1, p2, p3]
        m1 = np.column_stack((eig_vecs))

        # get 3x1 matrix of eigen values multiplied by random variable draw from normal
        # distribution with mean of 0 and standard deviation of 0.1
        m2 = np.zeros((3, 1))
        # according to the paper alpha should only be draw once per augmentation (not once per channel)
        # alpha = np.random.normal(0, alpha_std)

        # broad cast to speed things up
        # m2[:, 0] = alpha * eig_vals[:]
        # Need alpha to be the same to compare with CorrectFancyPCA
        m2[:, 0] = np.array(4.) * eig_vals[:]

        # this is the vector that we're going to add to each pixel in a moment
        add_vect = np.matrix(m1) * np.matrix(m2)

        for idx in range(3):  # RGB
            orig_img[..., idx] += add_vect[idx] * 255

        # for image processing it was found that working with float 0.0 to 1.0
        # was easier than integers between 0-255
        # orig_img /= 255.0
        orig_img = np.clip(orig_img, 0.0, 255.0)

        # orig_img *= 255
        orig_img = orig_img.astype(np.uint8)

        return orig_img

    def get_params(self):
        return {"alpha": random.gauss(0, self.alpha)}

    def get_transform_init_args_names(self):
        return ("alpha",)

class CorrectFancyPCA(ImageOnlyTransform):
    """Augment RGB image using FancyPCA from Krizhevsky's paper
    "ImageNet Classification with Deep Convolutional Neural Networks"

    Args:
        alpha (float):  how much to perturb/scale the eigen vecs and vals.
            scale is samples from gaussian distribution (mu=0, sigma=alpha)

    Targets:
        image

    Image types:
        3-channel uint8 images only

    Credit:
        http://papers.nips.cc/paper/4824-imagenet-classification-with-deep-convolutional-neural-networks.pdf
        https://deshanadesai.github.io/notes/Fancy-PCA-with-Scikit-Image
        https://pixelatedbrian.github.io/2018-04-29-fancy_pca/
    """

    def __init__(self, eigenvectors, eigenvalues, alpha=(0.1, 0.1, 0.1),
                 always_apply=False, p=0.5):
        super(CorrectFancyPCA, self).__init__(always_apply=always_apply, p=p)
        self.alpha = alpha
        # eigenvalues and eigenvectors must be the output of np.linalg.eigh(
        # cov) for cov the covariance matrix of an array of pixels
        self.eigenvalues = eigenvalues
        self.eigenvectors = eigenvectors

    def apply(self, img, alpha=(0.1, 0.1, 0.1), **params):

        if img.dtype != np.uint8:
            raise TypeError("Image must be RGB image in uint8 format.")

        orig_img = img.astype(float).copy()

        sort_perm = self.eigenvalues[::-1].argsort()
        self.eigenvalues[::-1].sort()
        eig_vecs = self.eigenvectors[:, sort_perm]
        m1 = np.column_stack((eig_vecs))
        # get 3x1 matrix of eigen values multiplied by random variable draw from normal
        # distribution with mean of 0 and standard deviation of 0.1
        m2 = np.zeros((3, 1))
        # broad cast to speed things up
        # m2[:, 0] = alpha * self.eigenvalues[:]
        m2[:, 0] = np.array((4., 4., 4.)) * self.eigenvalues[:]
        # this is the vector that we're going to add to each pixel in a moment
        add_vect = np.matrix(m1) * np.matrix(m2)
        for idx in range(3):  # RGB
            orig_img[..., idx] += add_vect[idx] * 255
        # for image processing it was found that working with float 0.0 to 1.0
        # was easier than integers between 0-255
        # orig_img /= 255.0
        orig_img = np.clip(orig_img, 0.0, 255.0)
        # orig_img *= 255
        orig_img = orig_img.astype(np.uint8)

        return orig_img

    def get_params(self):
        return {"alpha": np.random.normal(0, self.alpha)}

    def get_transform_init_args_names(self):
        return ("alpha",)

def visualize(image, mask, original_image=None, original_mask=None):
    fontsize = 18

    if original_image is None and original_mask is None:
        f, ax = plt.subplots(2, 1, figsize=(8, 8))

        ax[0].imshow(image)
        ax[1].imshow(mask)
    else:
        f, ax = plt.subplots(2, 2, figsize=(8, 8))

        ax[0, 0].imshow(original_image)
        ax[0, 0].set_title('Incorrect FancyPCA', fontsize=fontsize)

        ax[1, 0].imshow(original_mask)
        ax[1, 0].set_title('Original mask', fontsize=fontsize)

        ax[0, 1].imshow(image)
        ax[0, 1].set_title('Correct FancyPCA', fontsize=fontsize)

        ax[1, 1].imshow(mask)
        ax[1, 1].set_title('Transformed mask', fontsize=fontsize)

ISPRS_EIGENVALS = np.array([0.00073204, 0.01302568, 0.06742553])
ISPRS_EIGENVECS = np.array(
    [[-0.03848249, -0.65840048, -0.75168338],
     [-0.66870032, 0.57594731, -0.47023885],
    [ 0.74253551, 0.48455496, -0.4624365]]
)
aug = CorrectFancyPCA(eigenvalues=ISPRS_EIGENVALS,
                      eigenvectors=ISPRS_EIGENVECS,
                      alpha=(2., 2., 2.),
                      p=1)
mistaken = MistakenFancyPCA(p=1)

ds = IsprsVLabeled(
    data_path='/home/pierre/Documents/ONERA/ai4geo/ISPRS_VAIHINGEN',
    idxs=list(range(2)),
    crop=128,
    augmentations=A.NoOp()
)

dl = DataLoader(
    ds,
    batch_size=1,
)

image, mask = next(iter(dl))
image = image[0,...].numpy()
mask = mask[0,...].numpy()
augmented = aug(image=image, mask=mask)
mist = mistaken(image=image, mask=mask)
visualize(augmented['image'], augmented['mask'], original_image=mist['image'],
          original_mask=mist['mask'])
plt.show()
