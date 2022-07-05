import torchvision.transforms.functional as F


class ImagenetNormalize:

    def __call__(self, img, label=None):

        img = F.normalize(
                img,
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
                )
        return img, label

