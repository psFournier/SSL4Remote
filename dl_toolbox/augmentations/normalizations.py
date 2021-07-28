


class Hflip:

    def __call__(self, img, label=None):

        if torch.rand(1).item() < self.p:
            img = F.hflip(img)
            if label is not None: label = F.hflip(label)
            return img, label

        return img, label