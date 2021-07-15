from dl_toolbox.lightning_modules import SupervisedBaseline, MeanTeacher
import rasterio as rio
import matplotlib.pyplot as plt
from dl_toolbox.torch_datasets import *
import cv2
import torch
from dl_toolbox.utils import get_tiles
from argparse import ArgumentParser
import torchvision.transforms.functional as F
from dl_toolbox.augmentations import *

def infer(net,
          device,
          nb_class,
          img_file,
          height,
          width,
          tile_height,
          tile_width,
          col_step,
          row_step,
          batch_size):

    windows = []
    tiles = []
    for window in get_tiles(width=tile_width,
                            height=tile_height,
                            col_step=col_step,
                            row_step=row_step,
                            nols=width,
                            nrows=height):

        tile = img_file.read(window=window, out_dtype=np.float32) / 255
        tiles.append(torch.as_tensor(tile))
        windows.append(window)

    nb_tiles = len(tiles)
    l = list(range(0, nb_tiles, batch_size))+[nb_tiles]

    preds = []
    pred_windows = []
    for i, j in zip(l[:-1], l[1:]):

        batch = torch.stack(tiles[i:j]).to(device)

        for angle in [0,90,270]:
            for ph in [0, 1]:
                for pv in [0, 1]:
                    for pt in [0, 1]:
                        t = Compose([
                            Rotate(p=1, angles=(angle,)),
                            Hflip(p=ph),
                            Vflip(p=pv),
                            Transpose(p=pt)
                        ])
                        aug_batch = t(batch)[0]
                        with torch.no_grad():
                            aug_pred = net(aug_batch).detach()
                        anti_t = Compose([
                            Transpose(p=pt),
                            Vflip(p=pv),
                            Hflip(p=ph),
                            Rotate(p=1, angles=(-angle,))
                        ])
                        pred = anti_t(aug_pred)[0].cpu().numpy()
                        preds += [np.squeeze(e, axis=0) for e in np.split(pred, pred.shape[0], axis=0)]
                        pred_windows += windows[i:j]

    stitched_pred = np.zeros(shape=(nb_class, height, width))
    nb_tiles = np.zeros(shape=(height, width))
    for pred, window in zip(preds, pred_windows):
        stitched_pred[:, window.row_off:window.row_off+window.width,
        window.col_off:window.col_off+window.height] += pred
        nb_tiles[window.row_off:window.row_off+window.width,
        window.col_off:window.col_off+window.height] += 1
    avg_pred = np.argmax(stitched_pred, axis=0)

    return avg_pred


if __name__ == "__main__":

    parser = ArgumentParser()
    parser.add_argument("--ckpt_path", type=str)
    parser.add_argument("--image_path", type=str)
    parser.add_argument("--output", type=str)
    parser.add_argument("--label_path", type=str)
    args = parser.parse_args()
    args_dict = vars(args)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    ckpt = torch.load(args.ckpt_path)
    module = SupervisedBaseline()
    module.load_state_dict(ckpt['state_dict'])
    net = module.network.to(device).eval()

    image_file = rasterio.open(args.image_path)

    width = 500
    height = 500

    image = image_file.read(window=Window(col_off=0,
                                          row_off=0,
                                          width=width,
                                          height=height),
                            out_dtype=np.float32) / 255

    avg_pred = infer(net=net,
                     device=device,
                     nb_class=module.num_classes,
                     img_file=image_file,
                     height=height,
                     width=width,
                     tile_height=128,
                     tile_width=128,
                     col_step=128,
                     row_step=128,
                     batch_size=16)

    pred_profile = image_file.profile
    pred_profile.update(
        height=height,
        width=width,
        count=1,
    )
    with rasterio.open(args.output, 'w', **pred_profile) as dst:
        dst.write(np.uint8(avg_pred),
                  window=Window(col_off=0,
                                row_off=0,
                                width=width,
                                height=height),
                  indexes=1)

    bool_avg_pred = avg_pred.astype(bool)

    label_file = rasterio.open(args.label_path)

    label = label_file.read(window=Window(col_off=0,
                                          row_off=0,
                                          width=width,
                                          height=height),
                            out_dtype=np.uint8)

    gt = AustinLabeled.colors_to_labels(label)
    gt = np.argmax(gt, axis=0).astype(bool)

    overlay = image.copy().transpose(1,2,0)
    image = image.transpose(1,2,0)
    # Correct predictions painted with green
    overlay[gt & bool_avg_pred] = np.array([0, 250, 0], dtype=overlay.dtype)
    # Misses painted with red
    overlay[gt & ~bool_avg_pred] = np.array([250, 0, 0], dtype=overlay.dtype)
    # False alarm painted with blue
    overlay[~gt & bool_avg_pred] = np.array([0, 0, 250], dtype=overlay.dtype)

    overlay = cv2.addWeighted(image, 0.5, overlay, 0.5, 0)

    plt.switch_backend("TkAgg")

    f, ax = plt.subplots(2, 2, figsize=(16, 16))

    ax[0, 0].imshow(image)
    ax[0, 0].set_title('Original image', fontsize=16)

    ax[1, 0].imshow(gt)
    ax[1, 0].set_title('Ground truth', fontsize=16)

    ax[0, 1].imshow(avg_pred)
    ax[0, 1].set_title('Model predictions', fontsize=16)

    ax[1, 1].imshow(overlay)
    ax[1, 1].set_title('Prediction quality', fontsize=16)

    plt.tight_layout()
    plt.show()




