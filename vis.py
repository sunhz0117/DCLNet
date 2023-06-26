from IPython.terminal.embed import EmbeddedMagics
import torch
import os
import numpy as np
from PIL import Image
import argparse

from IPython import embed
from torch._C import dtype


class UnNormalize(object):
    """UnNormalize a tensor image with mean and standard deviation.
    Given mean: ``(mean[1],...,mean[n])`` and std: ``(std[1],..,std[n])`` for ``n``
    channels, this transform will normalize each channel of the input
    ``torch.*Tensor`` i.e.,
    ``output[channel] = input[channel] * std[channel] + mean[channel]``

    .. note::
        This transform acts out of place, i.e., it does not mutate the input tensor.

    Args:
        mean (sequence): Sequence of means for each channel.
        std (sequence): Sequence of standard deviations for each channel.
        inplace(bool,optional): Bool to make this operation in-place.

    """

    def __init__(self, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225], inplace=False):
        self.mean = mean
        self.std = std
        self.inplace = inplace

    def __call__(self, tensor):
        
        if not torch.is_tensor(tensor):
            raise TypeError('tensor should be a torch tensor. Got {}.'.format(type(tensor)))

        if tensor.ndimension() != 3:
            raise ValueError('Expected tensor to be a tensor image of size (C, H, W). Got tensor.size() = '
                            '{}.'.format(tensor.size()))

        if not self.inplace:
            tensor = tensor.clone()

        dtype = tensor.dtype
        mean = torch.as_tensor(self.mean, dtype=dtype, device=tensor.device)
        std = torch.as_tensor(self.std, dtype=dtype, device=tensor.device)
        if (std == 0).any():
            raise ValueError('std evaluated to zero after conversion to {}, leading to division by zero.'.format(dtype))
        if mean.ndim == 1:
            mean = mean[:, None, None]
        if std.ndim == 1:
            std = std[:, None, None]
        # tensor.sub_(mean).div_(std)
        tensor.mul_(std).add_(mean)
        tensor = torch.clamp(tensor, min=0.0, max=1.0)
        
        # for i in range(137,138):
        #     for j in range(211,218):
        #         print(tensor[:, j, i])
        
        return tensor

def get_arguments():
    """Parse all the arguments provided from the CLI.
    Returns:
      A list of parsed arguments.
    """
    parser = argparse.ArgumentParser(description="Visualization")

    parser.add_argument("--pseudo_labels_dir", type=str, default='./cluster_mask/origin_ly2')
    parser.add_argument("--visualization_dir", type=str, default='./cluster_mask/vis_origin_ly2')
    parser.add_argument("--original_image_dir", type=str, default='./data')
    parser.add_argument("--num_part", type=int, default=7)

    return parser.parse_args()


def get_palette(num_cls):
    """ Returns the color map for visualizing the segmentation mask.
    Args:
        num_cls: Number of classes
    Returns:
        The color map
    """
    n = num_cls
    palette = [0] * (n * 3)
    for j in range(0, n):
        lab = j
        palette[j * 3 + 0] = 0
        palette[j * 3 + 1] = 0
        palette[j * 3 + 2] = 0
        i = 0
        while lab:
            palette[j * 3 + 0] |= (((lab >> 0) & 1) << (7 - i))
            palette[j * 3 + 1] |= (((lab >> 1) & 1) << (7 - i))
            palette[j * 3 + 2] |= (((lab >> 2) & 1) << (7 - i))
            i += 1
            lab >>= 3
    return palette

color_map = {
    0: (0, 0, 0),
    1: (254, 67, 101),
    2: (252, 157, 154),
    3: (249, 205, 173),
    4: (200, 200, 169),
    5: (131, 175, 155),
    6: (38, 188, 213),
}

def main():
    args = get_arguments()

    num_part = args.num_part
    pseudo_labels_dir = args.pseudo_labels_dir
    visualization_dir = args.visualization_dir
    original_image_dir = args.original_image_dir

    os.makedirs(args.visualization_dir, exist_ok=True)

    palette = get_palette(num_part)

    for p in range(64):
        h, w = 72, 36
        ori_img = Image.open("vis/ori/{}.png".format(p)).resize((w,h), Image.ANTIALIAS)
        mask_result = Image.open("vis/mask/{}.png".format(p))
        ot_result = Image.open("vis/ot/{}.png".format(p))
        part_result = Image.open("vis/part/{}.png".format(p))
        cam_result = Image.open("vis/cam/{}.png".format(p))
        erase_result = Image.open("vis/erase/{}.png".format(p))

        ori_result = np.array(ori_img.convert('RGB'))
        mask_result = np.array(mask_result)
        ot_result = np.array(ot_result)
        part_result = np.array(part_result)
        cam_result = np.array(cam_result)
        erase_result = np.array(erase_result)

        mask_img = np.zeros_like(ori_result)
        ot_img = np.zeros_like(ori_result)
        part_img = np.zeros_like(ori_result)
        cam_img = np.zeros_like(ori_result)
        erase_img = np.zeros_like(ori_result)

        for i in range(mask_img.shape[0]):
            for j in range(mask_img.shape[1]):
                if mask_result[i][j] > 6:
                    raise NotImplementedError("class > 6 = {}".format(mask_result[i][j]))
                mask_img[i][j] = color_map[mask_result[i][j]]
        mask_img = Image.fromarray(mask_img).resize((w, h), Image.NEAREST).convert('RGBA')

        for i in range(ot_img.shape[0]):
            for j in range(ot_img.shape[1]):
                if ot_result[i][j] > 6:
                    raise NotImplementedError("class > 6 = {}".format(ot_result[i][j]))
                ot_img[i][j] = color_map[ot_result[i][j]]
        ot_img = Image.fromarray(ot_img).resize((w, h), Image.NEAREST).convert('RGBA')
        

        for i in range(part_img.shape[0]):
            for j in range(part_img.shape[1]):
                if part_result[i][j] > 6:
                    raise NotImplementedError("class > 6 = {}".format(part_result[i][j]))
                part_img[i][j] = color_map[part_result[i][j]]
        part_img = Image.fromarray(part_img).resize((w, h), Image.NEAREST).convert('RGBA')

        for i in range(cam_img.shape[0]):
            for j in range(cam_img.shape[1]):
                if cam_result[i][j] > 6:
                    raise NotImplementedError("class > 6 = {}".format(cam_result[i][j]))
                cam_img[i][j] = color_map[cam_result[i][j]]
        cam_img = Image.fromarray(cam_img).resize((w, h), Image.NEAREST).convert('RGBA')

        for i in range(erase_img.shape[0]):
            for j in range(erase_img.shape[1]):
                if erase_result[i][j] > 6:
                    raise NotImplementedError("class > 6 = {}".format(erase_result[i][j]))
                erase_img[i][j] = color_map[erase_result[i][j]]
        erase_img = Image.fromarray(erase_img).resize((w, h), Image.NEAREST).convert('RGBA')


        fusion_img = Image.new('RGBA', (w*6, h), (255, 255, 255))
        fusion_img.paste(ori_img, (0, 0))
        fusion_img.paste(mask_img, (w, 0))
        fusion_img.paste(ot_img, (w*2, 0))
        fusion_img.paste(part_img, (w*3, 0))
        fusion_img.paste(cam_img, (w*4, 0))
        fusion_img.paste(erase_img, (w*5, 0))
        fusion_img.save("vis/res/{}.png".format(p))

        

    return


if __name__ == '__main__':
    main()