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
        tensor = torch.clamp(tensor, 0.0, 1.0)
        
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

    for cams in os.listdir(pseudo_labels_dir):
        print(cams)
        for pids in os.listdir(os.path.join(pseudo_labels_dir, cams)):
            for pseudo_labels in os.listdir(os.path.join(pseudo_labels_dir, cams, pids)):
                parsing_result = Image.open(os.path.join(pseudo_labels_dir, cams, pids, pseudo_labels))

                # parsing_result.putpalette(palette)
                orig_img = Image.open(os.path.join(original_image_dir, cams, pids, pseudo_labels[:-3] + "jpg"))
                parsing_result = np.array(parsing_result)
                h, w = 288, 144
                orig_img = orig_img.resize((w,h), Image.ANTIALIAS).convert('RGBA')
                
                parsing_img = np.expand_dims(np.zeros_like(parsing_result), -1).repeat(3, axis=-1)
                for i in range(parsing_img.shape[0]):
                    for j in range(parsing_img.shape[1]):
                        if parsing_result[i][j] > 6:
                            raise NotImplementedError("class > 6 = {}".format(parsing_result[i][j]))
                        parsing_img[i][j] = color_map[parsing_result[i][j]]
                parsing_img = Image.fromarray(parsing_img)
                parsing_img = parsing_img.resize((w, h), Image.NEAREST)
                parsing_img = parsing_img.convert('RGBA')
                # blend_img = Image.blend(orig_img, parsing_img, 0.5)

                fusion_img = Image.new('RGBA', (w*2, h), (255, 255, 255))
                fusion_img.paste(orig_img, (0, 0))
                fusion_img.paste(parsing_img, (w, 0))
                os.makedirs(os.path.join(visualization_dir, cams, pids), exist_ok=True)
                fusion_img.save(os.path.join(visualization_dir, cams, pids, pseudo_labels[:-3] + "png"))

    return


if __name__ == '__main__':
    main()



