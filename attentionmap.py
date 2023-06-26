from IPython.terminal.embed import EmbeddedMagics
from numpy.lib.npyio import save
import torch
import os
import numpy as np
from PIL import Image
import argparse

from IPython import embed
from torch._C import dtype
import cv2
from matplotlib import pyplot as plt

h, w = 288, 144

def visualize_grid_attention_v2(img_path, save_path, attention_mask, ratio=1, cmap="jet", save_image=False,
                             save_original_image=False, quality=200):
    """
    img_path:   image file path to load
    save_path:  image file path to save
    attention_mask:  2-D attention map with np.array type, e.g, (h, w) or (w, h)
    ratio:  scaling factor to scale the output h and w
    cmap:  attention style, default: "jet"
    quality:  saved image quality
    """
    print("load image from: ", img_path)
    img = Image.open(img_path, mode='r')
    img_h, img_w = img.size[0], img.size[1]
    plt.subplots(nrows=1, ncols=1, figsize=(0.02 * img_h, 0.02 * img_w))

    # scale the image
    img_h, img_w = int(img.size[0] * ratio), int(img.size[1] * ratio)
    img = img.resize((img_h, img_w))
    plt.imshow(img, alpha=1)
    plt.axis('off')

    # normalize the attention map
    attention_mask = np.array(attention_mask, dtype=np.float32)
    mask = cv2.resize(attention_mask, (img_h, img_w))
    normed_mask = mask / mask.max()
    normed_mask = (normed_mask * 255).astype('uint8')
    plt.imshow(normed_mask, alpha=0.5, interpolation='nearest', cmap=cmap)

    if save_image:
        # build save path
        if not os.path.exists(save_path):
            os.mkdir(save_path)
        img_name = img_path.split('/')[-1].split('.')[0] + ".png"
        img_with_attention_save_path = os.path.join(save_path, img_name)
        
        # pre-process and save image
        print("save image to: " + save_path + " as " + img_name)
        plt.axis('off')
        plt.subplots_adjust(top=1, bottom=0, right=1,  left=0, hspace=0, wspace=0)
        plt.margins(0, 0)
        plt.savefig(img_with_attention_save_path, dpi=quality)

    if save_original_image:
        # build save path
        if not os.path.exists(save_path):
            os.mkdir(save_path)

        # save original image file
        print("save original image at the same time")
        img_name = img_path.split('/')[-1].split('.')[0] + "_original.jpg"
        original_image_save_path = os.path.join(save_path, img_name)
        img.save(original_image_save_path, quality=quality)


def main():

    os.makedirs("vis/heat", exist_ok=True)

    for p in range(64):
        img_path = "vis/ori/{}.png".format(p)
        save_path = "vis/heat/"
        mask = Image.open("vis/ot/{}.png".format(p))
        mask = np.array(mask)
        
        visualize_grid_attention_v2(img_path, save_path, mask, save_image=True)


        # mask_result = Image.open("vis/mask/{}.png".format(p))
        # ot_result = Image.open("vis/ot/{}.png".format(p))
        # part_result = Image.open("vis/part/{}.png".format(p))
        # cam_result = Image.open("vis/cam/{}.png".format(p))
        # erase_result = Image.open("vis/erase/{}.png".format(p))

        # ori_result = np.array(ori_img.convert('RGB'))
        # mask_result = np.array(mask_result)
        # ot_result = np.array(ot_result)
        # part_result = np.array(part_result)
        # cam_result = np.array(cam_result)
        # erase_result = np.array(erase_result)

        # mask_img = np.zeros_like(ori_result)
        # ot_img = np.zeros_like(ori_result)
        # part_img = np.zeros_like(ori_result)
        # cam_img = np.zeros_like(ori_result)
        # erase_img = np.zeros_like(ori_result)

        # mask_img = Image.fromarray(mask_result).resize((w, h), Image.NEAREST).convert('RGBA')
        # mask_img = visualize_grid_attention_v2(ori_img, mask_img)

        # for i in range(ot_img.shape[0]):
        #     for j in range(ot_img.shape[1]):
        #         if ot_result[i][j] > 6:
        #             raise NotImplementedError("class > 6 = {}".format(ot_result[i][j]))
        #         ot_img[i][j] = color_map[ot_result[i][j]]
        # ot_img = Image.fromarray(ot_img).resize((w, h), Image.NEAREST).convert('RGBA')
        

        # for i in range(part_img.shape[0]):
        #     for j in range(part_img.shape[1]):
        #         if part_result[i][j] > 6:
        #             raise NotImplementedError("class > 6 = {}".format(part_result[i][j]))
        #         part_img[i][j] = color_map[part_result[i][j]]
        # part_img = Image.fromarray(part_img).resize((w, h), Image.NEAREST).convert('RGBA')

        # for i in range(cam_img.shape[0]):
        #     for j in range(cam_img.shape[1]):
        #         if cam_result[i][j] > 6:
        #             raise NotImplementedError("class > 6 = {}".format(cam_result[i][j]))
        #         cam_img[i][j] = color_map[cam_result[i][j]]
        # cam_img = Image.fromarray(cam_img).resize((w, h), Image.NEAREST).convert('RGBA')

        # for i in range(erase_img.shape[0]):
        #     for j in range(erase_img.shape[1]):
        #         if erase_result[i][j] > 6:
        #             raise NotImplementedError("class > 6 = {}".format(erase_result[i][j]))
        #         erase_img[i][j] = color_map[erase_result[i][j]]
        # erase_img = Image.fromarray(erase_img).resize((w, h), Image.NEAREST).convert('RGBA')


        # fusion_img = Image.new('RGBA', (w*6, h), (255, 255, 255))
        # fusion_img.paste(ori_img, (0, 0))
        # fusion_img.paste(mask_img, (w, 0))
        # fusion_img.paste(ot_img, (w*2, 0))
        # fusion_img.paste(part_img, (w*3, 0))
        # fusion_img.paste(cam_img, (w*4, 0))
        # fusion_img.paste(erase_img, (w*5, 0))
        # fusion_img.save("vis/res/{}.png".format(p))

        

    return


if __name__ == '__main__':
    main()