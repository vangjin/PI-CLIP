# -*- coding:UTF-8 -*-
import matplotlib.pyplot as plt

from pytorch_grad_cam import GradCAM
import torch
import clip
from PIL import Image
import numpy as np
import cv2
import os

from tqdm import tqdm
from pytorch_grad_cam.utils.image import scale_cam_image
from utils import parse_xml_to_dict, scoremap2bbox
from clip.clip_text import class_names, new_class_names, new_class_names_coco#, imagenet_templates
import argparse
from lxml import etree
import torch.nn.functional as F
from torch import multiprocessing
from torchvision.transforms import Compose, Resize, CenterCrop, ToTensor, Normalize, RandomHorizontalFlip
try:
    from torchvision.transforms import InterpolationMode
    BICUBIC = InterpolationMode.BICUBIC
except ImportError:
    BICUBIC = Image.BICUBIC
import warnings
warnings.filterwarnings("ignore")
_CONTOUR_INDEX = 1 if cv2.__version__.split('.')[0] == '3' else 0

def reshape_transform(tensor, height=28, width=28):
    tensor = tensor.permute(1, 0, 2)
    result = tensor[:, 1:, :].reshape(tensor.size(0), height, width, tensor.size(2))

    # Bring the channels to the first dimension,
    # like in CNNs.
    result = result.transpose(2, 3).transpose(1, 2)
    return result

def split_dataset(dataset, n_splits):
    if n_splits == 1:
        return [dataset]
    part = len(dataset) // n_splits
    dataset_list = []
    for i in range(n_splits - 1):
        dataset_list.append(dataset[i*part:(i+1)*part])
    dataset_list.append(dataset[(i+1)*part:])

    return dataset_list

class ClipOutputTarget:
    def __init__(self, category):
        self.category = category
    def __call__(self, model_output):
        if len(model_output.shape) == 1:
            return model_output[self.category]
        return model_output[:, self.category]


def _convert_image_to_rgb(image):
    return image.convert("RGB")

def _transform_resize(h, w):
    return Compose([
        Resize((h,w), interpolation=BICUBIC),
        _convert_image_to_rgb,
        ToTensor(),
        Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)),
    ])

def img_ms_and_flip(que_img, ori_height, ori_width, scales=[1.0], patch_size=16):

    for scale in scales:
        preprocess = _transform_resize(int(np.ceil(scale * int(ori_height) / patch_size) * patch_size), int(np.ceil(scale * int(ori_width) / patch_size) * patch_size))
        que_img = que_img.cpu().detach().numpy().astype(np.uint8)
        que_img = Image.fromarray(que_img.transpose(1,2,0))
        image = preprocess(que_img)
    return image


def get_img_cam(que_img, tmp_que_name, que_class, model, bg_text_features, fg_text_features, cam, annotation_root, flag=None):
    model = model.cuda()
    bg_text_features = bg_text_features.cuda()
    fg_text_features = fg_text_features.cuda()
    refined_cam_all_scales = []
    for i in range(0, len(tmp_que_name)):
        que_name = tmp_que_name[i]
        if 'VOC' in annotation_root:
            xmlfile = os.path.join(annotation_root, str(que_name))
            xmlfile = xmlfile.replace('.jpg', '.xml')
            with open(xmlfile) as fid:
                xml_str = fid.read()
            xml = etree.fromstring(xml_str)
            data = parse_xml_to_dict(xml)["annotation"]

            ori_width = int(data['size']['width'])
            ori_height = int(data['size']['height'])

            label_list = []
            label_id_list = []
            for obj in data["object"]:
                obj["name"] = new_class_names[class_names.index(obj["name"])]
                if obj["name"] not in label_list:
                    label_list.append(obj["name"])
                    label_id_list.append(new_class_names.index(obj["name"]))
        else:
            ori_height, ori_width = np.asarray(que_img[i].cpu().detach()).shape[1:]
            if flag == False:
                tmp_label_img_path = os.path.join(annotation_root, 'val2014')
                label_img_path = os.path.join(tmp_label_img_path, que_name).replace('jpg', 'png')
            else:
                tmp_label_img_path = os.path.join(annotation_root, 'train2014')
                label_img_path = os.path.join(tmp_label_img_path, que_name).replace('jpg', 'png')
            label_img = cv2.imread(label_img_path, cv2.IMREAD_GRAYSCALE)
            label_id_list = np.unique(label_img).tolist()
            if 0 in label_id_list:
                label_id_list.remove(0)
            if 255 in label_id_list:
                label_id_list.remove(255)
            label_id_list = [x - 1 for x in label_id_list]

            label_list = []
            for lid in label_id_list:
                label_list.append(new_class_names_coco[int(lid)])
            
        if que_class[i] not in label_id_list:
            return [torch.full((64,64),255).float().cuda()]
            
        image = img_ms_and_flip(que_img[i], ori_height, ori_width, scales=[1.0])

        image = image.unsqueeze(0)
        h, w = image.shape[-2], image.shape[-1]
        image = image.cuda()
        image_features, attn_weight_list = model.encode_image(image, h, w)

        bg_features_temp = bg_text_features[label_id_list].cuda()
        fg_features_temp = fg_text_features[label_id_list].cuda()
        text_features_temp = torch.cat([fg_features_temp, bg_features_temp], dim=0)
        input_tensor = [image_features, text_features_temp, h, w]

        for idx, label in enumerate(label_list):
            if 'VOC' in annotation_root:
                label_id = new_class_names.index(label)
            else:
                label_id = new_class_names_coco.index(label)
            if label_id == que_class[i]:
                # keys.append(label_id)
                targets = [ClipOutputTarget(label_list.index(label))]
                #torch.cuda.empty_cache()
                grayscale_cam, logits_per_image, attn_weight_last = cam(input_tensor=input_tensor,
                                                                                   targets=targets,
                                                                                   target_size=None)

                grayscale_cam = grayscale_cam[0, :]

                # if idx == 0:
                attn_weight_list.append(attn_weight_last)
                attn_weight = [aw[:, 1:, 1:] for aw in attn_weight_list]
                attn_weight = torch.stack(attn_weight, dim=0)[-8:]
                attn_weight = torch.mean(attn_weight, dim=0)
                attn_weight = attn_weight[0].detach()
                attn_weight = attn_weight.float()

                box, cnt = scoremap2bbox(scoremap=grayscale_cam, threshold=0.4, multi_contour_eval=True)
                aff_mask = torch.zeros((grayscale_cam.shape[0],grayscale_cam.shape[1])).cuda()
                for i_ in range(cnt):
                    x0_, y0_, x1_, y1_ = box[i_]
                    aff_mask[y0_:y1_, x0_:x1_] = 1

                aff_mask = aff_mask.view(1,grayscale_cam.shape[0] * grayscale_cam.shape[1])
                aff_mat = attn_weight

                trans_mat = aff_mat / torch.sum(aff_mat, dim=0, keepdim=True)
                trans_mat = trans_mat / torch.sum(trans_mat, dim=1, keepdim=True)


                H_trans_mat = trans_mat @ trans_mat.t()
                trans_mat = torch.max(trans_mat, H_trans_mat)

                for _ in range(1):
                    trans_mat = torch.matmul(trans_mat, trans_mat)

                trans_mat = trans_mat * aff_mask

                cam_to_refine = torch.FloatTensor(grayscale_cam).cuda()
                cam_to_refine = cam_to_refine.view(-1,1)

                # (n,n) * (n,1)->(n,1)
                cam_refined = torch.matmul(trans_mat, cam_to_refine).reshape(h //16, w // 16)
        cam_refined = cam_refined.cpu().numpy().astype(np.float32)
        cam_refined_highres = scale_cam_image([cam_refined], (ori_width, ori_height))[0]
        refined_cam_all_scales.append(torch.tensor(cam_refined_highres).cuda())
    return refined_cam_all_scales

