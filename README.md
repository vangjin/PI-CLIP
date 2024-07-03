# Rethinking Prior Information Generation with CLIP for Few-Shot Segmentation

This is the official implementation for our **CVPR 2024** [paper](https://arxiv.org/abs/2405.08458) "*Rethinking Prior Information Generation with CLIP for Few-Shot Segmentation*".

Please note that the experimental results may vary due to different environments and settings. In all experiments on
PASCAL-5{i} and COCO-20{i}, the images are set to 473×473. For COCO-20{i}, setting higher resolution can get higher performance, besides, PI-CLIP is only trained for 30 epochs
on both PASCAL-5{i} and COCO-20{i} the model can perform better if can be trained for more epochs. However, it is still acceptable to compare your results with those reported in the paper.



> **Abstract:** *Few-shot segmentation remains challenging due to the limitations of its labeling information for unseen classes. Most previous approaches rely on extracting high-level feature maps from the frozen visual encoder to compute the pixel-wise similarity as a key prior guidance for the decoder. However, such a prior representation suffers from coarse granularity and poor generalization to new classes since these high-level feature maps have obvious category bias. In this work, we propose to replace the visual prior representation with the visual-text alignment capacity to capture more reliable guidance and enhance the model generalization. Specifically, we design two kinds of training-free prior information generation strategy that attempts to utilize the semantic alignment capability of the Contrastive Language-Image Pre-training model (CLIP) to locate the target class. Besides, to acquire more accurate prior guidance, we build a high-order relationship of attention maps and utilize it to refine the initial prior information. Experiments on both the PASCAL-5{i} and COCO-20{i} datasets show that our method obtains a clearly substantial improvement and reaches the new state-of-the-art performance.*


## Get Started

### Environment

- python == 3.10.4
- torch == 1.12.1
- torchvision == 0.13.1
- cuda == 11.6
- mmcv-full == 1.7.1
- mmsegmentation == 0.30.0

```
cd PI-CLIP
git clone https://github.com/lucasb-eyer/pydensecrf
cd pydensecrf
python setup.py install

#install other packages
cd PI_CLIP
python env.py
```



### Dataset
Please download the following datasets and put them into the `../data` directory.:

+ PASCAL-5<sup>i</sup>: [**PASCAL VOC 2012**](http://host.robots.ox.ac.uk/pascal/VOC/voc2012/) and [**SBD**](http://home.bharathh.info/pubs/codes/SBD/download.html)

+ COCO-20<sup>i</sup>: [**COCO 2014**](https://cocodataset.org/#download).

The lists generation are followed [PFENet](https://github.com/dvlab-research/PFENet). You can direct download and put them into the `./lists` directory.

 Before running the code, you should generate the annotations for base classes by running `util/get_mulway_base_data.py`, more details are available at [BAM](https://github.com/chunbolang/BAM).
## Models

We have adopted the same procedures as [BAM](https://github.com/chunbolang/BAM) and [HDMNet](https://github.com/Pbihao/HDMNet) for the pre-trained backbones, placing them in the `../initmodel` directory. 

Download CLIP pre-trained ViT-B/16 at [**here**](https://openaipublic.azureedge.net/clip/models/5806e77cd80f8b59890b7e101eabd078d9fb84e6937f9e85e4ecb61988df416f/ViT-B-16.pt) and put it to `../initmodel/clip`


## Scripts
- First update the configurations in the `./config` for training or testing

- Train script
```
sh train.sh [exp_name] [dataset] [GPUs]

# Example (split0 | PASCAL VOC2012 | 2 GPUs for traing):
# sh train.sh split0 pascal 2
```
- Test script
```
sh test.sh [exp_name] [dataset] [GPUs]

# Example (split0 | COCO dataset | 1 GPU for testing):
# sh test.sh split0 coco 1
```

## References

This repository owes its existence to the exceptional contributions of other projects:

* PFENet: https://github.com/dvlab-research/PFENet
* BAM: https://github.com/chunbolang/BAM
* HDMNet: https://github.com/Pbihao/HDMNet

Many thanks for their excellent work.

## Question
If you have any question, welcome email me at 'wangjin@s.upc.deu.cn'


## BibTeX

If you find our work and this repository useful. Please consider giving a star and citation.

```bibtex
@inproceedings{wang2024rethinking,
  title={Rethinking Prior Information Generation with CLIP for Few-Shot Segmentation},
  author={Wang, Jin and Zhang, Bingfeng and Pang, Jian and Chen, Honglong and Liu, Weifeng},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
  pages={3941--3951},
  year={2024}
}
```



