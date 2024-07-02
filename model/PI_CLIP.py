import torch
from torch import nn
import torch.nn.functional as F
from model.Transformer import Transformer
import model.resnet as models
from model.PSPNet import OneModel as PSPNet
from einops import rearrange

# add
import clip
import math
from model.get_cam import get_img_cam
from pytorch_grad_cam import GradCAM
from clip.clip_text import new_class_names, new_class_names_coco

def zeroshot_classifier(classnames, templates, model):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    with torch.no_grad():
        zeroshot_weights = []
        for classname in classnames:
            texts = [template.format(classname) for template in templates] #format with class
            texts = clip.tokenize(texts).to(device) #tokenize
            class_embeddings = model.encode_text(texts) #embed with text encoder
            class_embeddings /= class_embeddings.norm(dim=-1, keepdim=True)
            class_embedding = class_embeddings.mean(dim=0)
            class_embedding /= class_embedding.norm()
            zeroshot_weights.append(class_embedding)
        zeroshot_weights = torch.stack(zeroshot_weights, dim=1).to(device)
    return zeroshot_weights.t()

def Weighted_GAP(supp_feat, mask):
    supp_feat = supp_feat * mask
    feat_h, feat_w = supp_feat.shape[-2:][0], supp_feat.shape[-2:][1]
    area = F.avg_pool2d(mask, (supp_feat.size()[2], supp_feat.size()[3])) * feat_h * feat_w + 0.0005
    supp_feat = F.avg_pool2d(input=supp_feat, kernel_size=supp_feat.shape[-2:]) * feat_h * feat_w / area
    return supp_feat


def get_similarity(q, s, mask):
    if len(mask.shape) == 3:
        mask = mask.unsqueeze(1)
    mask = F.interpolate((mask == 1).float(), q.shape[-2:])
    cosine_eps = 1e-7
    s = s * mask
    bsize, ch_sz, sp_sz, _ = q.size()[:]
    tmp_query = q
    tmp_query = tmp_query.contiguous().view(bsize, ch_sz, -1)
    tmp_query_norm = torch.norm(tmp_query, 2, 1, True)
    tmp_supp = s
    tmp_supp = tmp_supp.contiguous().view(bsize, ch_sz, -1).contiguous()
    tmp_supp = tmp_supp.contiguous().permute(0, 2, 1).contiguous()
    tmp_supp_norm = torch.norm(tmp_supp, 2, 2, True)
    similarity = torch.bmm(tmp_supp, tmp_query) / (torch.bmm(tmp_supp_norm, tmp_query_norm) + cosine_eps)
    similarity = similarity.max(1)[0].view(bsize, sp_sz * sp_sz)
    similarity = similarity.view(bsize, 1, sp_sz, sp_sz)
    return similarity


def get_gram_matrix(fea):
    b, c, h, w = fea.shape
    fea = fea.reshape(b, c, h * w)  # C*N
    fea_T = fea.permute(0, 2, 1)  # N*C
    fea_norm = fea.norm(2, 2, True)
    fea_T_norm = fea_T.norm(2, 1, True)
    gram = torch.bmm(fea, fea_T) / (torch.bmm(fea_norm, fea_T_norm) + 1e-7)  # C*C
    return gram


def get_vgg16_layer(model):
    layer0_idx = range(0, 7)
    layer1_idx = range(7, 14)
    layer2_idx = range(14, 24)
    layer3_idx = range(24, 34)
    layer4_idx = range(34, 43)
    layers_0 = []
    layers_1 = []
    layers_2 = []
    layers_3 = []
    layers_4 = []
    for idx in layer0_idx:
        layers_0 += [model.features[idx]]
    for idx in layer1_idx:
        layers_1 += [model.features[idx]]
    for idx in layer2_idx:
        layers_2 += [model.features[idx]]
    for idx in layer3_idx:
        layers_3 += [model.features[idx]]
    for idx in layer4_idx:
        layers_4 += [model.features[idx]]
    layer0 = nn.Sequential(*layers_0)
    layer1 = nn.Sequential(*layers_1)
    layer2 = nn.Sequential(*layers_2)
    layer3 = nn.Sequential(*layers_3)
    layer4 = nn.Sequential(*layers_4)
    return layer0, layer1, layer2, layer3, layer4


def reshape_transform(tensor, height=28, width=28):
    tensor = tensor.permute(1, 0, 2)
    result = tensor[:, 1:, :].reshape(tensor.size(0), height, width, tensor.size(2))

    # Bring the channels to the first dimension,
    # like in CNNs.
    result = result.transpose(2, 3).transpose(1, 2)
    return result

class OneModel(nn.Module):
    def __init__(self, args, cls_type=None):
        super(OneModel, self).__init__()

        self.cls_type = cls_type  # 'Base' or 'Novel'
        self.dataset = args.data_set
        if self.dataset == 'pascal':
            self.base_classes = 15
        elif self.dataset == 'coco':
            self.base_classes = 60
        self.low_fea_id = args.low_fea[-1]

        assert args.layers in [50, 101, 152]
        from torch.nn import BatchNorm2d as BatchNorm
        self.criterion = nn.CrossEntropyLoss(ignore_index=args.ignore_label)
        self.shot = args.shot
        self.vgg = args.vgg
        models.BatchNorm = BatchNorm

        PSPNet_ = PSPNet(args)
        new_param = torch.load(args.pre_weight, map_location=torch.device('cpu'))['state_dict']
        try:
            PSPNet_.load_state_dict(new_param)
        except RuntimeError:
            for key in list(new_param.keys()):
                new_param[key[7:]] = new_param.pop(key)
            PSPNet_.load_state_dict(new_param)
        self.layer0, self.layer1, self.layer2, self.layer3, self.layer4 = PSPNet_.layer0, PSPNet_.layer1, PSPNet_.layer2, PSPNet_.layer3, PSPNet_.layer4
        self.ppm = PSPNet_.ppm
        self.cls = nn.Sequential(PSPNet_.cls[0], PSPNet_.cls[1])
        self.base_learnear = nn.Sequential(PSPNet_.cls[2], PSPNet_.cls[3], PSPNet_.cls[4])

        if self.vgg:
            fea_dim = 512 + 256
        else:
            fea_dim = 1024 + 512
        #
        self.down_supp = nn.Sequential(
            nn.Conv2d(fea_dim, 256, kernel_size=1, padding=0, bias=False),
            nn.ReLU(inplace=True),
            nn.Dropout2d(p=0.5)
        )
        self.down_query = nn.Sequential(
            nn.Conv2d(fea_dim, 256, kernel_size=1, padding=0, bias=False),
            nn.ReLU(inplace=True),
            nn.Dropout2d(p=0.5)
        )

        if self.shot==1:
            channel = 516
        else:
            channel = 524
        self.query_merge = nn.Sequential(
            nn.Conv2d(channel, 64, kernel_size=1, padding=0, bias=False),
            nn.ReLU(inplace=True),
        )

        self.supp_merge = nn.Sequential(
            nn.Conv2d(512, 64, kernel_size=1, padding=0, bias=False),
            nn.ReLU(inplace=True),
        )

        self.transformer = Transformer(shot=self.shot)

        self.gram_merge = nn.Conv2d(2, 1, kernel_size=1, bias=False)
        self.gram_merge.weight = nn.Parameter(torch.tensor([[1.0], [0.0]]).reshape_as(self.gram_merge.weight))

        # Learner Ensemble
        self.cls_merge = nn.Conv2d(2, 1, kernel_size=1, bias=False)
        self.cls_merge.weight = nn.Parameter(torch.tensor([[1.0], [0.0]]).reshape_as(self.cls_merge.weight))

        # K-Shot Reweighting
        if args.shot > 1:
            self.kshot_trans_dim = args.kshot_trans_dim
            if self.kshot_trans_dim == 0:
                self.kshot_rw = nn.Conv2d(self.shot, self.shot, kernel_size=1, bias=False)
                self.kshot_rw.weight = nn.Parameter(torch.ones_like(self.kshot_rw.weight) / args.shot)
            else:
                self.kshot_rw = nn.Sequential(
                    nn.Conv2d(self.shot, self.kshot_trans_dim, kernel_size=1),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(self.kshot_trans_dim, self.shot, kernel_size=1))

        # add
        self.annotation_root = args.annotation_root
        self.clip_model, _ = clip.load(args.clip_path)
        if self.dataset == 'pascal':
            self.bg_text_features = zeroshot_classifier(new_class_names, ['a photo without {}.'],
                                                        self.clip_model)
            self.fg_text_features = zeroshot_classifier(new_class_names, ['a photo of {}.'],
                                                        self.clip_model)
        elif self.dataset == 'coco':
            self.bg_text_features = zeroshot_classifier(new_class_names_coco, ['a photo without {}.'],
                                                        self.clip_model)
            self.fg_text_features = zeroshot_classifier(new_class_names_coco, ['a photo of {}.'],
                                                        self.clip_model)

    def forward(self, x, x_cv2, que_name, class_name, y_m=None, y_b=None, s_x=None, s_y=None, cat_idx=None):
        mask = rearrange(s_y, "b n h w -> (b n) 1 h w")
        mask = (mask == 1).float()
        h, w = x.shape[-2:]
        s_x = rearrange(s_x, "b n c h w -> (b n) c h w")

        # extract the cnn features
        _, _, query_feat_2, query_feat_3, query_feat_4, query_feat_5 = self.extract_feats(x)
        supp_feat_0, supp_feat_1, supp_feat_2, supp_feat_3, supp_feat_4, supp_feat_5 = self.extract_feats(s_x, mask)

        supp_feat_cnn = torch.cat([supp_feat_3, supp_feat_2], 1)
        supp_feat_cnn = self.down_supp(supp_feat_cnn)
        query_feat_cnn = torch.cat([query_feat_3, query_feat_2], 1)
        query_feat_cnn = self.down_query(query_feat_cnn)

        supp_feat_item = eval('supp_feat_' + self.low_fea_id)
        supp_feat_item = rearrange(supp_feat_item, "(b n) c h w -> b n c h w", n=self.shot)
        supp_feat_list_ori = [supp_feat_item[:, i, ...] for i in range(self.shot)]

        # extract the clip features
        if mask is not None:
            tmp_mask = F.interpolate(mask, size=x.shape[-2], mode='nearest')
            s_x_mask = s_x * tmp_mask
        tmp_supp_clip_fts, supp_attn_maps = self.clip_model.encode_image(s_x_mask, h, w, extract=True)[:]
        tmp_que_clip_fts, que_attn_maps = self.clip_model.encode_image(x, h, w, extract=True)[:]

        supp_clip_fts = [ss[1:, :, :] for ss in tmp_supp_clip_fts]
        que_clip_fts = [ss[1:, :, :] for ss in tmp_que_clip_fts]

        tmp_supp_clip_feat_all = [ss.permute(1, 2, 0) for ss in supp_clip_fts]
        supp_clip_feat_all = [aw.reshape(
            tmp_supp_clip_feat_all[0].shape[0], tmp_supp_clip_feat_all[0].shape[1], int(math.sqrt(tmp_supp_clip_feat_all[0].shape[2])),
            int(math.sqrt(tmp_supp_clip_feat_all[0].shape[2]))).float()
            for aw in tmp_supp_clip_feat_all]

        tmp_que_clip_feat_all = [qq.permute(1, 2, 0) for qq in que_clip_fts]
        que_clip_feat_all = [aw.reshape(
            tmp_que_clip_feat_all[0].shape[0], tmp_que_clip_feat_all[0].shape[1], int(math.sqrt(tmp_que_clip_feat_all[0].shape[2])),
            int(math.sqrt(tmp_que_clip_feat_all[0].shape[2]))).float()
            for aw in tmp_que_clip_feat_all]

        #get the vvp
        if self.shot == 1:
            similarity2 = get_similarity(que_clip_feat_all[10], supp_clip_feat_all[10], s_y)
            similarity1 = get_similarity(que_clip_feat_all[11], supp_clip_feat_all[11], s_y)
        else:
            mask = rearrange(mask, "(b n) c h w -> b n c h w", n=self.shot)
            supp_clip_feat_all = [rearrange(ss, "(b n) c h w -> b n c h w", n=self.shot) for ss in supp_clip_feat_all]
            clip_similarity_1 = [get_similarity(que_clip_feat_all[11], supp_clip_feat_all[11][:, i, ...], mask=mask[:, i, ...]) for i in
                           range(self.shot)]
            clip_similarity_2 = [get_similarity(que_clip_feat_all[10], supp_clip_feat_all[10][:, i, ...], mask=mask[:, i, ...]) for i in
                           range(self.shot)]
            mask = rearrange(mask, "b n c h w -> (b n) c h w")
            similarity1 = torch.cat(clip_similarity_1, dim=1)
            similarity2 = torch.cat(clip_similarity_2, dim=1)
        clip_similarity = torch.cat([similarity1, similarity2], dim=1).cuda()
        clip_similarity = F.interpolate(clip_similarity, size=(supp_feat_cnn.shape[2], supp_feat_cnn.shape[3]), mode='bilinear', align_corners=True)

        # get the vtp
        target_layers = [self.clip_model.visual.transformer.resblocks[-1].ln_1]
        cam = GradCAM(model=self.clip_model, target_layers=target_layers, reshape_transform=reshape_transform)
        img_cam_list = get_img_cam(x_cv2, que_name, class_name, self.clip_model, self.bg_text_features, self.fg_text_features, cam, self.annotation_root, self.training)
        img_cam_list = [F.interpolate(t_img_cam.unsqueeze(0).unsqueeze(0), size=(supp_feat_cnn.shape[2], supp_feat_cnn.shape[3]), mode='bilinear',
                                      align_corners=True) for t_img_cam in img_cam_list]
        img_cam = torch.cat(img_cam_list, 0)
        img_cam = img_cam.repeat(1,2,1,1)

        supp_pro = Weighted_GAP(supp_feat_cnn, \
                                F.interpolate(mask, size=(supp_feat_cnn.size(2), supp_feat_cnn.size(3)),
                                              mode='bilinear', align_corners=True))
        supp_feat_bin = supp_pro.repeat(1, 1, supp_feat_cnn.shape[-2], supp_feat_cnn.shape[-1])

        supp_feat = self.supp_merge(torch.cat([supp_feat_cnn, supp_feat_bin],
                                              dim=1))

        # K-Shot Reweighting
        bs = x.shape[0]
        que_gram = get_gram_matrix(eval('query_feat_' + self.low_fea_id))
        norm_max = torch.ones_like(que_gram).norm(dim=(1, 2))
        est_val_list = []
        for supp_item in supp_feat_list_ori:
            supp_gram = get_gram_matrix(supp_item)
            gram_diff = que_gram - supp_gram
            est_val_list.append((gram_diff.norm(dim=(1, 2)) / norm_max).reshape(bs, 1, 1, 1))
        est_val_total = torch.cat(est_val_list, 1)
        if self.shot > 1:
            val1, idx1 = est_val_total.sort(1)
            val2, idx2 = idx1.sort(1)
            weight = self.kshot_rw(val1)
            idx3 = idx1.gather(1, idx2)
            weight = weight.gather(1, idx3)
            weight_soft = torch.softmax(weight, 1)
        else:
            weight_soft = torch.ones_like(est_val_total)
        est_val = (weight_soft * est_val_total).sum(1, True)  # [bs, 1, 1, 1]

        supp_feat_bin = rearrange(supp_feat_bin, "(b n) c h w -> b n c h w", n=self.shot)
        supp_feat_bin = torch.mean(supp_feat_bin, dim=1)
        query_feat = self.query_merge(torch.cat([query_feat_cnn, supp_feat_bin, img_cam * 10, clip_similarity * 10], dim=1))

        meta_out, weights = self.transformer(query_feat, supp_feat, mask, img_cam, clip_similarity)
        base_out = self.base_learnear(query_feat_5)

        meta_out_soft = meta_out.softmax(1)
        base_out_soft = base_out.softmax(1)

        # Following the implementation of BAM ( https://github.com/chunbolang/BAM )
        meta_map_bg = meta_out_soft[:, 0:1, :, :]
        meta_map_fg = meta_out_soft[:, 1:, :, :]
        if self.training and self.cls_type == 'Base':
            c_id_array = torch.arange(self.base_classes + 1).cuda()
            base_map_list = []
            for b_id in range(bs):
                c_id = cat_idx[0][b_id] + 1
                c_mask = (c_id_array != 0) & (c_id_array != c_id)
                base_map_list.append(base_out_soft[b_id, c_mask, :, :].unsqueeze(0).sum(1, True))
            base_map = torch.cat(base_map_list, 0)
        else:
            base_map = base_out_soft[:, 1:, :, :].sum(1, True)

        map_h, map_w = meta_map_bg.shape[-2], meta_map_bg.shape[-1]
        base_map = F.interpolate(base_map, size=(map_h, map_w), mode='bilinear', align_corners=True)

        est_map = est_val.expand_as(meta_map_fg)

        meta_map_bg = self.gram_merge(torch.cat([meta_map_bg, est_map], dim=1))
        meta_map_fg = self.gram_merge(torch.cat([meta_map_fg, est_map], dim=1))

        merge_map = torch.cat([meta_map_bg, base_map], 1)
        merge_bg = self.cls_merge(merge_map)  # [bs, 1, 60, 60]

        final_out = torch.cat([merge_bg, meta_map_fg], dim=1)

        # Output Part
        meta_out = F.interpolate(meta_out, size=(h, w), mode='bilinear', align_corners=True)
        base_out = F.interpolate(base_out, size=(h, w), mode='bilinear', align_corners=True)
        final_out = F.interpolate(final_out, size=(h, w), mode='bilinear', align_corners=True)

        # Loss
        if self.training:
            main_loss = self.criterion(final_out, y_m.long())
            aux_loss1 = self.criterion(meta_out, y_m.long())
            aux_loss2 = self.criterion(base_out, y_b.long())

            weight_t = (y_m == 1).float()
            weight_t = torch.masked_fill(weight_t, weight_t == 0, -1e9)
            for i, weight in enumerate(weights):
                if i == 0:
                    distil_loss = self.disstil_loss(weight_t, weight)
                else:
                    distil_loss += self.disstil_loss(weight_t, weight)
                weight_t = weight.detach()

            return final_out.max(1)[1], main_loss + aux_loss1, distil_loss / 3, aux_loss2
        else:
            return final_out, meta_out, base_out

    def disstil_loss(self, t, s):
        if t.shape[-2:] != s.shape[-2:]:
            t = F.interpolate(t.unsqueeze(1), size=s.shape[-2:], mode='bilinear').squeeze(1)
        t = rearrange(t, "b h w -> b (h w)")
        s = rearrange(s, "b h w -> b (h w)")
        s = torch.softmax(s, dim=1)
        t = torch.softmax(t, dim=1)
        loss = t * torch.log(t + 1e-12) - t * torch.log(s + 1e-12)
        loss = loss.sum(1).mean()
        return loss

    def get_optim(self, model, args, LR):
        optimizer = torch.optim.AdamW(
            [
                {'params': model.transformer.mix_transformer.parameters()},
                {'params': model.supp_merge.parameters(), "lr": LR * 10},
                {'params': model.query_merge.parameters(), "lr": LR * 10},
                {'params': model.cls_merge.parameters(), "lr": LR * 10},
                {'params': model.down_supp.parameters(), "lr": LR * 10},
                {'params': model.down_query.parameters(), "lr": LR * 10},
                {'params': model.gram_merge.parameters(), "lr": LR * 10},
            ], lr=LR, weight_decay=args.weight_decay, betas=(0.9, 0.999))
        return optimizer

    def freeze_modules(self, model):
        for param in model.layer0.parameters():
            param.requires_grad = False
        for param in model.layer1.parameters():
            param.requires_grad = False
        for param in model.layer2.parameters():
            param.requires_grad = False
        for param in model.layer3.parameters():
            param.requires_grad = False
        for param in model.layer4.parameters():
            param.requires_grad = False
        for param in model.ppm.parameters():
            param.requires_grad = False
        for param in model.cls.parameters():
            param.requires_grad = False
        for param in model.base_learnear.parameters():
            param.requires_grad = False

    def extract_feats(self, x, mask=None):
        results = []
        with torch.no_grad():
            if mask is not None:
                tmp_mask = F.interpolate(mask, size=x.shape[-2], mode='nearest')
                x = x * tmp_mask
            feat = self.layer0(x)
            results.append(feat)
            layers = [self.layer1, self.layer2, self.layer3, self.layer4]
            for _, layer in enumerate(layers):
                feat = layer(feat)
                results.append(feat.clone())
            feat = self.ppm(feat)
            feat = self.cls(feat)
            results.append(feat)
        return results
