from torch.nn import functional as F
import torch
import torch.nn as nn
import random

from dassl.engine import TRAINER_REGISTRY, TrainerXU, SimpleNet
from dassl.optim import build_optimizer, build_lr_scheduler
from dassl.data import DataManager
from dassl.data.transforms import build_transform
from dassl.utils import count_num_param
from dassl.modeling import build_head

from .adain.adain import AdaIN


class MLP(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super().__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.bn1 = nn.BatchNorm1d(hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, num_classes)
    
    def forward(self, x):
        x = self.fc1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x

class NormalClassifier(nn.Module):
    def __init__(self, num_features, num_classes):
        super().__init__()
        self.linear = nn.Linear(num_features, num_classes)

    def forward(self, x):
        return self.linear(x)

class Projector(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim=128):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, output_dim)
        )
    
    def forward(self, x):
        x = self.mlp(x)
        return F.normalize(x, p=2, dim=1)


@TRAINER_REGISTRY.register()
class RPCW(TrainerXU):
    def __init__(self, cfg):
        super().__init__(cfg)
        self.conf_thre = cfg.TRAINER.RPCW.CONF_THRE
        self.weight_aug = cfg.TRAINER.RPCW.WEIGHT_AUG
        self.weight_sty = cfg.TRAINER.RPCW.WEIGHT_STYLE
        self.weight_sty_scr = cfg.TRAINER.RPCW.WEIGHT_STYLE_SCR
        self.weight_aux = cfg.TRAINER.RPCW.WEIGHT_AUX

        self.strong_aug = cfg.TRAINER.RPCW.STRONG_AUG
        self.style = cfg.TRAINER.RPCW.STYLE
        self.style_cons = cfg.TRAINER.RPCW.STYLE_CONS
        self.aux_task = cfg.TRAINER.RPCW.AUX_TASK
        self.sigma = cfg.TRAINER.RPCW.SIGMA

        norm_mean = None
        norm_std = None

        if "normalize" in cfg.INPUT.TRANSFORMS:
            norm_mean = cfg.INPUT.PIXEL_MEAN
            norm_std = cfg.INPUT.PIXEL_STD

        self.adain = AdaIN(
            cfg.TRAINER.RPCW.ADAIN_DECODER,
            cfg.TRAINER.RPCW.ADAIN_VGG,
            self.device,
            norm_mean=norm_mean,
            norm_std=norm_std,
        )
    

    def check_cfg(self, cfg):
        assert len(cfg.TRAINER.RPCW.STRONG_TRANSFORMS) > 0
        assert cfg.TRAINER.RPCW.SIGMA > 0
        assert cfg.DATALOADER.TRAIN_X.SAMPLER == "SeqDomainSampler"
        assert cfg.DATALOADER.TRAIN_U.SAME_AS_X

    def build_data_loader(self):
        cfg = self.cfg
        tsf_train = build_transform(cfg, is_train=True)
        custom_tsf_train = [tsf_train]
        choices = cfg.TRAINER.RPCW.STRONG_TRANSFORMS
        tsf_train_strong = build_transform(cfg, is_train=True, choices=choices)
        custom_tsf_train += [tsf_train_strong]
        self.dm = DataManager(self.cfg, custom_tfm_train=custom_tsf_train)
        self.train_loader_x = self.dm.train_loader_x
        self.train_loader_u = self.dm.train_loader_u
        self.val_loader = self.dm.val_loader
        self.test_loader = self.dm.test_loader
        self.num_classes = self.dm.num_classes
        self.num_source_domains = self.dm.num_source_domains
        self.lab2cname = self.dm.lab2cname

    def build_model(self):
        cfg = self.cfg

        print("Building Backbone F:")
        self.F = SimpleNet(cfg, cfg.MODEL, 0)
        self.F.to(self.device)
        print(f"# params: {count_num_param(self.F):,}")
        self.optim_F = build_optimizer(self.F, cfg.OPTIM)
        self.sched_F = build_lr_scheduler(self.optim_F, cfg.OPTIM)
        self.register_model("F", self.F, self.optim_F, self.sched_F)
        
        print("Building C:")
        self.C = MLP(self.F.fdim, self.F.fdim, self.num_classes)
        # self.C = NormalClassifier(self.F.fdim, self.num_classes)
        self.C.to(self.device)
        print(f"# params: {count_num_param(self.C):,}")
        self.optim_C = build_optimizer(self.C, cfg.TRAINER.RPCW.C_OPTIM)
        self.sched_C = build_lr_scheduler(self.optim_C, cfg.TRAINER.RPCW.C_OPTIM)
        self.register_model("C", self.C, self.optim_C, self.sched_C)

        print("Building P:")
        self.P = Projector(self.F.fdim, self.F.fdim)
        self.P.to(self.device)
        print(f"# params: {count_num_param(self.P):,}")
        self.optim_P = build_optimizer(self.P, cfg.TRAINER.RPCW.P_OPTIM)
        self.sched_P = build_lr_scheduler(self.optim_P, cfg.TRAINER.RPCW.P_OPTIM)
        self.register_model("P", self.P, self.optim_P, self.sched_P)

        device_count = torch.cuda.device_count()
        if device_count > 1:
            print(f"Detected {device_count} GPUs (use nn.DataParallel)")
            self.F = nn.DataParallel(self.F)
            self.C = nn.DataParallel(self.C)
            self.P = nn.DataParallel(self.P)

    def assess_y_pred_quality(self, y_pred, y_true, mask, stats):
        n_masked_correct = (y_pred.eq(y_true).float() * mask).sum()
        acc_thre = n_masked_correct / (mask.sum() + 1e-5)  # accuracy after threshold
        acc_raw = y_pred.eq(y_true).sum() / y_pred.numel()  # raw accuracy
        keep_rate = mask.sum() / mask.numel()
        if stats == "y_u_pred_stats":
            output = {"acc_thre": acc_thre, "acc_raw": acc_raw, "keep_rate": keep_rate}
        elif stats == "y_u_styl_pred_stats":
            output = {"acc_thre_styl": acc_thre, "acc_raw_styl": acc_raw, "keep_rate_styl": keep_rate}
        elif stats == "y_u_styl2_pred_stats":
            output = {"acc_thre_styl2": acc_thre, "acc_raw_styl2": acc_raw, "keep_rate_styl2": keep_rate}
        return output

    def forward_backward(self, batch_x, batch_u):
        K = self.num_source_domains
        K = 3 if K == 1 or 2 else K

        batch_return = self.batch_train_transmit(batch_x, batch_u)
        spl_x0 = batch_return["spl_x0"]
        spl_x = batch_return["spl_x"]
        spl_strong_x = batch_return["spl_strong_x"]
        spl_true_x = batch_return["spl_true_x"]
        spl_domain_x = batch_return["spl_domain_x"]

        spl_u0 = batch_return["spl_u0"]
        spl_u = batch_return["spl_u"]
        spl_strong_u = batch_return["spl_strong_u"]
        spl_true_u = batch_return["spl_true_u"]

        # Auxiliary Task
        if self.aux_task:
            spl_aux_x = batch_return["spl_aux_x"]
            spl_aux_u = batch_return["spl_aux_u"]

            ori_x0 = []
            r_true_x = []
            ori_u0 = []
            for k in range(K):
                ori_x0.append(spl_x0[k])
                ori_u0.append(spl_u0[k])
                r_true_x.append(spl_true_x[k])

        # Image Style Enhancement
        if self.style or self.style_cons:
            r_styl = []
            r_styl2 = []
            for k in range(K):
                r_k = torch.cat([spl_x0[k], spl_u0[k]], 0)
                dom_inx = random.sample([i for i in range(K) if i != k], 2)

                r_k2 = torch.cat([spl_x0[dom_inx[0]], spl_u0[dom_inx[0]]], 0)
                r_k3 = torch.cat([spl_x0[dom_inx[1]], spl_u0[dom_inx[1]]], 0)
                r_styl.append(self.adain(r_k, r_k2))
                r_styl2.append(self.adain(r_k, r_k3))
                
        # Pseudo label
        with torch.no_grad():
            po_xu = []
            po_styl = []
            po_styl2 = []

            for i in range(K):
                output_xu = F.softmax(self.C(self.F(torch.cat([spl_x[i], spl_u[i]], 0))), 1)
                po_xu.append(output_xu)

                if self.style_cons:
                    output_styl = F.softmax(self.C(self.F(r_styl[i])), 1)
                    output_styl2 = F.softmax(self.C(self.F(r_styl2[i])), 1)
                    po_styl.append(output_styl)
                    po_styl2.append(output_styl2)

            po_xu = torch.cat(po_xu, 0)
            po_xu_max, po_xu_pred = po_xu.max(1)      # Pseudo label information (weak)
            mask_xu = (po_xu_max >= self.conf_thre).float()

            po_xu_pred = po_xu_pred.chunk(K)
            mask_xu = mask_xu.chunk(K)

            # Calculate pseudo-label's accuracy
            y_u_pred = []
            mask_u = []
            for y_xu_k_pred, mask_xu_k in zip(po_xu_pred, mask_xu):
                y_u_pred.append(y_xu_k_pred.chunk(2)[1])  # only take the 2nd half (unlabeled data)
                mask_u.append(mask_xu_k.chunk(2)[1])
            
            y_u_pred = torch.cat(y_u_pred, 0)
            mask_u = torch.cat(mask_u, 0)
            y_u_pred_stats = self.assess_y_pred_quality(y_u_pred, spl_true_u, mask_u, "y_u_pred_stats")

            # Pseudo label information (style)
            if self.style_cons:
                po_styl = torch.cat(po_styl, 0)
                po_styl_max, po_styl_pred = po_styl.max(1)
                mask_styl = (po_styl_max >= self.conf_thre).float()

                po_styl_pred = po_styl_pred.chunk(K)
                mask_styl = mask_styl.chunk(K)

                # Calculate style_1 pseudo-label's accuracy
                y_u_pred_styl = []
                mask_u_styl = []
                for y_xu_k_pred, mask_xu_k in zip(po_styl_pred, mask_styl):
                    y_u_pred_styl.append(y_xu_k_pred.chunk(2)[1])  # only take the 2nd half (unlabeled data)
                    mask_u_styl.append(mask_xu_k.chunk(2)[1])
            
                y_u_pred_styl = torch.cat(y_u_pred_styl, 0)
                mask_u_styl = torch.cat(mask_u_styl, 0)
                y_u_styl_pred_stats = self.assess_y_pred_quality(y_u_pred_styl, spl_true_u, mask_u_styl, "y_u_styl_pred_stats")

                ###################################
                po_styl2 = torch.cat(po_styl2, 0)
                po_styl2_max, po_styl2_pred = po_styl2.max(1)
                mask_styl2 = (po_styl2_max >= self.conf_thre).float()

                po_styl2_pred = po_styl2_pred.chunk(K)
                mask_styl2 = mask_styl2.chunk(K)

                # Calculate style_2 pseudo-label's accuracy
                y_u_pred_styl2 = []
                mask_u_styl2 = []
                for y_xu_k_pred, mask_xu_k in zip(po_styl2_pred, mask_styl2):
                    y_u_pred_styl2.append(y_xu_k_pred.chunk(2)[1])  # only take the 2nd half (unlabeled data)
                    mask_u_styl2.append(mask_xu_k.chunk(2)[1])
            
                y_u_pred_styl2 = torch.cat(y_u_pred_styl2, 0)
                mask_u_styl2 = torch.cat(mask_u_styl2, 0)
                y_u_styl2_pred_stats = self.assess_y_pred_quality(y_u_pred_styl2, spl_true_u, mask_u_styl2, "y_u_styl2_pred_stats")


        # Supervised loss
        loss_x = 0
        for k in range(K):
            spl_x_k = spl_x[k]
            spl_true_x_k = spl_true_x[k]

            output_x = self.C(self.F(spl_x_k))
            loss_x += F.cross_entropy(output_x, spl_true_x_k)

        # Unsupervised loss
        loss_u_aug = 0
        loss_styl = 0
        loss_styl_cons = 0
        for k in range(K):
            if self.strong_aug:
                r_strong_k = torch.cat([spl_strong_x[k], spl_strong_u[k]], 0)
                strong_  = self.C(self.F(r_strong_k))
                loss_aug_k = F.cross_entropy(strong_, po_xu_pred[k], reduction="none")
                loss_u_aug += (loss_aug_k * mask_xu[k]).mean()
            
            if self.style or self.style_cons:
                styl_1 = self.C(self.F(r_styl[k]))
                styl_2 = self.C(self.F(r_styl2[k]))

                feat_styl_1 = self.F(r_styl[k])
                feat_styl_2 = self.F(r_styl2[k])
                feat_l2 = self.F(torch.cat([spl_x[k], spl_u[k]], 0))

                gamma_j = F.cosine_similarity(feat_styl_1, feat_l2, dim=1)
                gamma_j = torch.exp(gamma_j / self.sigma).mean()
                gamma_v = F.cosine_similarity(feat_styl_2, feat_l2, dim=1)
                gamma_v = torch.exp(gamma_v / self.sigma).mean()

                gamma_sum = gamma_j + gamma_v
                norm_j = gamma_j / gamma_sum
                norm_v = gamma_v / gamma_sum

            if self.style:
                loss_styl_k = F.cross_entropy(styl_1, po_xu_pred[k], reduction="none")
                loss_styl_k = (loss_styl_k * mask_xu[k]).mean()
                loss_styl2_k = F.cross_entropy(styl_2, po_xu_pred[k], reduction="none")
                loss_styl2_k = (loss_styl2_k * mask_xu[k]).mean()

                # dual style loss
                loss_styl += (norm_j * loss_styl_k) + (norm_v * loss_styl2_k)

            if self.style_cons:
                loss_styl_j = F.cross_entropy(styl_2, po_styl_pred[k], reduction="none")
                loss_styl_j = (loss_styl_j * mask_styl[k]).mean()
                loss_styl_v = F.cross_entropy(styl_1, po_styl2_pred[k], reduction="none")
                loss_styl_v = (loss_styl_v * mask_styl2[k]).mean()

                # mulit-view consistency regularity
                loss_styl_cons += (norm_j * loss_styl_j) + (norm_v * loss_styl_v)

        loss_aux_x = 0
        loss_aux_u = 0
        # Auxiliary task loss
        if self.aux_task:
            # x
            ori_x0 = torch.cat(ori_x0, 0)
            aux_feat_x = self.P(self.F(torch.cat([ori_x0, spl_aux_x], dim=0)))  # [2 * batch_size_x, feat_dim]
            r_true_x = torch.cat(r_true_x, 0)
            aux_lab_x = torch.cat([r_true_x, r_true_x], dim=0)  # [2 * batch_size_x]
            aux_domain_x = torch.cat([spl_domain_x, spl_domain_x], dim=0)  # [2 * batch_size_x]

            sim_matrix_x = F.cosine_similarity(aux_feat_x.unsqueeze(1), aux_feat_x.unsqueeze(0), dim=2) / self.sigma

            batch_size_x = len(ori_x0)
            pos_mask_x = torch.zeros_like(sim_matrix_x, dtype=torch.bool)

            for i in range(2 * batch_size_x):
                pos_mask_x[i] = (aux_lab_x == aux_lab_x[i]) & (aux_domain_x == aux_domain_x[i])
                pos_mask_x[i, i] = False
            
            neg_mask_x = (aux_domain_x.unsqueeze(0) != aux_domain_x.unsqueeze(1)) | \
                    ((aux_domain_x.unsqueeze(0) == aux_domain_x.unsqueeze(1)) & 
                        (aux_lab_x.unsqueeze(0) != aux_lab_x.unsqueeze(1)))
            
            exp_sim_x = torch.exp(sim_matrix_x)
            pos_sim_x = torch.sum(exp_sim_x * pos_mask_x.float(), dim=1)
            neg_sim_x = torch.sum(exp_sim_x * neg_mask_x.float(), dim=1)
            loss_aux_x += -torch.log(pos_sim_x / (pos_sim_x + neg_sim_x)).mean()

            # u
            ori_u0 = torch.cat(ori_u0, 0)
            aux_feat_u = self.P(self.F(torch.cat([ori_u0, spl_aux_u], dim=0)))  # [2 * batch_size_u, feat_dim]

            sim_matrix_u = F.cosine_similarity(aux_feat_u.unsqueeze(1), aux_feat_u.unsqueeze(0), dim=2) / self.sigma
            
            batch_size_u = len(ori_u0)
            mask_aux_u = torch.eye(2 * batch_size_u, dtype=torch.bool, device=self.device)
            pos_mask_u = torch.zeros_like(sim_matrix_u, dtype=torch.bool)

            for i in range(batch_size_u):
                pos_mask_u[i, batch_size_u + i] = True
                pos_mask_u[batch_size_u + i, i] = True
            
            pos_mask_u = pos_mask_u & ~mask_aux_u
            neg_mask_u = ~pos_mask_u & ~mask_aux_u

            exp_sim_u = torch.exp(sim_matrix_u)
            pos_sim_u = torch.sum(exp_sim_u * pos_mask_u.float(), dim=1)
            neg_sim_u = torch.sum(exp_sim_u * neg_mask_u.float(), dim=1)
            loss_aux_u += -torch.log(pos_sim_u / (pos_sim_u + neg_sim_u)).mean()


        loss_inform = {}
        loss_overall = 0
        loss_overall += loss_x
        loss_inform["loss_x"] = loss_x.item()

        if self.strong_aug:
            loss_overall += loss_u_aug * self.weight_aug
            loss_inform["loss_u_aug"] = loss_u_aug.item()

        if self.style:
            loss_overall += loss_styl * self.weight_sty
            loss_inform["loss_styl"] = loss_styl.item()
        
        if self.style_cons:
            loss_overall += loss_styl_cons * self.weight_sty_scr
            loss_inform["loss_styl_cons"] = loss_styl_cons.item()

        if self.aux_task:
            loss_overall += loss_aux_u * self.weight_aux + loss_aux_x
            loss_inform["loss_aux_x"] = loss_aux_x.item()
            loss_inform["loss_aux_u"] = loss_aux_u.item()
        
        self.model_backward_and_update(loss_overall)

        loss_inform["y_u_pred_acc_thre"] = y_u_pred_stats["acc_thre"]
        loss_inform["y_u_pred_acc_raw"] = y_u_pred_stats["acc_raw"]
        loss_inform["y_u_pred_keep_rate"] = y_u_pred_stats["keep_rate"]

        if self.style_cons:
            loss_inform["y_u_styl_pred_acc_thre"] = y_u_styl_pred_stats["acc_thre_styl"]
            loss_inform["y_u_styl_pred_acc_raw"] = y_u_styl_pred_stats["acc_raw_styl"]
            loss_inform["y_u_styl_pred_keep_rate"] = y_u_styl_pred_stats["keep_rate_styl"]
    
            loss_inform["y_u_styl2_pred_acc_thre"] = y_u_styl2_pred_stats["acc_thre_styl2"]
            loss_inform["y_u_styl2_pred_acc_raw"] = y_u_styl2_pred_stats["acc_raw_styl2"]
            loss_inform["y_u_styl2_pred_keep_rate"] = y_u_styl2_pred_stats["keep_rate_styl2"]

        if (self.batch_idx + 1) == self.num_batches:
            self.update_lr()

        return loss_inform

    def batch_train_transmit(self, batch_x, batch_u):
        # To generate style transfer images
        K = self.num_source_domains
        K = 3 if K == 1 or 2 else K
        
        spl_x0 = batch_x["img0"]
        spl_x = batch_x["img"]
        spl_strong_x = batch_x["img2"]
        spl_aux_x = batch_x["aux_task"]
        spl_true_x = batch_x["label"]
        spl_domain_x = batch_x["domain"]

        spl_x0 = spl_x0.to(self.device)
        spl_x = spl_x.to(self.device)
        spl_strong_x = spl_strong_x.to(self.device)
        spl_aux_x = spl_aux_x.to(self.device)
        spl_true_x = spl_true_x.to(self.device)
        spl_domain_x = spl_domain_x.to(self.device)

        spl_x0 = spl_x0.chunk(K)
        spl_x = spl_x.chunk(K)
        spl_strong_x = spl_strong_x.chunk(K)
        spl_true_x = spl_true_x.chunk(K)

        spl_u0 = batch_u["img0"]
        spl_u = batch_u["img"]
        spl_strong_u = batch_u["img2"]
        spl_aux_u = batch_u["aux_task"]
        spl_true_u = batch_u["label"]

        spl_u0 = spl_u0.to(self.device)
        spl_u = spl_u.to(self.device)
        spl_strong_u = spl_strong_u.to(self.device)
        spl_aux_u = spl_aux_u.to(self.device)
        spl_true_u = spl_true_u.to(self.device)

        spl_u0 = spl_u0.chunk(K)
        spl_u = spl_u.chunk(K)
        spl_strong_u = spl_strong_u.chunk(K)

        batch = {
            "spl_x0" : spl_x0,
            "spl_x" : spl_x,
            "spl_strong_x" : spl_strong_x,
            "spl_aux_x" : spl_aux_x,
            "spl_true_x" : spl_true_x,
            "spl_domain_x" : spl_domain_x,
            # u
            "spl_u0" : spl_u0,
            "spl_u" : spl_u,
            "spl_strong_u" : spl_strong_u,
            "spl_aux_u" : spl_aux_u,
            "spl_true_u" : spl_true_u
        }
        return batch
    
    def model_inference(self, input):
        return self.C(self.F(input))
