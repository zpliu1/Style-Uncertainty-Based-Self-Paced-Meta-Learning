from __future__ import print_function, absolute_import, division
import time
import numpy as np
import collections

import torch
import torch.nn as nn
import math
import torch.nn.functional as F
from .utils.meters import AverageMeter
from .models import *
from .evaluation_metrics import accuracy
from .models.MetaModules import MixUpBatchNorm1d as MixUp1D

from random import shuffle
import copy

class Trainer(object):
    def __init__(self, args, model, memory, memories_new, memories_new2, criterion):
        super(Trainer, self).__init__()
        self.model = model
        self.memory = memory
        self.memories_new = memories_new
        self.memories_new2 = memories_new2
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.criterion = criterion
        self.args = args




    def getgrad(self, net):
        g = []
        for param in net.module.params():
        #for param in net.params():
            if param.requires_grad:

                if param.grad is not None:
                    #g.append(torch.tensor(param.grad).view(-1).contiguous())
                    g.append(param.grad.clone().detach().view(-1).contiguous())

        g = torch.cat(g, dim=0).detach()
        return g



    def compute_agr_mask(self, grad_0, grad_1, grad_2):

        grad_sign = torch.stack([torch.sign(grad_0), torch.sign(grad_1), torch.sign(grad_2)])
        agr_mask = torch.where(grad_sign.sum(0).abs() == 3, 1, 0)
        return agr_mask.bool()



    def set_grads(self, network, new_grads):
        start = 0
        for k, p in enumerate(network.module.params()):

            dims = p.shape
            end = start + dims.numel()
            p.grad.data = new_grads[start:end].reshape(dims)
            start = end

    def pcgrad(self, grad_0, grad_1, grad_2):
        """ Projecting conflicting gradients (PCGrad). """

        domain_grads = [grad_0, grad_1, grad_2]

        task_order = list(range(len(domain_grads)))

        # Run tasks in random order
        shuffle(task_order)
        # Initialize task gradients
        grad_pc = [g.clone() for g in domain_grads]
        for i in task_order:
            # Run other tasks
            other_tasks = [j for j in task_order if j != i]
            for j in other_tasks:
                grad_j = domain_grads[j]
                # Compute inner product and check for conflicting gradients
                inner_prod = torch.dot(grad_pc[i], grad_j)
                if inner_prod < 0:
                    # Sustract the conflicting component
                    print ('**********')
                    grad_pc[i] -= inner_prod / (grad_j ** 2).sum() * grad_j
        # Sum task gradients
        new_grads = torch.stack(grad_pc).sum(0)
        return new_grads

    def lunif(self, x, t=2):
        sq_pdist = torch.pdist(x, p=2).pow(2)
        return sq_pdist.mul(-t).exp().mean().log()


    def train(self, epoch, data_loaders, optimizer, print_freq=10, train_iters=400):
        self.model.train()

        batch_time = AverageMeter()
        data_time = AverageMeter()
        losses = AverageMeter()
        losses_meta_train = AverageMeter()
        losses_meta_test = AverageMeter()
        metaLR = optimizer.param_groups[0]['lr'] 

        source_count = len(data_loaders)


        optimizer_mte = torch.optim.SGD(self.model.module.params(), lr=metaLR) #3.5e-06
        optimizer_mte2 = torch.optim.SGD(self.model.module.params(), lr=metaLR)  # 3.5e-06

        end = time.time()
        for i in range(train_iters):
            metaTestID = np.random.choice(source_count)
            network_bns = [x for x in list(self.model.modules()) if isinstance(x, MixUp1D)]

            for bn in network_bns:
                bn.meta_mean1 = torch.zeros(bn.meta_mean1.size()).float().cuda()
                bn.meta_var1 = torch.zeros(bn.meta_var1.size()).float().cuda()
                bn.meta_mean2 = torch.zeros(bn.meta_mean2.size()).float().cuda()
                bn.meta_var2 = torch.zeros(bn.meta_var2.size()).float().cuda()

            # with torch.autograd.set_detect_anomaly(True):
            if True:
                data_loader_index = [i for i in range(source_count)] ## 0 2
                #del data_loader_index[metaTestID]
                batch_data = [data_loaders[i].next() for i in range(source_count)]
                metaTestinputs = batch_data[metaTestID]
                data_time.update(time.time() - end)
                # process inputs
                testInputs, testPids, _, _, _, _ = self._parse_data(metaTestinputs)
                loss_meta_train = 0.
                loss_meta_train_ali = 0.
                loss_meta_train1 = 0.
                loss_meta_train2 = 0.
                loss_meta2_test = 0.
                loss_meta_test = 0.

                data_time.update(time.time() - end)
                traininputs_0 = batch_data[0]
                traininputs_1 = batch_data[1]
                traininputs_2 = batch_data[2]
                inputs_0, targets_0, _, _, _, _ = self._parse_data(traininputs_0)
                inputs_1, targets_1, _, _, _, _ = self._parse_data(traininputs_1)
                inputs_2, targets_2, _, _, _, _ = self._parse_data(traininputs_2)

                ## data 0 ##
                f_out_0, tri_features_0 = self.model(inputs_0, MTE='', save_index=1, index=0)
                loss_mtr_tri_0, dist_mat_0 = self.criterion(tri_features_0, targets_0)
                loss_s_0 = self.memory[0](f_out_0, targets_0).mean()




                ## data 1 ##
                f_out_2, tri_features_2 = self.model(inputs_1, MTE='', save_index=2, index=0)
                loss_mtr_tri_2, dist_mat_2 = self.criterion(tri_features_2, targets_1)
                loss_s_2 = self.memory[1](f_out_2, targets_1).mean()





                ## data 2 ##
                f_out_4, tri_features_4 = self.model(inputs_2, MTE='', save_index=3, index=0)
                loss_mtr_tri_4, dist_mat_4 = self.criterion(tri_features_4, targets_2)
                loss_s_4 = self.memory[2](f_out_4, targets_2).mean()






                #loss_meta_train = loss_meta_train + loss_mtr_tri_0 + loss_s_0 + loss_mtr_tri_1 + loss_s_1 + loss_mtr_tri_2 + loss_s_2 + loss_mtr_tri_3 + loss_s_3 + loss_mtr_tri_4 + loss_s_4 + loss_mtr_tri_5 + loss_s_5 + loss_com_0 + loss_com_1 + loss_com_2
                #loss_meta_train = loss_meta_train + loss_mtr_tri_0 + loss_s_0 + loss_mtr_tri_1 + loss_s_1 + loss_mtr_tri_2 + loss_s_2 + loss_mtr_tri_3 + loss_s_3 + loss_mtr_tri_4 + loss_s_4 + loss_mtr_tri_5 + loss_s_5
                loss_meta_train1 = loss_meta_train1 + loss_mtr_tri_0 + loss_s_0 + loss_mtr_tri_2 + loss_s_2 + loss_mtr_tri_4 + loss_s_4
                #loss_meta_train2 = loss_meta_train2 + loss_mtr_tri_1 + loss_s_1 + loss_mtr_tri_3 + loss_s_3 + loss_mtr_tri_5 + loss_s_5

                #loss_meta_train = loss_meta_train / 3
                #loss_final = loss_meta_train / 3
                #loss_meta_test = loss_meta_train
                loss_meta_train1 = loss_meta_train1 / 3
                #loss_meta_train2 = loss_meta_train2 / 3

                loss_meta_train1 = loss_meta_train1 / 3

                loss_meta_train = loss_meta_train1



                model_dice_0 = {}
                model_dice_0 = copy.deepcopy(self.model.state_dict())



                optimizer.zero_grad()
                optimizer_mte2.zero_grad()
                optimizer_mte.zero_grad()

                loss_meta_train1.backward()
                grad_0 = self.getgrad(self.model)
                optimizer_mte.step()

                optimizer.zero_grad()
                optimizer_mte2.zero_grad()
                optimizer_mte.zero_grad()




                f_test_0, mte_tri_0 = self.model(inputs_0, MTE='', save_index=1, index=0)
                loss_meta_test_0 = self.memory[0](f_test_0, targets_0).mean()
                loss_mte_tri_0, dist_mte_0 = self.criterion(mte_tri_0, targets_0)

                f_test_1, mte_tri_1 = self.model(inputs_0, MTE='', save_index=1, index=1)
                loss_meta_test_1 = self.memories_new[0](f_test_1, targets_0).mean()
                loss_mte_tri_1, dist_mte_1 = self.criterion(mte_tri_1, targets_0)

                loss_mte_com_0 = F.pairwise_distance(dist_mte_0, dist_mte_1, p=2).mean()
                #loss_mte_com_0 = loss_mte_com_0 * 2



                ## data 2 ##
                f_test_2, mte_tri_2 = self.model(inputs_1, MTE='', save_index=2, index=0)
                loss_meta_test_2 = self.memory[1](f_test_2, targets_1).mean()
                loss_mte_tri_2, dist_mte_2 = self.criterion(mte_tri_2, targets_1)

                f_test_3, mte_tri_3 = self.model(inputs_1, MTE='', save_index=2, index=1)
                loss_meta_test_3 = self.memories_new[1](f_test_3, targets_1).mean()
                loss_mte_tri_3, dist_mte_3 = self.criterion(mte_tri_3, targets_1)

                loss_mte_com_1 = F.pairwise_distance(dist_mte_2, dist_mte_3, p=2).mean()
                #loss_mte_com_1 = loss_mte_com_1 * 2


                ## data 3 ##
                f_test_4, mte_tri_4 = self.model(inputs_2, MTE='', save_index=3, index=0)
                loss_meta_test_4 = self.memory[2](f_test_4, targets_2).mean()
                loss_mte_tri_4, dist_mte_4 = self.criterion(mte_tri_4, targets_2)

                f_test_5, mte_tri_5 = self.model(inputs_2, MTE='', save_index=3, index=1)
                loss_meta_test_5 = self.memories_new[2](f_test_5, targets_2).mean()
                loss_mte_tri_5, dist_mte_5 = self.criterion(mte_tri_5, targets_2)

                loss_mte_com_2 = F.pairwise_distance(dist_mte_4, dist_mte_5, p=2).mean()
                #loss_mte_com_2 = loss_mte_com_2 * 2


                #loss_meta_test = loss_meta_test_1 + loss_mte_tri_1 + loss_meta_test_3 + loss_mte_tri_3 + loss_meta_test_5 + loss_mte_tri_5 + loss_mte_com_0 + loss_mte_com_1 + loss_mte_com_2 + loss_meta_test_0 + loss_mte_tri_0 + loss_meta_test_2 + loss_mte_tri_2 + loss_meta_test_4 + loss_mte_tri_4
                loss_meta_test = loss_meta_test + loss_meta_test_1 + loss_mte_tri_1 + loss_meta_test_3 + loss_mte_tri_3 + loss_meta_test_5 + loss_mte_tri_5 + loss_mte_com_0 + loss_mte_com_1 + loss_mte_com_2+ loss_meta_test_0 + loss_mte_tri_0 + loss_meta_test_2 + loss_mte_tri_2 + loss_meta_test_4 + loss_mte_tri_4

                loss_meta_test = loss_meta_test/3

                loss_meta_test = loss_meta_test/3

                #loss_meta_test = loss_meta_test * (24/27)


                #grad_1 = 1
                optimizer.zero_grad()
                optimizer_mte2.zero_grad()
                optimizer_mte.zero_grad()

                loss_meta_test.backward()
                grad_1 = self.getgrad(self.model)

                optimizer.zero_grad()
                optimizer_mte2.zero_grad()
                optimizer_mte.zero_grad()

                self.model.load_state_dict(model_dice_0)


                #losses.update(loss_final.item())

                optimizer.zero_grad()
                optimizer_mte2.zero_grad()
                optimizer_mte.zero_grad()
                # grad_new = grad_0 + grad_1
                grad_new = grad_0 + grad_1
                self.set_grads(self.model, grad_new)
                # loss_final.backward()
                optimizer_mte2.step()

                optimizer.zero_grad()
                optimizer_mte2.zero_grad()
                optimizer_mte.zero_grad()


                ##
                f2_test_0, mte2_tri_0 = self.model(inputs_0, MTE='', save_index=1, index=0)
                loss_meta2_test_0 = self.memory[0](f2_test_0, targets_0).mean()
                loss_mte2_tri_0, dist_mte2_0 = self.criterion(mte2_tri_0, targets_0)

                f2_test_1, mte2_tri_1 = self.model(inputs_0, MTE='', save_index=1, index=2)
                loss_meta2_test_1 = self.memories_new2[0](f2_test_1, targets_0).mean()
                loss_mte2_tri_1, dist_mte2_1 = self.criterion(mte2_tri_1, targets_0)

                loss_mte2_com_0 = F.pairwise_distance(dist_mte2_0, dist_mte2_1, p=2).mean()
                #loss_mte2_com_0 = loss_mte2_com_0 * 2




                f2_test_2, mte2_tri_2 = self.model(inputs_1, MTE='', save_index=2, index=0)
                loss_meta2_test_2 = self.memory[1](f2_test_2, targets_1).mean()
                loss_mte2_tri_2, dist_mte2_2 = self.criterion(mte2_tri_2, targets_1)

                f2_test_3, mte2_tri_3 = self.model(inputs_1, MTE='', save_index=2, index=2)
                loss_meta2_test_3 = self.memories_new2[1](f2_test_3, targets_1).mean()
                loss_mte2_tri_3, dist_mte2_3 = self.criterion(mte2_tri_3, targets_1)

                loss_mte2_com_1 = F.pairwise_distance(dist_mte2_2, dist_mte2_3, p=2).mean()
                #loss_mte2_com_1 = loss_mte2_com_1 * 2




                f2_test_4, mte2_tri_4 = self.model(inputs_2, MTE='', save_index=3, index=0)
                loss_meta2_test_4 = self.memory[2](f2_test_4, targets_2).mean()
                loss_mte2_tri_4, dist_mte2_4 = self.criterion(mte2_tri_4, targets_2)

                f2_test_5, mte2_tri_5 = self.model(inputs_2, MTE='', save_index=3, index=2)
                loss_meta2_test_5 = self.memories_new2[2](f2_test_5, targets_2).mean()
                loss_mte2_tri_5, dist_mte2_5 = self.criterion(mte2_tri_5, targets_2)

                loss_mte2_com_2 = F.pairwise_distance(dist_mte2_4, dist_mte2_5, p=2).mean()
                #loss_mte2_com_2 = loss_mte2_com_2 * 2




                loss_meta2_test = loss_meta2_test + loss_meta2_test_1 + loss_mte2_tri_1 + loss_meta2_test_3 + loss_mte2_tri_3 + loss_meta2_test_5 + loss_mte2_tri_5 + loss_mte2_com_0 + loss_mte2_com_1 + loss_mte2_com_2+ loss_meta2_test_0 + loss_mte2_tri_0 + loss_meta2_test_2 + loss_mte2_tri_2 + loss_meta2_test_4 + loss_mte2_tri_4

                loss_meta2_test = loss_meta2_test / 3

                loss_meta2_test = loss_meta2_test / 3


                optimizer.zero_grad()
                optimizer_mte2.zero_grad()
                optimizer_mte.zero_grad()

                loss_meta2_test.backward()
                grad_2 = self.getgrad(self.model)

                optimizer.zero_grad()
                optimizer_mte2.zero_grad()
                optimizer_mte.zero_grad()

                self.model.load_state_dict(model_dice_0)


                optimizer.zero_grad()
                optimizer_mte2.zero_grad()
                optimizer_mte.zero_grad()

                #grad_new = grad_0 + grad_1
                grad_new2 = grad_0 + grad_1 + grad_2
                self.set_grads(self.model, grad_new2)
                #loss_final.backward()
                optimizer.step()

                optimizer.zero_grad()
                optimizer_mte2.zero_grad()
                optimizer_mte.zero_grad()


                losses_meta_train.update(loss_meta_train.item())
                losses_meta_test.update(loss_meta_test.item())

                #optimizer.zero_grad()
                #loss_final.backward()
                #optimizer.step()
                loss_final = losses_meta_test


                with torch.no_grad():
                    for m_ind in range(source_count):
                        _, pids, _, _, _, imgss = self._parse_data(batch_data[m_ind])
                        f_new, _ = self.model(imgss, index=0)
                        self.memory[m_ind].module.MomentumUpdate(f_new, pids)

                        f_new_1, _ = self.model(imgss, index=1)
                        self.memories_new[m_ind].module.MomentumUpdate(f_new_1, pids)

                        f_new_2, _ = self.model(imgss, index=2)
                        self.memories_new2[m_ind].module.MomentumUpdate(f_new_2, pids)


                #losses.update(loss_final.item())

            # print log
            batch_time.update(time.time() - end)
            end = time.time()


            if (i + 1) % print_freq == 0:
                print('Epoch: [{}][{}/{}]\t'
                      'Time {:.3f} ({:.3f})\t'
                      'Total loss {:.3f} ({:.3f})\t'
                      'Loss {:.3f}({:.3f})\t'
                      'LossMeta {:.3f}({:.3f})'
                      .format(epoch, i + 1, train_iters,
                              batch_time.val, batch_time.avg,
                              losses.val, losses.avg,
                              losses_meta_train.val, losses_meta_train.avg,
                              losses_meta_test.val, losses_meta_test.avg))

    def _parse_data(self, inputs):
        imgs, names, pids, cams, dataset_id, indexes, imgs_raw = inputs
        return imgs.cuda(), pids.cuda(), indexes.cuda(), cams.cuda(), dataset_id.cuda(), imgs_raw.cuda()


