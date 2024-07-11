# PFLlib: Personalized Federated Learning Algorithm Library
# Copyright (C) 2021  Jianqing Zhang

# This program is free software; you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation; either version 2 of the License, or
# (at your option) any later version.

# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.

# You should have received a copy of the GNU General Public License along
# with this program; if not, write to the Free Software Foundation, Inc.,
# 51 Franklin Street, Fifth Floor, Boston, MA 02110-1301 USA.

import copy
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
import numpy as np
import time
from flcore.clients.clientbase import Client
from sklearn.preprocessing import label_binarize
from sklearn import metrics
from utils.data_utils import read_client_data, DataHandler
from multiprocessing import Pool
from utils.SCL_losses import SupervisedContrastiveLoss, SCLWithCosMargin, SCLWithArcMargin, RelaxedCL, SCLWithEnhancedCosMargin, SCLWithPrototypePenalty, SCLWithMask, SCLWithcNCE


class clientASCL(Client):
    def __init__(self, args, id, train_samples, test_samples, **kwargs):
        super().__init__(args, id, train_samples, test_samples, **kwargs)

        self.margin_type = args.margin_type
        self.n_views = args.n_views
        self.class_prototypes = None
        self.class_labels = None
        self.id = id

        if self.margin_type == "raw":
            self.SCL_loss = SupervisedContrastiveLoss()
            self.SCL_loss = SCLWithcNCE(alpha=args.Alpha)
        else:
            raise NotImplementedError

    def train(self):
        if self.margin_type == "cNCE": # SupCon + class conditional infoNCE
            data_handler = DataHandler(dataset=self.dataset, client_id=self.id, batch_size=self.batch_size, n_views=self.n_views)
            trainloader = data_handler.load_augmented_train_data()
        else:
            trainloader = self.load_train_data()

        self.model.train()
        
        start_time = time.time()

        max_local_epochs = self.local_epochs
        if self.train_slow:
            max_local_epochs = np.random.randint(1, max_local_epochs // 2)

        for epoch in range(max_local_epochs):
            if self.margin_type == "sclp":
                self.compute_class_prototypes()
            for x, y in trainloader:
                if self.margin_type == "cNCE":
                    x_raw = x[0].to(self.device)
                    x_aug = x[1].to(self.device)
                elif type(x) == type([]):
                    x[0] = x[0].to(self.device)
                else:
                    x = x.to(self.device)
                y = y.to(self.device)

                if self.margin_type == "cNCE":
                    x = torch.cat([x_raw, x_aug], dim=0)
                if self.train_slow:
                    time.sleep(0.1 * np.abs(np.random.rand()))

                proj_feat, output = self.model(x)

                if self.margin_type == "cNCE":
                    f1, f2 = torch.split(proj_feat, [self.batch_size, self.batch_size], dim=0)
                    proj_feat = torch.cat([f1.unsqueeze(1), f2.unsqueeze(1)], dim=1)
                    output, _ = torch.split(output, [self.batch_size, self.batch_size], dim=0)

                loss = self.loss(output, y)
                loss += self.SCL_loss(proj_feat, y) 

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

        if self.learning_rate_decay:
            self.learning_rate_scheduler.step()

        self.train_time_cost['num_rounds'] += 1
        self.train_time_cost['total_cost'] += time.time() - start_time

    def train_metrics(self):
        if self.margin_type == "cNCE": # SupCon + class conditional infoNCE
            data_handler = DataHandler(dataset=self.dataset, client_id=self.id, batch_size=self.batch_size, n_views=self.n_views)
            trainloader = data_handler.load_augmented_train_data()
        else:
            trainloader = self.load_train_data()
        # self.model = self.load_model('model')
        # self.model.to(self.device)
        self.model.eval()

        train_num = 0
        sup_losses = 0
        ce_losses = 0

        with torch.no_grad():
            for x, y in trainloader:
                if self.margin_type == "cNCE":
                    x_raw = x[0].to(self.device)
                    x_aug = x[1].to(self.device)
                elif type(x) == type([]):
                    x[0] = x[0].to(self.device)
                else:
                    x = x.to(self.device)
                y = y.to(self.device)

                if self.margin_type == "cNCE":
                    x = torch.cat([x_raw, x_aug], dim=0)

                proj_feat, output = self.model(x)

                if self.margin_type == "cNCE":
                    f1, f2 = torch.split(proj_feat, [self.batch_size, self.batch_size], dim=0)
                    proj_feat = torch.cat([f1.unsqueeze(1), f2.unsqueeze(1)], dim=1)
                    output, _ = torch.split(output, [self.batch_size, self.batch_size], dim=0)

                loss = self.loss(output, y)
                ce_losses += loss.item() * y.shape[0]
                if self.SCL_loss != None:
                    sup_loss = self.SCL_loss(proj_feat, y) 
                    loss += sup_loss
                    sup_losses += sup_loss.item() * y.shape[0]
                
                train_num += y.shape[0]
                
        return ce_losses, sup_losses, train_num
    
    def test_metrics(self):
        testloaderfull = self.load_test_data()
        # self.model = self.load_model('model')
        # self.model.to(self.device)
        self.model.eval()

        test_acc = 0
        test_num = 0
        y_prob = []
        y_true = []
        
        with torch.no_grad():
            for x, y in testloaderfull:
                if type(x) == type([]):
                    x[0] = x[0].to(self.device)
                else:
                    x = x.to(self.device)
                y = y.to(self.device)
                _, output = self.model(x)

                test_acc += (torch.sum(torch.argmax(output, dim=1) == y)).item()
                test_num += y.shape[0]

                y_prob.append(output.detach().cpu().numpy())
                nc = self.num_classes
                if self.num_classes == 2:
                    nc += 1
                lb = label_binarize(y.detach().cpu().numpy(), classes=np.arange(nc))
                if self.num_classes == 2:
                    lb = lb[:, :2]
                y_true.append(lb)

        y_prob = np.concatenate(y_prob, axis=0)
        y_true = np.concatenate(y_true, axis=0)

        auc = metrics.roc_auc_score(y_true, y_prob, average='micro')
        
        return test_acc, test_num, auc