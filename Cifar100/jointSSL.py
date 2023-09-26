from turtle import update
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms
from torch.nn import functional as F
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import StepLR, CosineAnnealingLR
from torch.autograd import Variable
import os
import sys
import numpy as np
from myNetwork import joint_network
from iCIFAR100 import iCIFAR100, TwoCropsTransform
import logging

class jointSSL:
    def __init__(self, args, encoder, task_size, device):
        self.args = args
        self.size = 32
        self.epochs = args.epochs
        self.learning_rate = args.learning_rate
        self.model = joint_network(args.fg_nc, encoder)
        self.radius = 0
        self.prototype = None
        # self.class_label = None
        self.numsamples = None
        self.numclass = args.fg_nc
        self.task_size = task_size
        self.device = device
        self.old_model = None
        self.train_transform = transforms.Compose([transforms.RandomCrop((32, 32), padding=4),
                                                  transforms.RandomHorizontalFlip(p=0.5),
                                                  transforms.ColorJitter(brightness=0.24705882352941178),
                                                  transforms.ToTensor(),
                                                  transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761))])
        self.test_transform = transforms.Compose([transforms.ToTensor(),
                                                 transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761))])
        self.train_dataset = iCIFAR100(args.root, transform=self.train_transform, download=True)
        self.test_dataset = iCIFAR100(args.root, test_transform=self.test_transform, train=False, download=True)
        self.train_loader = None
        self.test_loader = None

    def beforeTrain(self, current_task):
        self.model.eval()
        if current_task == 0:
            classes = [0, self.numclass]
        else:
            classes = [self.numclass-self.task_size, self.numclass]
        self.train_loader, self.test_loader = self._get_train_and_test_dataloader(classes)
        if current_task > 0:
            self.model.Incremental_learning(self.numclass)  # 保存fc层之前学的参数
        self.model.train()
        self.model.to(self.device)

    def _get_train_and_test_dataloader(self, classes):
        self.train_dataset.getTrainData(classes)
        self.test_dataset.getTestData(classes)
        train_loader = DataLoader(dataset=self.train_dataset,
                                  shuffle=True,
                                  batch_size=self.args.batch_size)

        test_loader = DataLoader(dataset=self.test_dataset,
                                 shuffle=True,
                                 batch_size=self.args.batch_size)

        return train_loader, test_loader

    def _get_test_dataloader(self, classes):
        self.test_dataset.getTestData_up2now(classes)
        test_loader = DataLoader(dataset=self.test_dataset,
                                 shuffle=True,
                                 batch_size=self.args.batch_size)
        return test_loader

    def train(self, current_task, old_class=0):
        opt = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate, weight_decay=2e-4)
        scheduler = CosineAnnealingLR(opt, T_max=32)
        accuracy = 0
        for epoch in range(self.epochs):
            running_loss = 0.0
            scheduler.step()
            for step, (indexs, images, labels) in enumerate(self.train_loader):
                images, labels = images.to(self.device), labels.to(self.device)

                images = torch.stack([torch.rot90(images, k, (2, 3)) for k in range(4)], 1)
                images = images.view(-1, 3, self.size, self.size)
                joint_labels = torch.stack([labels * 4 + k for k in range(4)], 1).view(-1)

                opt.zero_grad()
                loss = self._compute_loss(images, joint_labels, labels, old_class)
                opt.zero_grad()
                loss.backward()
                running_loss += loss.item()
                opt.step()
            if epoch % self.args.print_freq == 0:
                accuracy = self._test(self.test_loader)
                print('epoch:%d, accuracy:%.5f' % (epoch, accuracy))
                logging.info('task:%d, epoch:%d, accuracy:%.5f' % (current_task, epoch, accuracy))
            logging.info('train loss:%.6f'%(running_loss / len(self.train_loader)))
        self.protoSave(self.model, self.train_loader, current_task)

    def _test(self, testloader):
        self.model.eval()
        correct, total = 0.0, 0.0
        for setp, (indexs, imgs, labels) in enumerate(testloader):
            imgs, labels = imgs.to(self.device), labels.to(self.device)
            with torch.no_grad():
                outputs = self.model(imgs)
            predicts = torch.max(outputs, dim=1)[1]
            correct += (predicts.cpu() == labels.cpu()).sum()
            total += len(labels)
        accuracy = correct.item() / total
        self.model.train()
        return accuracy

    def _compute_loss(self, imgs, joint_labels, labels, old_class=0):
        feature = self.model.feature(imgs)

        joint_preds = self.model.fc(feature)
        single_preds = self.model.classifier(feature)[::4]
        joint_preds, joint_labels, single_preds, labels = joint_preds.to(self.device), joint_labels.to(self.device), single_preds.to(self.device), labels.to(self.device)

        joint_loss = nn.CrossEntropyLoss()(joint_preds/self.args.temp, joint_labels)
        signle_loss = nn.CrossEntropyLoss()(single_preds/self.args.temp, labels)

        agg_preds = 0
        for i in range(4):
            agg_preds = agg_preds + joint_preds[i::4, i::4] / 4

        distillation_loss = F.kl_div(F.log_softmax(single_preds, 1),
                                    F.softmax(agg_preds.detach(), 1),
                                    reduction='batchmean')

        if old_class == 0:
            return joint_loss + signle_loss + distillation_loss
        else:
            feature_old = self.old_model.feature(imgs)
            loss_kd = torch.dist(feature, feature_old, 2)

            proto_aug = []
            proto_aug_label = []
            old_class_list = list(self.prototype.keys())
            for _ in range(feature.shape[0] // 4): # batch_size = feature.shape[0] // 4
                i = np.random.randint(0, feature.shape[0])
                np.random.shuffle(old_class_list)
                lam = np.random.beta(0.5, 0.5)
                if lam > 0.6:
                    # lam = 1 - lam
                    lam = lam * 0.6
                if np.random.random() >= 0.5:
                    temp = (1 + lam) * self.prototype[old_class_list[0]] - lam * feature.detach().cpu().numpy()[i]
                else:
                    temp = (1 - lam) * self.prototype[old_class_list[0]] + lam * feature.detach().cpu().numpy()[i]
                # temp = (1 - lam) * self.prototype[old_class_list[0]] + lam * feature.detach().cpu().numpy()[i]

                proto_aug.append(temp)
                proto_aug_label.append(old_class_list[0])

            proto_aug = torch.from_numpy(np.float32(np.asarray(proto_aug))).float().to(self.device)
            proto_aug_label = torch.from_numpy(np.asarray(proto_aug_label)).to(self.device)
            aug_preds = self.model.classifier(proto_aug)
            joint_aug_preds = self.model.fc(proto_aug)

            agg_preds = 0
            agg_preds = agg_preds + joint_aug_preds[:, ::4]

            aug_distillation_loss = F.kl_div(F.log_softmax(aug_preds, 1),
                                            F.softmax(agg_preds.detach(), 1),
                                            reduction='batchmean')
            loss_protoAug = nn.CrossEntropyLoss()(aug_preds/self.args.temp, proto_aug_label) + nn.CrossEntropyLoss()(joint_aug_preds/self.args.temp, proto_aug_label*4) + aug_distillation_loss

            return joint_loss + signle_loss + distillation_loss + self.args.protoAug_weight*loss_protoAug + self.args.kd_weight*loss_kd


    def afterTrain(self, log_root):
        path = log_root + '/'
        if not os.path.isdir(path):
            os.makedirs(path)
        self.numclass += self.task_size
        filename = path + '%d_model.pkl' % (self.numclass - self.task_size)
        torch.save(self.model, filename)
        self.old_model = torch.load(filename)
        self.old_model.to(self.device)
        self.old_model.eval()

    def protoSave(self, model, loader, current_task):
        features = []
        labels = []
        model.eval()
        with torch.no_grad():
            for i, (indexs, images, target) in enumerate(loader):

                feature = model.feature(images.to(self.device))
                if feature.shape[0] == self.args.batch_size:
                    labels.append(target.numpy())
                    features.append(feature.cpu().numpy())
        labels_set = np.unique(labels)  # 从大到小
        labels = np.array(labels)
        labels = np.reshape(labels, labels.shape[0] * labels.shape[1])
        features = np.array(features)
        features = np.reshape(features, (features.shape[0] * features.shape[1], features.shape[2]))
        # feature_dim = features.shape[1]

        prototype = {}
        # radius = []
        class_label = []
        numsamples = {}
        for item in labels_set:
            index = np.where(item == labels)[0]
            class_label.append(item)
            feature_classwise = features[index]
            prototype[item] = np.mean(feature_classwise, axis=0)
            numsamples[item] = feature_classwise.shape[0]

        if current_task == 0:
            # self.radius = np.sqrt(np.mean(radius))
            self.prototype = prototype
            self.class_label = class_label
            self.numsamples = numsamples
        else:
            self.prototype.update(prototype)
            self.class_label = np.concatenate((class_label, self.class_label), axis=0)
            self.numsamples.update(numsamples)
