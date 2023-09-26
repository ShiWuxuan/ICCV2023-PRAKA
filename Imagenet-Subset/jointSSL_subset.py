import torch
from torchvision import transforms
from torch.nn import functional as F
from PIL import Image
import torch.optim as optim
from torch.utils.data import DataLoader
import torch.nn as nn
from torch.optim.lr_scheduler import StepLR, CosineAnnealingLR
from torch.autograd import Variable
import numpy as np
import os
import sys
from myNetwork_imagenet import joint_network
from data_manager_imagenet import *


class jointSSL:
    def __init__(self, args, feature_extractor, task_size, device):
        self.args = args
        self.model = joint_network(args.fg_nc, feature_extractor)
        self.size = 224
        self.prototype = None
        self.numclass = args.fg_nc
        self.task_size = task_size
        self.old_model = None
        self.device = device
        self.data_manager = DataManager()
        self.train_transform = transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
        ])
        self.test_transform = transforms.Compose([
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
        ])
        self.train_loader = None
        self.test_loader = None

    def beforeTrain(self, current_task):
        self.model.eval()
        class_set = list(range(100))
        if current_task == 0:
            classes = class_set[:self.numclass]
        else:
            classes = class_set[self.numclass-self.task_size: self.numclass]
        print(classes)

        trainfolder = self.data_manager.get_dataset(self.train_transform, index=classes, train=True)
        testfolder = self.data_manager.get_dataset(self.test_transform, index=class_set[:self.numclass], train=False)

        self.train_loader = torch.utils.data.DataLoader(trainfolder, batch_size=self.args.batch_size,
            shuffle=True, drop_last=True, num_workers=8)
        self.test_loader = torch.utils.data.DataLoader(testfolder, batch_size=self.args.batch_size,
            shuffle=False, drop_last=False, num_workers=8)

        if current_task > 0:
            self.model.Incremental_learning(self.numclass)
        self.model.to(self.device)
        self.model.train()

    def train(self, current_task, old_class=0):
        opt = torch.optim.Adam(self.model.parameters(), lr=self.args.learning_rate, weight_decay=2e-4)
        scheduler = CosineAnnealingLR(opt, T_max=32)
        for epoch in range(self.args.epochs):
            # print('training epoch!')
            running_loss = 0.0
            scheduler.step()
            for step, data in enumerate(self.train_loader):
                images, labels = data
                images, labels = images.to(self.device), labels.to(self.device)

                # # self-supervised learning based label augmentation
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
                self._test_per_task(current_task)
                logging.info('task: %d, epoch:%d, accuracy:%.5f' % (current_task, epoch, accuracy))
            logging.info('train loss:%.6f'%(running_loss / len(self.train_loader)))
        self.protoSave(self.model, self.train_loader, current_task)

    def _test(self, testloader):
        self.model.eval()
        correct, total = 0.0, 0.0
        for setp, data in enumerate(testloader):
            imgs, labels = data
            imgs, labels = imgs.to(self.device), labels.to(self.device)
            with torch.no_grad():
                outputs = self.model(imgs)
            predicts = torch.max(outputs, dim=1)[1]
            correct += (predicts.cpu() == labels.cpu()).sum()
            total += len(labels)
        accuracy = correct.item() / total
        self.model.train()
        return accuracy

    def _test_per_task(self, current_task):
        class_set = list(range(100))
        logging.info('==================== Test for each Task ====================') # 单独测试每个任务的性能
        self.model.eval()

        acc_up2now = []
        for i in range(current_task+1):
            if i == 0:
                classes = class_set[:self.args.fg_nc]
            else:
                classes = class_set[(self.args.fg_nc + (i-1)*self.task_size):(self.args.fg_nc + i*self.task_size)]
            test_dataset = self.data_manager.get_dataset(self.test_transform, index=classes, train=False)
            test_loader = DataLoader(dataset=test_dataset, shuffle=False, batch_size=self.args.batch_size)

            correct, total = 0.0, 0.0
            for setp, (imgs, labels) in enumerate(test_loader):
                imgs, labels = imgs.to(self.device), labels.to(self.device)
                with torch.no_grad():
                    outputs = self.model(imgs)
                # outputs = outputs[:, ::4]
                predicts = torch.max(outputs, dim=1)[1]
                correct += (predicts.cpu() == labels.cpu()).sum()
                total += len(labels)
            accuracy = correct.item() / total
            acc_up2now.append(accuracy)
        if current_task < self.args.task_num:
            acc_up2now.extend((self.args.task_num-current_task)*[0])
        logging.info("phase%d:" % (current_task))
        logging.info(acc_up2now)
        self.model.train()

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
            for _ in range(feature.shape[0]//4):
                i = np.random.randint(0, feature.shape[0])
                np.random.shuffle(old_class_list)
                lam = np.random.beta(0.5, 0.5)
                if lam > 0.6:
                    lam = lam * 0.6
                
                # if np.random.random() >= 0.5:
                #     temp = (1 + lam) * self.prototype[old_class_list[0]] - lam * feature.detach().cpu().numpy()[i]
                # else:
                #     temp = (1 - lam) * self.prototype[old_class_list[0]] + lam * feature.detach().cpu().numpy()[i]
                temp = (1 - lam) * self.prototype[old_class_list[0]] + lam * feature.detach().cpu().numpy()[i]
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
            for step, data in enumerate(loader):
                images, target = data
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
        # class_label = []
        # numsamples = {}
        for item in labels_set:
            index = np.where(item == labels)[0]
            # class_label.append(item)
            feature_classwise = features[index]
            prototype[item] = np.mean(feature_classwise, axis=0)
            # numsamples[item] = feature_classwise.shape[0]
            # if current_task == 0:
            #     cov = np.cov(feature_classwise.T) # 计算协方差
            #     radius.append(np.trace(cov) / feature_dim) 

        if current_task == 0:
            # self.radius = np.sqrt(np.mean(radius))
            self.prototype = prototype
            # self.class_label = class_label
            # self.numsamples = numsamples
            # print(self.radius)
        else:
            self.prototype.update(prototype)
            # self.class_label = np.concatenate((class_label, self.class_label), axis=0)
            # self.numsamples.update(numsamples)


