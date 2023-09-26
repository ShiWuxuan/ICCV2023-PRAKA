import torch
import torch.nn as nn
from torchvision import transforms
from torch.utils.data import DataLoader
from torch.autograd import Variable
import numpy as np
import os
import sys
import argparse
from jointSSL_tiny import jointSSL
from ResNet import resnet18_cbam
from data_manager_tiny import *
import logging
from datetime import datetime

parser = argparse.ArgumentParser(description='Prototype Expansion and for Non-Exampler Class Incremental Learning')
parser.add_argument('--epochs', default=100, type=int, help='Total number of epochs to run')
parser.add_argument('--batch_size', default=128, type=int, help='Batch size for training')
parser.add_argument('--print_freq', default=10, type=int, help='print frequency (default: 10)')
parser.add_argument('--data_name', default='tiny', type=str, help='Dataset name to use')
parser.add_argument('--total_nc', default=200, type=int, help='class number for the dataset')
parser.add_argument('--fg_nc', default=100, type=int, help='the number of classes in first task')
parser.add_argument('--task_num', default=10, type=int, help='the number of incremental steps')
parser.add_argument('--learning_rate', default=0.001, type=float, help='initial learning rate')
parser.add_argument('--protoAug_weight', default=15.0, type=float, help='protoAug loss weight')
parser.add_argument('--kd_weight', default=15.0, type=float, help='knowledge distillation loss weight')
parser.add_argument('--temp', default=0.1, type=float, help='trianing time temperature')
parser.add_argument('--gpu', default='0', type=str, help='GPU id to use')
parser.add_argument('--save_path', default='../model_saved_check/', type=str, help='save files directory')
parser.add_argument('--seed', default=0, type=int, help='random seed')
parser.add_argument('--root', default='/data/CL_data', type=str, help='data root directory')

args = parser.parse_args()
print(args)

def set_random_seed(seed: int) -> None:
    """
    Sets the seeds at a certain value.
    :param seed: the value to be set
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True


def main():
    s = datetime.now().strftime('%Y%m%d%H%M%S')
    log_root = 'PRAKA/log/%s/phases_%d/seed_%d-%s-b64-step'%(args.data_name, args.task_num, args.seed, s)
    if not os.path.exists(log_root): 
        os.makedirs(log_root)
    logging.basicConfig(filename='%s/train.log'%log_root,format='%(asctime)s %(message)s', level=logging.INFO)
    logging.info(args)
    set_random_seed(args.seed)
    cuda_index = 'cuda:' + args.gpu
    device = torch.device(cuda_index if torch.cuda.is_available() else "cpu")
    task_size = int((args.total_nc - args.fg_nc) / args.task_num)  # number of classes in each incremental step
    # file_name = args.data_name + '_' + str(args.fg_nc) + '_' + str(args.task_num) + '*' + str(task_size) + 'debug'
    feature_extractor = resnet18_cbam()
    data_manager = DataManager()

    model = jointSSL(args, feature_extractor, task_size, device)
    class_set = list(range(args.total_nc))

    for i in range(args.task_num+1):
        if i == 0:
            old_class = 0
        else:
            old_class = len(class_set[:args.fg_nc + (i - 1) * task_size])
        # print(classes)
        model.beforeTrain(i)
        model.train(i, old_class=old_class)
        model.afterTrain(log_root)


    ####### Test ######
    test_transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))])
    print("############# Test for each Task #############")
    acc_all = []
    for current_task in range(args.task_num+1):
        class_index = args.fg_nc + current_task*task_size
        filename = log_root + '/' + '%d_model.pkl' % (class_index)
        model = torch.load(filename)
        model.eval()
        acc_up2now = []
        for i in range(current_task+1):
            if i == 0:
                classes = class_set[:args.fg_nc]
            else:
                classes = class_set[(args.fg_nc + (i-1)*task_size):(args.fg_nc + i*task_size)]

            test_dataset = data_manager.get_dataset(test_transform, index=classes, train=False)
            test_loader = DataLoader(dataset=test_dataset, shuffle=False, batch_size=args.batch_size)
            correct, total = 0.0, 0.0
            for setp, (imgs, labels) in enumerate(test_loader):
                imgs, labels = imgs.to(device), labels.to(device)
                with torch.no_grad():
                    outputs = model(imgs)
                # outputs = outputs[:, ::4]
                predicts = torch.max(outputs, dim=1)[1]
                correct += (predicts.cpu() == labels.cpu()).sum()
                total += len(labels)
            accuracy = correct.item() / total
            acc_up2now.append(accuracy)
        if current_task < args.task_num:
            acc_up2now.extend((args.task_num-current_task)*[0])
        acc_all.append(acc_up2now)
        print(acc_up2now)
        logging.info("phase%d:" % (current_task))
        logging.info(acc_up2now)
    print(acc_all)

    a = np.array(acc_all)
    result = []
    for i in range(args.task_num + 1):
        if i == 0:
            result.append(0)
        else:
            res = 0
            for j in range(i + 1):
                res += (np.max(a[:, j]) - a[i][j])
            res = res / i
            result.append(100 * res)
    logging.info(30 * '#')
    logging.info("Forgetting result:")
    logging.info(result)

    print("############# Test for up2now Task #############")
    average_acc = 0
    for current_task in range(args.task_num+1):
        class_index = args.fg_nc + current_task*task_size
        filename = log_root + '/' + '%d_model.pkl' % (class_index)
        model = torch.load(filename)
        model.to(device)
        model.eval()

        classes = class_set[:args.fg_nc+current_task*task_size]
        test_dataset = data_manager.get_dataset(test_transform, index=classes, train=False)
        test_loader = DataLoader(dataset=test_dataset, shuffle=False, batch_size=args.batch_size)
        correct, total = 0.0, 0.0
        for setp, (imgs, labels) in enumerate(test_loader):
            imgs, labels = imgs.to(device), labels.to(device)
            with torch.no_grad():
                outputs = model(imgs)
            # outputs = outputs[:, ::4]
            predicts = torch.max(outputs, dim=1)[1]
            correct += (predicts.cpu() == labels.cpu()).sum()
            total += len(labels)
        accuracy = correct.item() / total
        print(accuracy)
        logging.info("phase%d: %.8f" % (current_task, accuracy))
        average_acc += accuracy
    logging.info('average incremental acc: %.8f' % (average_acc / (args.task_num+1)))


if __name__ == "__main__":
    main()
