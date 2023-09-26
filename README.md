# PRAKA - Official PyTorch Implementation

### [ICCV 2023] Prototype Reminiscence and Augmented Asymmetric Knowledge Aggregation for Non-Exemplar Class-Incremental Learning

Our code is coming soon.


## Usage

* Training on CIFAR-100 dataset:

```
$ python Cifar100/main.py --gpu 0 --task_num 10 --fg_nc 50 --root [your dataset path]
```
Arguments you can freely tweak given a dataset:

* --gpu: which gpu used
* --task_num: number of tasks for incremental learning
* --fg_nc: number of classes of initial tasks
* --root: path of datasets (replace [your dataset path] with your own dataset root)


## Citation
If you use this code for your research, please consider citing:

```

```

## Acknowledgement

This work is partially supported by the Key Research and Development Program of Hubei Province (2021BAA187), National Natural Science Foundation of China under Grant (62176188), Zhejiang lab (NO.2022NF0AB01), the Special Fund of Hubei Luojia Laboratory (220100015) and CAAI-Huawei MindSpore Open Fund.

**We thank the following repos providing helpful components/functions in our work.**
* [PASS](https://github.com/Impression2805/CVPR21_PASS)
