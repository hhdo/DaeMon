# Adaptive Path-Memory Network for Temporal Knowledge Graph Reasoning

This is the official code release of the following paper:

Hao Dong, Zhiyuan Ning, Pengyang Wang, Ziyue Qiao, Pengfei Wang, Yuanchun Zhou and Yanjie Fu. "[Adaptive Path-Memory Network for Temporal Knowledge Graph Reasoning](https://arxiv.org/abs/2304.12604)." IJCAI 2023.

<img src="https://github.com/hhdo/DaeMon/blob/main/img/DaeMon.png" alt="DaeMon_Architecture" width="800" class="center">

## Quick Start

### Dependencies

```
python==3.8
torch==1.10.0
torchvision==0.11.1
dgl-cu113==0.9.1
tqdm
torch-scatter>=2.0.8
pyg==2.0.4
```

### Train models

0. Switch to `src/` folder
```
cd src/
``` 

1. Run scripts

```
python main.py --gpus 0 -d YAGO --batch_size 64 --history_len 10
```

- To run with multiple GPUs which is **highly recommended**, use the following commands
```
python -m torch.distributed.launch --nproc_per_node=4 main.py --gpus 0 1 2 3 -d YAGO --batch_size 16 --history_len 10
```

### Evaluate models

To generate the evaluation results of a pre-trained model (if exist), simply add the `--test` flag in the commands above.

```
python main.py --gpus 0 -d YAGO --batch_size 64 --history_len 10 --test
```

### Change the hyperparameters
To get the optimal result reported in the paper, change the hyperparameters and other setting according to the details of Section 5.1 in the [paper](https://arxiv.org/abs/2304.12604). 

## Citation
If you find the resource in this repository helpful, please cite

```bibtex
@article{dong2023adaptive,
  title={Adaptive Path-Memory Network for Temporal Knowledge Graph Reasoning},
  author={Dong, Hao and Ning, Zhiyuan and Wang, Pengyang and Qiao, Ziyue and Wang, Pengfei and Zhou, Yuanchun and Fu, Yanjie},
  journal={arXiv preprint arXiv:2304.12604},
  year={2023}
}
```
