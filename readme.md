# Understanding the Robustness of Distributed Self-Supervised Learning Frameworks Against Non-IID Data

The PyTorch implementation for paper "Understanding the Robustness of Distributed Self-Supervised Learning Frameworks Against Non-IID Data".

## Environment Installation

1. Install Anaconda from the [Anaconda official website](https://www.anaconda.com/)
2. Run the following commands to install the virtual environment

```
cd MAE_non_iid_study
conda env create -f environment.yml
```
3. Run the following command to switch to the installed virtual environment
```
conda activate fedMae
```
4. Find the appropriate command from the [PyTorch official website](https://pytorch.org/) to install PyTorch framework to the virtual environment

5. For any issue of missing library, try with the installation command:
```
pip install xxx
```

## Instruction 

### -- Dataset Preparation

Our codes support the following datasets:

0. CIFAR-10 (Supported by PyTorch)
1. CIFAR-100 (Supported by PyTorch)
2. food-101 (Supported by PyTorch)
3. ImageNet (Need to download from [ImageNet official website](https://image-net.org/index.php). How to extract: [link](https://gist.github.com/BIGBALLON/8a71d225eff18d88e469e6ea9b39cef4))
4. Mini-ImageNet (After downloading ImageNet, see repo [Tools for mini-ImageNet Dataset](https://github.com/yaoyao-liu/mini-imagenet-tools#about-mini-ImageNet))
5. ImageNet32 (Need to download from [ImageNet official website](https://image-net.org/index.php))
6. Mini-INat2021 (Need to download from repo [iNaturalist Competition Datasets](https://github.com/visipedia/inat_comp/tree/master/2021))

For datasets that need to be additionally downloaded, also remember to specify the correct path to data folder in **prepare_dataset.py** after the download.

### -- Federated Pretraining


Example 1:
* Pretrain with Tiny-ImageNet dataset
* Use MAE pre-training (the MAR loss with gamma=0.001)
* Choose ViT backbone
* Client data follows Non-IID with Dirichet Parameter 0.1
* Default federated learning settings
```
python main.py -p pretrain -d 4 -samp iid -alpha 0.1 -rd 100 -le 2 -tc 100 -sc 5 -bl 0 -so 0 -gamma 1e-3
```

Example 2:
* Pretrain with CIFAR-10 dataset
* Use Simsiam pre-training 
* Choose ResNet backbone
* Client data follows IID distribution
* Default federated learning settings
```
python main.py -p pretrain -d 0 -samp iid -bl 1 -so 0 
```

### -- Decentralized Pretraining

Example 1:
* Pretrain with Tiny-ImageNet dataset
* Use MAE pre-training (the MAR loss with gamma=0.01)
* Choose ViT backbone
* Client data follows Non-IID with Dirichet Parameter 0.01
* Run 50 rounds
* Train in decentralized setup with 20 clients and the maximum connectivity is 10
```
python main.py -p pretrain -d 4 -samp dir -alpha 0.01 -rd 50 -tc 20 -mc 10 -bl 0 -so 1 -gamma 1e-2
```

### -- Finetuning (Requires pretrained checkpoint)

Example 1:
* Require: the folder directory for the checkpoint of pretrained model 
* Pretraining is completed in federated learning framework
* Model Backbone is ViT
* Finetuning with CIFAR-10 dataset
* Use 100% labeled data
```
python main.py -p finetune -d 0 -ra 1 -bl 0 -so 0
```

Example 2:
* Decentralized pretraining has completed
* Model Backbone is ResNet
* Finetuning with CIFAR-100 dataset
* Use 10% labeled data
```
python main.py -p finetune -d 1 -ra 0.1 -bl 1 -so 1
```


## File Structure

```
├── util/ <codes under this directory are taken from MAE repo>
├── debug.py <codes for debugging>
├── engine_finetune.py <the training engine for finetuning>
├── engine_pretrain.py <the training engine for pretraining>
├── environment.yaml <information about the conda environment>
├── decentralized.py <includes codes for decentralized learning>
├── federated_learning.py <includes codes for federated learning>
├── find_mean_std.py <codes that can be run to find mean and std of dataset>
├── imagenetLoad.py <dataloader codes for ImageNet32>
├── main.py <file for switch between different experiments and running the program>
├── baseline_models.py <implementation of default Simsiam>
├── model_mae_resnet.py <implementation of pre-training ResNet with MAE>
├── model_mae.py <implementation of default MAE model>
├── model_ViT_simsiam.py <implementation of pre-training ViT with Simsiam>
├── model_ViT.py <implementation of ViT model>
├── prepare_dataset.py <codes for loading datsets, preparing dataloaders and data division>
├── readme.md <ReadMe file>
├── train_client.py <main program of local pretraining on clients>
├── train_server.py <main program of finetuning>
├── visualization.py <program for visualizing the feature space>
├── ammd.py <program for implementing the ammd loss and other tricks>
```

# Reference

This code is implemented based on the repository [Masked Autoencoders: A PyTorch Implementation](https://github.com/facebookresearch/deit), which is the PyTorch implementation of paper [Masked Autoencoders Are Scalable Vision Learners](https://arxiv.org/abs/2111.06377):
```
@Article{MaskedAutoencoders2021,
  author  = {Kaiming He and Xinlei Chen and Saining Xie and Yanghao Li and Piotr Doll{\'a}r and Ross Girshick},
  journal = {arXiv:2111.06377},
  title   = {Masked Autoencoders Are Scalable Vision Learners},
  year    = {2021},
}
```

In the MAE repo, it also has references to the following projects:
 * [DeiT repo](https://github.com/facebookresearch/deit)
 * [timm](https://github.com/rwightman/pytorch-image-models)
 * [ELECTRA](https://github.com/google-research/electra)
 * [BEiT](https://github.com/microsoft/unilm/tree/master/beit)
 * [MoCo v3](https://github.com/facebookresearch/moco-v3)
 * [Transformer](https://github.com/tensorflow/models/blob/master/official/nlp/transformer/model_utils.py)


