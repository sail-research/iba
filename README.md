# Backdoor Attack and Defense in Federated Learning (FL)
This code is based on the setup for backdoor attack and defense in Federated Learning (FL) from the repository [Attack of the Tails: Yes, You Really Can Backdoor Federated Learning](https://github.com/ksreenivasan/OOD_Federated_Learning).

## 1. Setup

1. Set up the environment by installing the dependencies from the `environment.yml` file:

```bash
make install
```
2. Download the data and pretrained models for MNIST, CIFAR-10, and Tiny-ImageNet from the [DBA Github repository](https://github.com/AI-secure/DBA).



## 2. Running the Experiments
To reproduce the results presented in the paper, run the following commands:
```bash
./exps/mnist-multi-krum.sh
./exps/cifar10-multi-krum.sh 
./exps/timagenet-multi-krum.sh 
```

You can modify the following parameters to reproduce different scenarios:

- --dataset: Specify the dataset.
- --model: Select the model.
- --defense_method: Choose the defense method.
- --attack_method: Select the attack method.
- --attack_freq: Set the attack frequency.
- --attack_case: Specify the attack case.
- --model_replacement: Enable or disable model replacement.

Please refer to the paper for detailed results and additional configurations.

Please note that I have made some assumptions and corrections based on the given context. Make sure to review and adjust the changes if necessary.

Please cite the paper, as below, when using this repository:
```
@inproceedings{
    nguyen2023iba,
    title={{IBA}: Towards Irreversible Backdoor Attacks in Federated Learning},
    author={Dung Thuy Nguyen and Tuan Minh Nguyen and Anh Tuan Tran and Khoa D Doan and KOK-SENG WONG},
    booktitle={Thirty-seventh Conference on Neural Information Processing Systems},
    year={2023},
    url={https://openreview.net/forum?id=cemEOP8YoC}
}
```

