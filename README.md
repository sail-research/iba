# IBA: Towards Irreversible Backdoor Attacks in Federated Learning (NeurIPS'23)

## 1. Setup

1. Set up the environment by installing the dependencies from the `environment.yml` file:

```bash
make install
```
2. The pretrained models are stored at `./checkpoint`.
3. Logs will be saved at `./results`.
4. MNIST/CIFAR-10 datasets will be automatically downloaded. Please download and process Tiny-ImageNet-200 from https://github.com/AI-secure/DBA. The raw data could be downloaded using `wget http://cs231n.stanford.edu/tiny-imagenet-200.zip`.
The checkpoint for imagenet can be downloaded [here](https://vanderbilt.box.com/s/o3w002m4b9w7l5hysr5ob3xbn5ieo670).



## 2. Running the Experiments
To reproduce the results presented in the paper, run the following commands:
```bash
./exps/iba_cifar10_fedavg.sh # IBA under FedAvg (fixed-freq)

./exps/iba_cifar10_fedavg_multikrum.sh # IBA with CIFAR10 under MultiKrum

./exps/iba_mnist_fedavg_multikrum.sh # IBA with MNIST under MultiKrum

./exps/iba_cifar10_fedavg_fixed_pool.sh # # IBA with CIFAR10 (fixed-pool)
```

You can modify the following parameters to reproduce different scenarios:

- --dataset: Specify the dataset.
- --model: Select the model.
- --defense_method: Choose the defense method.
- --attack_method: Select the attack method.
- --attack_freq: Set the attack frequency.
- --model_replacement: Enable or disable model replacement.
- --attacker_pool_size: Number of attackers
- --num_nets: Total number of clients

Please refer to the paper for detailed results and additional configurations.

# Acknowledgements
This code is based on the setup for backdoor attack and defense in Federated Learning (FL) from the repository [Attack of the Tails: Yes, You Really Can Backdoor Federated Learning](https://github.com/ksreenivasan/OOD_Federated_Learning).

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

# License

This project is licensed under the [MIT License](LICENSE).  
You are free to use, modify, and distribute this software under the terms of the license.
