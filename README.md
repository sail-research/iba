# IBA: Towards Irreversible Backdoor Attacks in Federated Learning (NeurIPS'23)

## 1. Setup

1. Set up the environment by installing the dependencies from the `environment.yml` file:

```bash
make install
```
2. The pretrained models are stored at `./checkpoint`.
3. Logs will be saved at `./results`.

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

