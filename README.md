# Generalized Neural Sorting Networks with Error-Free Differentiable Swap Functions

It is the official repository of "[Generalized Neural Sorting Networks with Error-Free Differentiable Swap Functions](https://arxiv.org/abs/2310.07174)," which has been presented at [the 12th International Conference on Learning Representations (ICLR 2024)](https://iclr.cc/Conferences/2024).

For part of this repository, we modified the source code obtained from [the repository linked](https://github.com/Felix-Petersen/diffsort).

## Installation

You can install some required packages using the following command.

```shell
pip install .
```

## Experiments

You can run experiments with `src/main_multidigit.py` and `src/main_jigsaw.py`.

For example, they are run with the following commands in the `src` directory.

```shell
# experiments on sorting multi-digit images
python main_multidigit.py --dataset mnist_cnn
python main_multidigit.py --dataset svhn_cnn
python main_multidigit.py --dataset mnist_transformer
python main_multidigit.py --dataset svhn_transformer
python main_multidigit.py --dataset mnist_transformer_large
python main_multidigit.py --dataset svhn_transformer_large

# experiments on sorting image fragments
python main_jigsaw.py --dataset mnist_2_2 --model cnn
python main_jigsaw.py --dataset mnist_2_2 --model transformer
python main_jigsaw.py --dataset mnist_3_3 --model cnn
python main_jigsaw.py --dataset mnist_3_3 --model transformer
python main_jigsaw.py --dataset cifar10_2_2 --model cnn
python main_jigsaw.py --dataset cifar10_2_2 --model transformer
python main_jigsaw.py --dataset cifar10_3_3 --model cnn
python main_jigsaw.py --dataset cifar10_3_3 --model transformer
```

You can test different learning rate and steepness using options `--nloglr` and `--steepness`, respectively.

## Citation

```
@inproceedings{KimJ2024iclr,
    title={Generalized Neural Sorting Networks with Error-Free Differentiable Swap Functions},
    author={Kim, Jungtaek and Yoon, Jeongbeen and Cho, Minsu},
    booktitle={Proceedings of the International Conference on Learning Representations (ICLR)},
    year={2024},
    address={Vienna, Austria)
}
```

## License

[MIT License](LICENSE)
