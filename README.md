# attention_icm
A WIP implementation of Intrinsic Curiosity Module (https://arxiv.org/abs/1705.05363) in PyTorch.
Main implementation is finished, but correctness still needs to be confirmed by training.

A lot of the A3C code comes from [this repository](https://github.com/ikostrikov/pytorch-a3c).

## Running
Example

```python3 main.py --num-processes=16 --model=runs/test00 --icm=True```

Other parameters are also available
