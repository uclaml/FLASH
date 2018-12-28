# Fast Local minimA finding with third-order SmootHness (FLASH)

This repository contains pytorch code that produces the local minma finding algorithm in the paper: [Third-order Smoothness Helps: Faster Stochastic Optimization Algorithms for Finding Local Minima](http://papers.nips.cc/paper/7704-third-order-smoothness-helps-faster-stochastic-optimization-algorithms-for-finding-local-minima.pdf).

## Prerequisites:
* Python (3.6.4)
* Pytorch (0.4.1)
* NumPy
* CUDA

## Command Line Arguments:
* --LR-SCSG: learning rate for scsg
* --LR-NEG: learning rate for negative curvature descent
* --EPOCH: total epoch for the algorithm
* --BATCH-SIZE: mini batch size for scsg in training


## Reference
* [Third-order Smoothness Helps: Faster Stochastic Optimization Algorithms for Finding Local Minima](http://papers.nips.cc/paper/7704-third-order-smoothness-helps-faster-stochastic-optimization-algorithms-for-finding-local-minima.pdf). Yaodong Yu*, Pan Xu* and Quanquan Gu, (*: equal contribution). NeurIPS-2018.
