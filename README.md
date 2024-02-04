## Controls: Deep Learning Models for Experimental Controls

#### Overview

This GitHub repository, "controls," is designed to host deep learning models that function as experimental controls. The models included are pivotal for research experiments aimed at publication. They are meticulously crafted to ensure that each model can be trained from scratch. This criterion is critical to guarantee that the controls are executed on identical hardware for the same duration, providing a consistent baseline for comparison.

#### Models

Currently, the repository includes two primary models:

* A Fully Connected Neural Network (FCNN) trained on the MNIST dataset.
* A Residual Neural Network (ResNet) model trained on the CIFAR10 dataset.

Requirements

* PyTorch 2.1
* Python 3.10
* TorchMetrics
* TorchVision
* TensorBoard

## Execution

Execute models using PyTorch with the following commands:

```
# ResNet on MNIST
python3 models/run_fcnn_cntrl.py --epochs 256 --batch 1024 --basepath results/fcnn_cntrl

# ResNet on CIFAR10
python3 models/run_resnet_cntrl.py --step 0.0001 --epochs 256 --device cuda --batch 512 --basepath results/resnet_cntrl

# Visualize results
tensorboard --logdir=results
```

## Performance

* MNIST (FCNN): ~98.5% accuracy.
* CIFAR10 (ResNet): ~92.5% accuracy.

## Summary

This repository provides deep learning models for consistent experimental controls, requiring specific software and libraries for operation, and includes detailed instructions for model training and expected performance outcomes.
