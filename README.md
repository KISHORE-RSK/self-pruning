# self-pruning-neural-network
This project implements a self-pruning neural network in PyTorch that learns to eliminate unnecessary connections during training, improving model efficiency without manual intervention.

Objective

To design a neural network that can:

Automatically identify redundant weights
Reduce model complexity
Maintain competitive accuracy
 Core Idea

Each weight in the network is paired with a learnable gate:

Gate value → determines importance of a connection
Low gate value → connection is pruned
High gate value → connection is retained

The model learns this behavior during training using L1 regularization.

 Methodology
Built custom PrunableLinear layers
Introduced sigmoid-based gating mechanism
Applied sparsity constraint using:
Loss = CrossEntropy + λ × L1(Gates)
Implemented threshold-based pruning during forward pass


Key Highlights
✅ Dynamic pruning during training (no separate pruning phase)
✅ Lightweight architecture with reduced parameters
✅ Visual analysis of pruning using gate distributions
✅ Explored accuracy vs sparsity trade-off

Experimental Setup
Dataset: CIFAR-10
Model: Multi-layer perceptron with 4 layers
Optimizer: Adam
Framework: PyTorch

 Observations
Increasing λ leads to higher sparsity
Moderate pruning retains accuracy while reducing complexity
Gate distributions show clear separation between active and pruned weights

 Tech Stack
Python
PyTorch
NumPy
Matplotlib

 Run the Project
pip install -r requirements.txt
python train.py
