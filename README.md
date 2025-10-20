# Multilayer Perceptron (MLP) from Scratch

This project implements a complete **Multilayer Perceptron (MLP)** neural network from scratch using only NumPy and Python standard libraries. The implementation focuses on educational value, demonstrating core machine learning concepts through modular, well-documented code. The MLP is applied to a **breast cancer classification** task, distinguishing between benign and malignant tumors using medical features.

## Project Scope

### 🎯 **Learning Objectives**
- **Deep Understanding**: Implement neural networks from first principles without relying on high-level frameworks
- **Modular Design**: Create reusable, testable components following software engineering best practices
- **Mathematical Foundation**: Understand forward/backward propagation, activation functions, and optimization algorithms
- **Practical Application**: Apply MLP to real-world medical classification with comprehensive evaluation

### 🔬 **Technical Scope**
- **Core MLP Implementation**: Dense layers, activation functions, loss functions, optimizers
- **Advanced Features**: Weight initialization strategies, early stopping, data preprocessing
- **Optimization Algorithms**: SGD, Adam, RMSprop with configurable hyperparameters
- **Model Management**: Save/load functionality with complete state preservation
- **Experiment Framework**: Systematic hyperparameter tuning and performance analysis
- **Visualization**: Training dynamics, convergence analysis, and experiment comparison

### 📊 **Dataset & Task**
- **Dataset**: Breast Cancer Wisconsin (Diagnostic) Dataset
- **Task**: Binary classification (Benign vs Malignant)
- **Features**: 30 medical measurements (radius, texture, perimeter, etc.)
- **Samples**: ~570 training samples, ~113 test samples
- **Challenge**: High-dimensional feature space with potential class imbalance

## What I Learned: MLP Implementation Strategy

### 🏗️ **Modular Architecture Design**

The implementation follows a **layered, object-oriented approach** that mirrors the mathematical structure of neural networks:

For further reading behind the design choices and math, see [References & Reflection](#references--reflection).

#### **1. Abstract Base Classes (ABCs)**
```python
# Template pattern for consistent interfaces
class IWeightInitializer(ABC):
    @abstractmethod
    def gen_weights(self, input_size: int, output_size: int) -> np.ndarray:
        pass

class IOptimizer(ABC):
    @abstractmethod
    def update(self, layer: Dense) -> None:
        pass
```

**Key Learning**: ABCs enforce consistent interfaces while allowing multiple implementations, making the codebase extensible and maintainable.

#### **2. Core Components**

**Dense Layer (`Dense.py`)**:
- Encapsulates forward/backward propagation logic
- Manages weights, biases, and gradients
- Supports multiple activation functions and weight initialization strategies

**Activation Functions (`activations.py`)**:
- **Sigmoid**: For binary classification output
- **SoftMax**: For multi-class classification
- **ReLU**: For hidden layer non-linearity
- Each function implements both forward and backward (derivative) operations

**Loss Functions (`loss_functions.py`)**:
- **Binary Cross-Entropy (BCE)**: For binary classification
- **Cross-Entropy**: For multi-class classification
- Implements both loss computation and gradient calculation

#### **3. Optimization Strategies**

**SGD (Stochastic Gradient Descent)**:
```python
def update(self, layer: Dense):
    layer.weights = layer.weights - self.lr * layer.dW
    layer.biases = layer.biases - self.lr * layer.dB
```

**Adam (Adaptive Moment Estimation)**:
- Maintains exponential moving averages of gradients and squared gradients
- Adaptive learning rates per parameter
- Bias correction for initialization

**RMSprop (Root Mean Square Propagation)**:
- Maintains moving average of squared gradients
- Adaptive learning rates based on gradient magnitude

### 🧮 **Mathematical Implementation**

#### **Forward Propagation**
```python
def forward(self, inputs: np.ndarray) -> np.ndarray:
    # Linear transformation: Z = XW + b
    self.last_input = inputs
    z = np.dot(inputs, self.weights) + self.biases
    
    # Non-linear activation: A = σ(Z)
    return self.activation.forward(z)
```

#### **Backward Propagation**
```python
def backward(self, grad_output: np.ndarray) -> np.ndarray:
    # Gradient w.r.t. activation input
    grad_activation = self.activation.backward(self.last_output) * grad_output
    
    # Gradients w.r.t. weights and biases
    self.dW = np.dot(self.last_input.T, grad_activation)
    self.dB = np.sum(grad_activation, axis=0, keepdims=True)
    
    # Gradient w.r.t. input (for previous layer)
    return np.dot(grad_activation, self.weights.T)
```

**Key Learning**: The chain rule implementation requires careful tracking of intermediate values and proper gradient flow through the network.

### 🔧 **Advanced Features**

#### **Weight Initialization Strategies**
- **Xavier/He Initialization**: Prevents vanishing/exploding gradients
- **Random Uniform/Normal**: Baseline initialization methods
- **Configurable**: Easy to experiment with different strategies

#### **Data Preprocessing Pipeline**
```python
class ZscoreScaler(IScalar):
    def fit(self, X):
        self.mean_ = np.mean(X, axis=0)
        self.std_ = np.std(X, axis=0)
        return self
    
    def transform(self, X):
        return (X - self.mean_) / self.std_
```

**Key Learning**: Proper data preprocessing (z-score normalization) is crucial for neural network training, and the scaler must be fitted only on training data to prevent data leakage.

#### **Model Persistence**
- **Complete State Saving**: Weights, biases, hyperparameters, and scaler parameters
- **Zip-based Storage**: Single file containing all model artifacts
- **Backward Compatibility**: Handles different model versions gracefully

## Repository Structure

```
./
├── src/MLP/                          # Core MLP implementation
│   ├── __init__.py
│   ├── Dense.py                      # Dense layer implementation
│   ├── activations.py                # Activation functions (Sigmoid, SoftMax, ReLU)
│   ├── loss_functions.py             # Loss functions (BCE, CrossEntropy)
│   ├── models.py                     # Main MLP class and training logic
│   ├── optimizer.py                  # Optimizers (SGD, Adam, RMSprop)
│   ├── scalar.py                     # Data preprocessing (ZscoreScaler)
│   └── weight_init.py                # Weight initialization strategies
├── configs/                          # Configuration files
│   ├── nn/                          # Neural network architectures
│   │   ├── nn_config.json           # Default 3-layer architecture
│   │   └── nn_config2.json          # Alternative architecture
│   └── mlp/                         # Experiment configurations
│       ├── exp_1_batch_size.json    # Batch size experiments
│       ├── exp_2_optimizer.json     # Optimizer comparison
│       └── exp_3_optimizer_params.json # Advanced optimizer tuning
├── data/                            # Datasets
│   ├── data_with_headers.csv        # Main training dataset
│   ├── data_train.csv               # Preprocessed training data
│   └── data_test.csv                # Preprocessed test data
├── models/                          # Trained model artifacts
│   └── trained_mlp_*.zip            # Saved models from experiments
├── trainings/                       # Training history files
│   └── history_*.csv                # Epoch-wise training metrics
├── results/                         # Experiment results
│   └── experiment_results_*.csv     # Summary of all experiments
├── images/                          # Generated visualizations
│   ├── training_curves_comparison.png
│   ├── convergence_analysis.png
│   └── pairplot_*.png               # Data analysis plots
├── train.py                         # Main training script
├── experiment.py                    # Batch experiment runner
├── predict.py                       # Model prediction script
├── analyze_training_histories.py    # Training analysis and visualization
├── data_analysis.py                 # Exploratory data analysis
├── train_test_split.py              # Data splitting utility
├── main.py                          # Legacy training script
├── Makefile                         # Build automation
├── pyproject.toml                   # Dependencies and configuration
└── README.md                        # This file
```

## Quick Start Guide

### 🚀 **Installation**

#### Option A — uv (recommended)
```bash
# 1) Install uv if needed (macOS/Linux)
curl -LsSf https://astral.sh/uv/install.sh | sh

# 2) Create virtual environment and install dependencies
uv venv
uv sync
```

#### Option B — pip
```bash
python -m venv .venv
source .venv/bin/activate  # macOS/Linux
pip install -r requirements.txt
```

### 📊 **Basic Usage**

#### **1. Single Model Training**
```bash
# Train with default parameters
uv run python train.py --dataset data/data_with_headers.csv

# Train with custom hyperparameters
uv run python train.py --dataset data/data_with_headers.csv \
    --optimizer adam \
    --lr 0.001 \
    --batch_size 32 \
    --epoch 500

# Train with custom optimizer parameters
uv run python train.py --dataset data/data_with_headers.csv \
    --optimizer adam \
    --beta1 0.9 \
    --beta2 0.999 \
    --epsilon 1e-8
```

#### **2. Batch Experiments**
```bash
# Run batch size experiments
uv run python experiment.py --exp_config configs/mlp/exp_1_batch_size.json

# Run optimizer comparison
uv run python experiment.py --exp_config configs/mlp/exp_2_optimizer.json

# Run advanced optimizer tuning
uv run python experiment.py --exp_config configs/mlp/exp_3_optimizer_params.json
```

#### **3. Model Prediction**
```bash
# Predict on new data
uv run python predict.py --model_path models/trained_mlp.zip \
    --data_path data/data_test.csv \
    --output_path predictions.csv
```

#### **4. Training Analysis**
```bash
# Analyze training histories
uv run python analyze_training_histories.py \
    --trainings_dir trainings \
    --pattern "history_batch_*" \
    --output_dir images

# Compare all experiments
uv run python analyze_training_histories.py \
    --trainings_dir trainings \
    --pattern "history_*" \
    --show
```

### 🔧 **Configuration**

#### **Neural Network Architecture**
Edit `configs/nn/nn_config.json`:
```json
[
    {
        "nodes": 64,
        "activation": "relu",
        "weights_init": "he_uniform"
    },
    {
        "nodes": 32,
        "activation": "relu", 
        "weights_init": "he_uniform"
    },
    {
        "nodes": 2,
        "activation": "softmax",
        "weights_init": "xavier_uniform"
    }
]
```

#### **Experiment Configuration**
Edit `configs/mlp/exp_2_optimizer.json`:
```json
{
    "opt_sgd_lr_0.01": {
        "epoch": 1000,
        "lr": 0.01,
        "optimizer": "sgd",
        "batch_size": 16
    },
    "opt_adam_lr_0.001": {
        "epoch": 1000,
        "lr": 0.001,
        "optimizer": "adam",
        "beta1": 0.9,
        "beta2": 0.999,
        "epsilon": 1e-8
    }
}
```

## Key Features & Capabilities

### 🧠 **Neural Network Components**
- **Dense Layers**: Fully connected layers with configurable activation functions
- **Activation Functions**: Sigmoid, SoftMax, ReLU with proper derivatives
- **Loss Functions**: Binary Cross-Entropy, Cross-Entropy for classification
- **Weight Initialization**: Xavier, He, Random strategies

### ⚡ **Optimization Algorithms**
- **SGD**: Basic stochastic gradient descent
- **Adam**: Adaptive moment estimation with bias correction
- **RMSprop**: Root mean square propagation
- **Configurable Parameters**: Learning rates, momentum, decay rates

### 📈 **Training Features**
- **Early Stopping**: Prevents overfitting based on validation loss
- **Batch Training**: Configurable batch sizes for memory efficiency
- **Progress Monitoring**: Real-time loss and accuracy tracking
- **History Tracking**: Complete training metrics saved to CSV

### 🔬 **Experiment Framework**
- **Batch Experiments**: Systematic hyperparameter tuning
- **Configuration-Driven**: JSON-based experiment definitions
- **Results Analysis**: Automated performance comparison
- **Visualization**: Training curves and convergence analysis

### 💾 **Model Management**
- **Complete Persistence**: Save/load weights, biases, hyperparameters, scaler
- **Zip-based Storage**: Single file containing all model artifacts
- **Backward Compatibility**: Handles different model versions
- **Prediction Pipeline**: Load model and make predictions on new data

## Performance Results

### 🏆 **Best Achieved Performance**
- **Accuracy**: 99.12% on test set
- **Optimizer**: SGD with learning rate 0.01
- **Architecture**: 3-layer MLP (30 → 64 → 32 → 2)
- **Training Time**: ~50 epochs with early stopping

### 📊 **Experiment Insights**
- **Batch Size**: Smaller batches (2-8) converge faster but may be less stable
- **Optimizers**: Adam and RMSprop often converge faster than SGD
- **Learning Rates**: 0.001-0.01 range works well for this dataset
- **Architecture**: 2-3 hidden layers with 32-64 neurons per layer

## References & Reflection

### 📚 **Key Learning Resources**
- [MLP explained with visual](https://medium.com/data-science/multilayer-perceptron-explained-a-visual-guide-with-mini-2d-dataset-0ae8100c5d1c)
- [Softmax and its derivative](https://medium.com/data-science/derivative-of-the-softmax-function-and-the-categorical-cross-entropy-loss-ffceefc081d1)
- [Weights Initialization](https://www.geeksforgeeks.org/machine-learning/weight-initialization-techniques-for-deep-neural-networks/)
- [Optimizers](https://www.geeksforgeeks.org/deep-learning/adam-optimizer/)

### 🤔 **Reflection on Learning Journey**

#### **What Worked Well**
1. **Modular Design**: Breaking down the neural network into components made the code maintainable and testable
2. **Mathematical Understanding**: Implementing from scratch provided deep insight into how neural networks actually work
3. **Experiment Framework**: Systematic hyperparameter tuning revealed important insights about model behavior
4. **Visualization**: Plotting training dynamics helped identify overfitting and convergence issues

#### **Challenges Overcome**
1. **Gradient Flow**: Initially struggled with proper gradient computation in backpropagation
2. **Numerical Stability**: SoftMax and loss functions required careful handling of numerical precision
3. **Data Preprocessing**: Learning the importance of proper scaling and avoiding data leakage
4. **Hyperparameter Tuning**: Understanding the complex interactions between learning rate, batch size, and optimizer choice

#### **Key Insights**
1. **Weight Initialization Matters**: Proper initialization prevents vanishing/exploding gradients
2. **Data Preprocessing is Critical**: Z-score normalization significantly improved training stability
3. **Early Stopping is Essential**: Prevents overfitting and saves computational resources
4. **Optimizer Choice Affects Convergence**: Different optimizers have different convergence characteristics

#### **Future Improvements**
1. **Regularization**: Add L1/L2 regularization and dropout
2. **Advanced Architectures**: Implement convolutional and recurrent layers
3. **Hyperparameter Optimization**: Add Bayesian optimization for parameter tuning
4. **Distributed Training**: Support for multi-GPU training
5. **Model Compression**: Add quantization and pruning capabilities

### 🎯 **Educational Value**
This project successfully demonstrates:
- **Core ML Concepts**: Forward/backward propagation, gradient descent, activation functions
- **Software Engineering**: Modular design, testing, documentation, configuration management
- **Experimental Design**: Systematic hyperparameter tuning, performance evaluation
- **Data Science Pipeline**: Data preprocessing, model training, evaluation, visualization

The implementation serves as a solid foundation for understanding neural networks and can be extended for more complex architectures and applications.

---

## Code Quality & Standards

This project uses **Ruff** for linting and formatting:
```bash
uv run ruff check .
uv run ruff format .
```

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- **42 School** for providing the learning framework and project structure
- **Breast Cancer Wisconsin Dataset** for the real-world classification problem
- **Open Source Community** for the mathematical foundations and implementation references
