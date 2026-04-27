# LSTM Mathematics Documentation

This document provides a comprehensive mathematical explanation of Long Short-Term Memory (LSTM) networks, including the equations for all gates, cell state updates, and bidirectional processing used in the LSTM Text Prediction System.

## Table of Contents

1. [Introduction](#introduction)
2. [LSTM Cell Architecture](#lstm-cell-architecture)
3. [LSTM Equations](#lstm-equations)
4. [Gate Operations](#gate-operations)
5. [Bidirectional LSTM](#bidirectional-lstm)
6. [Model Architecture](#model-architecture)
7. [Training Process](#training-process)

## Introduction

Long Short-Term Memory (LSTM) networks are a type of Recurrent Neural Network (RNN) designed to address the vanishing gradient problem in traditional RNNs. LSTMs can learn long-term dependencies in sequential data through a sophisticated gating mechanism that controls information flow.

### Key Components

- **Cell State (C)**: Long-term memory that carries information across time steps
- **Hidden State (h)**: Short-term memory that represents the current output
- **Forget Gate (f)**: Controls what information to discard from cell state
- **Input Gate (i)**: Controls what new information to add to cell state
- **Output Gate (o)**: Controls what information to output from cell state

## LSTM Cell Architecture

```
                    ┌─────────────────────────────────────┐
                    │         LSTM Cell at time t         │
                    │                                     │
    C(t-1) ────────►│  ┌──────┐  ┌──────┐  ┌──────┐    │────────► C(t)
                    │  │Forget│  │Input │  │Output│    │
    h(t-1) ────────►│  │Gate  │  │Gate  │  │Gate  │    │────────► h(t)
                    │  └──────┘  └──────┘  └──────┘    │
    x(t)   ────────►│      │         │         │        │
                    │      ▼         ▼         ▼        │
                    │  ┌─────────────────────────┐      │
                    │  │   Cell State Update     │      │
                    │  └─────────────────────────┘      │
                    └─────────────────────────────────────┘
```

## LSTM Equations

### Notation

- $x_t$ : Input vector at time step $t$
- $h_t$ : Hidden state at time step $t$
- $C_t$ : Cell state at time step $t$
- $W$ : Weight matrices
- $b$ : Bias vectors
- $\sigma$ : Sigmoid activation function
- $\tanh$ : Hyperbolic tangent activation function
- $\odot$ : Element-wise multiplication (Hadamard product)

### 1. Forget Gate

The forget gate decides what information to discard from the cell state.

$$f_t = \sigma(W_f \cdot [h_{t-1}, x_t] + b_f)$$

Where:
- $f_t \in [0, 1]^n$ : Forget gate activation vector
- $W_f$ : Weight matrix for forget gate
- $b_f$ : Bias vector for forget gate
- $\sigma(x) = \frac{1}{1 + e^{-x}}$ : Sigmoid function

**Interpretation**: 
- $f_t = 1$ : Keep all information from previous cell state
- $f_t = 0$ : Discard all information from previous cell state

### 2. Input Gate

The input gate decides what new information to add to the cell state.

$$i_t = \sigma(W_i \cdot [h_{t-1}, x_t] + b_i)$$

$$\tilde{C}_t = \tanh(W_C \cdot [h_{t-1}, x_t] + b_C)$$

Where:
- $i_t \in [0, 1]^n$ : Input gate activation vector
- $\tilde{C}_t \in [-1, 1]^n$ : Candidate cell state values
- $W_i, W_C$ : Weight matrices for input gate and candidate values
- $b_i, b_C$ : Bias vectors for input gate and candidate values
- $\tanh(x) = \frac{e^x - e^{-x}}{e^x + e^{-x}}$ : Hyperbolic tangent function

**Interpretation**:
- $i_t$ : Controls which candidate values to add
- $\tilde{C}_t$ : New candidate information to potentially add to cell state

### 3. Cell State Update

The cell state is updated by combining the forget gate and input gate operations.

$$C_t = f_t \odot C_{t-1} + i_t \odot \tilde{C}_t$$

Where:
- $C_t$ : Updated cell state at time $t$
- $\odot$ : Element-wise multiplication

**Interpretation**:
- $f_t \odot C_{t-1}$ : Selectively forget information from previous cell state
- $i_t \odot \tilde{C}_t$ : Selectively add new candidate information
- The sum combines both operations to update the cell state

### 4. Output Gate

The output gate decides what information to output from the cell state.

$$o_t = \sigma(W_o \cdot [h_{t-1}, x_t] + b_o)$$

$$h_t = o_t \odot \tanh(C_t)$$

Where:
- $o_t \in [0, 1]^n$ : Output gate activation vector
- $h_t$ : Hidden state (output) at time $t$
- $W_o$ : Weight matrix for output gate
- $b_o$ : Bias vector for output gate

**Interpretation**:
- $o_t$ : Controls which parts of cell state to output
- $\tanh(C_t)$ : Squashes cell state to $[-1, 1]$ range
- $h_t$ : Final output combining cell state and output gate

## Gate Operations

### Sigmoid Activation Function

$$\sigma(x) = \frac{1}{1 + e^{-x}}$$

**Properties**:
- Range: $(0, 1)$
- Used for gates because output represents "how much" to let through
- $\sigma(0) = 0.5$ : Neutral point
- $\sigma(x) \to 1$ as $x \to \infty$ : Fully open gate
- $\sigma(x) \to 0$ as $x \to -\infty$ : Fully closed gate

### Hyperbolic Tangent Activation Function

$$\tanh(x) = \frac{e^x - e^{-x}}{e^x + e^{-x}} = \frac{e^{2x} - 1}{e^{2x} + 1}$$

**Properties**:
- Range: $(-1, 1)$
- Used for cell state and candidate values
- $\tanh(0) = 0$ : Neutral point
- $\tanh(x) \to 1$ as $x \to \infty$
- $\tanh(x) \to -1$ as $x \to -\infty$
- Centered around zero (unlike sigmoid)

### Element-wise Multiplication (Hadamard Product)

$$(A \odot B)_{ij} = A_{ij} \cdot B_{ij}$$

**Properties**:
- Multiplies corresponding elements
- Used to apply gate activations to vectors
- Allows selective information flow

## Bidirectional LSTM

Bidirectional LSTMs process sequences in both forward and backward directions, capturing context from both past and future.

### Forward LSTM

Processes sequence from left to right (time step 1 to T):

$$\overrightarrow{h}_t = \text{LSTM}_{forward}(x_t, \overrightarrow{h}_{t-1}, \overrightarrow{C}_{t-1})$$

### Backward LSTM

Processes sequence from right to left (time step T to 1):

$$\overleftarrow{h}_t = \text{LSTM}_{backward}(x_t, \overleftarrow{h}_{t+1}, \overleftarrow{C}_{t+1})$$

### Concatenation

The outputs from both directions are concatenated:

$$h_t = [\overrightarrow{h}_t; \overleftarrow{h}_t]$$

Where:
- $\overrightarrow{h}_t \in \mathbb{R}^n$ : Forward hidden state
- $\overleftarrow{h}_t \in \mathbb{R}^n$ : Backward hidden state
- $h_t \in \mathbb{R}^{2n}$ : Concatenated bidirectional hidden state

**Advantages**:
- Captures context from both past and future
- Improves prediction accuracy for sequence tasks
- Doubles the hidden state dimension

## Model Architecture

### Complete Architecture

Our LSTM Text Prediction System uses the following architecture:

```
Input Sequence (50 tokens)
         │
         ▼
┌─────────────────────┐
│  Embedding Layer    │  Vocabulary: 10,000 → Embedding: 256
└─────────────────────┘
         │
         ▼
┌─────────────────────┐
│ Bidirectional LSTM  │  Units: 512 (256 forward + 256 backward)
│  return_sequences   │  Output: (batch, 50, 512)
└─────────────────────┘
         │
         ▼
┌─────────────────────┐
│   Dropout (0.3)     │  Regularization
└─────────────────────┘
         │
         ▼
┌─────────────────────┐
│ Unidirectional LSTM │  Units: 256
│ return_sequences=F  │  Output: (batch, 256)
└─────────────────────┘
         │
         ▼
┌─────────────────────┐
│   Dropout (0.3)     │  Regularization
└─────────────────────┘
         │
         ▼
┌─────────────────────┐
│   Dense Layer       │  Units: 10,000 (vocabulary size)
│   Softmax           │  Output: Probability distribution
└─────────────────────┘
         │
         ▼
    Prediction
```

### Layer-by-Layer Equations

#### 1. Embedding Layer

$$E_t = W_e[x_t]$$

Where:
- $x_t \in \{1, 2, ..., 10000\}$ : Token index at position $t$
- $W_e \in \mathbb{R}^{10000 \times 256}$ : Embedding weight matrix
- $E_t \in \mathbb{R}^{256}$ : Embedded representation

#### 2. Bidirectional LSTM Layer

$$\overrightarrow{h}_t = \text{LSTM}_{forward}(E_t, \overrightarrow{h}_{t-1}, \overrightarrow{C}_{t-1})$$

$$\overleftarrow{h}_t = \text{LSTM}_{backward}(E_t, \overleftarrow{h}_{t+1}, \overleftarrow{C}_{t+1})$$

$$h_t^{(1)} = [\overrightarrow{h}_t; \overleftarrow{h}_t] \in \mathbb{R}^{512}$$

#### 3. Dropout Layer

$$h_t^{(1')} = \text{Dropout}(h_t^{(1)}, p=0.3)$$

Where dropout randomly sets 30% of activations to zero during training.

#### 4. Unidirectional LSTM Layer

$$h^{(2)} = \text{LSTM}_{unidirectional}(h_{50}^{(1')}, h^{(2)}_{prev}, C^{(2)}_{prev})$$

Note: Only the final time step output is used (return_sequences=False).

#### 5. Dropout Layer

$$h^{(2')} = \text{Dropout}(h^{(2)}, p=0.3)$$

#### 6. Dense Output Layer with Softmax

$$z = W_{out} \cdot h^{(2')} + b_{out}$$

$$p(w_i | x_{1:50}) = \frac{e^{z_i}}{\sum_{j=1}^{10000} e^{z_j}}$$

Where:
- $W_{out} \in \mathbb{R}^{10000 \times 256}$ : Output weight matrix
- $b_{out} \in \mathbb{R}^{10000}$ : Output bias vector
- $p(w_i | x_{1:50})$ : Probability of word $i$ given input sequence

## Training Process

### Loss Function

We use categorical cross-entropy loss:

$$L = -\frac{1}{N} \sum_{i=1}^{N} \sum_{j=1}^{V} y_{ij} \log(\hat{y}_{ij})$$

Where:
- $N$ : Number of training samples
- $V$ : Vocabulary size (10,000)
- $y_{ij}$ : True label (one-hot encoded)
- $\hat{y}_{ij}$ : Predicted probability

### Perplexity Metric

Perplexity measures how well the model predicts the next word:

$$\text{Perplexity} = e^{L} = \exp\left(-\frac{1}{N} \sum_{i=1}^{N} \sum_{j=1}^{V} y_{ij} \log(\hat{y}_{ij})\right)$$

**Interpretation**:
- Lower perplexity = Better predictions
- Perplexity of $k$ means the model is as uncertain as if it had to choose uniformly from $k$ words

### Optimization

We use the Adam optimizer with learning rate $\alpha = 0.001$:

$$m_t = \beta_1 m_{t-1} + (1 - \beta_1) g_t$$

$$v_t = \beta_2 v_{t-1} + (1 - \beta_2) g_t^2$$

$$\hat{m}_t = \frac{m_t}{1 - \beta_1^t}$$

$$\hat{v}_t = \frac{v_t}{1 - \beta_2^t}$$

$$\theta_t = \theta_{t-1} - \alpha \frac{\hat{m}_t}{\sqrt{\hat{v}_t} + \epsilon}$$

Where:
- $g_t$ : Gradient at time $t$
- $m_t$ : First moment estimate (mean)
- $v_t$ : Second moment estimate (variance)
- $\beta_1 = 0.9$ : Exponential decay rate for first moment
- $\beta_2 = 0.999$ : Exponential decay rate for second moment
- $\epsilon = 10^{-7}$ : Small constant for numerical stability

### Backpropagation Through Time (BPTT)

Gradients are computed using backpropagation through time:

$$\frac{\partial L}{\partial W} = \sum_{t=1}^{T} \frac{\partial L_t}{\partial W}$$

Where gradients are accumulated across all time steps in the sequence.

### Gradient Clipping

To prevent exploding gradients, we clip gradients by norm:

$$g = \begin{cases}
g & \text{if } ||g|| \leq \theta \\
\theta \frac{g}{||g||} & \text{if } ||g|| > \theta
\end{cases}$$

Where $\theta$ is the clipping threshold (typically 5.0).

## Temperature Sampling

During inference, we use temperature sampling to control prediction diversity:

$$p_i = \frac{e^{z_i / T}}{\sum_{j=1}^{V} e^{z_j / T}}$$

Where:
- $T$ : Temperature parameter
- $T < 1$ : More deterministic (sharper distribution)
- $T = 1$ : Original distribution
- $T > 1$ : More random (flatter distribution)

**Effect on Distribution**:

```
Temperature = 0.5 (Deterministic)
  ████████████████████ word1 (0.85)
  ███ word2 (0.10)
  █ word3 (0.03)

Temperature = 1.0 (Balanced)
  ████████████ word1 (0.60)
  ██████ word2 (0.25)
  ███ word3 (0.15)

Temperature = 2.0 (Creative)
  ██████ word1 (0.35)
  █████ word2 (0.30)
  █████ word3 (0.25)
```

## Mathematical Properties

### Vanishing Gradient Solution

LSTMs solve the vanishing gradient problem through the cell state:

$$\frac{\partial C_t}{\partial C_{t-1}} = f_t$$

Since $f_t \in [0, 1]$, gradients can flow through time without vanishing (when $f_t \approx 1$).

### Long-Term Dependencies

The cell state acts as a "highway" for information:

$$C_t = f_t \odot C_{t-1} + i_t \odot \tilde{C}_t$$

Information can be preserved across many time steps if $f_t \approx 1$ and $i_t \approx 0$.

### Gating Mechanism

Gates use sigmoid activations to create "soft" switches:

- $\sigma(x) \approx 0$ : Gate closed (block information)
- $\sigma(x) \approx 0.5$ : Gate partially open
- $\sigma(x) \approx 1$ : Gate open (allow information)

## References

1. Hochreiter, S., & Schmidhuber, J. (1997). Long Short-Term Memory. Neural Computation, 9(8), 1735-1780.

2. Graves, A., & Schmidhuber, J. (2005). Framewise phoneme classification with bidirectional LSTM and other neural network architectures. Neural Networks, 18(5-6), 602-610.

3. Goodfellow, I., Bengio, Y., & Courville, A. (2016). Deep Learning. MIT Press. Chapter 10: Sequence Modeling: Recurrent and Recursive Nets.

4. Olah, C. (2015). Understanding LSTM Networks. https://colah.github.io/posts/2015-08-Understanding-LSTMs/

5. Kingma, D. P., & Ba, J. (2014). Adam: A Method for Stochastic Optimization. arXiv preprint arXiv:1412.6980.

---

**Note**: This document uses LaTeX notation for mathematical equations. For best viewing, use a Markdown renderer that supports LaTeX/MathJax, such as GitHub, Jupyter Notebook, or VS Code with appropriate extensions.
