# Sentiment Analysis with Custom Transformer

A PyTorch implementation of a custom transformer-based sentiment analysis model using BERT embeddings.

## Overview

This project implements a sentiment analysis model that combines pre-trained BERT embeddings with a custom transformer architecture. The model uses multi-head attention mechanisms and feed-forward networks to classify text sentiment as positive or negative.

## Features

- **Custom Transformer Architecture**: Two-layer transformer with multi-head attention
- **BERT Integration**: Uses pre-trained BERT embeddings as input features
- **Masked Attention**: Properly handles variable-length sequences with attention masking
- **Memory Efficient**: Optimized for GPU memory usage with proper garbage collection
- **Configurable**: Easy to adjust hyperparameters and model architecture

## Architecture

The model consists of:

1. **BERT Embedder**: Frozen BERT-base-uncased for generating contextual embeddings
2. **Multi-Head Attention**: Custom implementation with 8 attention heads
3. **Feed-Forward Networks**: Position-wise feed-forward layers with ReLU activation
4. **Layer Normalization**: Applied after each sub-layer with residual connections
5. **Dropout**: Regularization to prevent overfitting
6. **Classification Head**: Multi-layer perceptron for binary classification

## Requirements

```
torch>=1.9.0
transformers>=4.0.0
datasets>=2.0.0
numpy>=1.21.0
```

## Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd sentiment-analysis
```

2. Install dependencies:

```
As seen in the notebook.
```

## Configuration

### Model Parameters

- `input_dim`: BERT embedding dimension (default: 768)
- `hidden_dim`: Feed-forward network hidden dimension (default: 512)
- `output_dim`: Number of output classes (default: 1 for binary classification)
- `num_heads`: Number of attention heads (default: 8)

### Training Parameters

- `batch_size`: Training batch size (default: 32)
- `max_length`: Maximum sequence length (default: 512)
- `learning_rate`: AdamW learning rate (default: 2e-5)
- `weight_decay`: L2 regularization (default: 1e-4)

### Data Parameters

- `max_length`: Maximum token length for BERT tokenization (default: 256)
- `batch_size`: Number of samples per batch (default: 32)

## Training Process

1. **Data Loading**: BERT tokenization and embedding generation
2. **Forward Pass**: Multi-head attention and feed-forward processing
3. **Pooling**: Masked average pooling over sequence length
4. **Classification**: Binary classification with BCEWithLogitsLoss
5. **Optimization**: AdamW optimizer with gradient clipping

## Performance Optimization

- **Memory Management**: Embeddings moved to CPU after generation
- **Gradient Clipping**: Prevents exploding gradients (max norm: 1.0)
- **Frozen BERT**: BERT parameters frozen to save memory and computation
- **Efficient Batching**: Generator-based data loading for memory efficiency

## Dataset Support

Currently configured for IMDB dataset, but can be adapted for any binary sentiment classification dataset:

```python
# For custom datasets, modify the Databuilder class
self.dataset = load_dataset('your_dataset_name')
```

## Troubleshooting

### Common Issues

1. **CUDA Out of Memory**: Reduce batch_size or max_length
2. **Slow Training**: Ensure CUDA is available and being used
3. **Poor Convergence**: Check learning rate and gradient clipping
4. **NaN Values**: Model handles NaN in attention weights automatically

## License

MIT License - see LICENSE file for details
