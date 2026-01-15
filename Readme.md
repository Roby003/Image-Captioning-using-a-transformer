# Transformer-Based Image Captioning

## Overview

This project implements a decoder-only transformer architecture for image captioning on the MS COCO dataset, benchmarked against the CNN-based convolutional captioning methods from Aneja et al. (CVPR 2018).

## Architecture

### Model: ImageCaptioningTransformer

**Core Design:** Decoder-only transformer with causal masking, treating image captioning as an autoregressive sequence generation task where visual features serve as a learned prefix to the text sequence.

### Architecture Components

1. **Image Embedding**
    
    - Input: VGG16 fc7 features (4096-dimensional) / ResNet101 (2048 - dimensional)
    - Projection: Linear layer to 512 dimensions
    - Purpose: Maps pre-extracted visual features into the model's embedding space
2. **Word Embeddings**
    - Vocabulary size: 9,221 words (including special tokens: `<START>`, `<END>`, `<PAD>`, `<UNK>`) - matching paper
    - Embedding dimension: 512
    - Learned from scratch during training
3. **Positional Embeddings**
    - Type: Learned positional embeddings (not sinusoidal)
    - Maximum sequence length: 15 words
    - Dimension: 512
4. **Transformer Decoder**
    - Layers: 4 transformer encoder layers repurposed as decoder with causal masking
    - Hidden dimension: 512
    - Feedforward dimension: 2048
    - Attention heads: 8
    - Dropout: 0.1
    - Total parameters: ~15-20M (comparable to paper's baseline)
5. **Output Layer**
    - Linear projection: 512 → 9,221 (vocabulary size)
    - Softmax activation for word probability distribution

### Key Design Decisions

**Why decoder-only?** Simpler than encoder-decoder architecture for this task. The image features are treated as a prefix, and the model autoregressively generates the caption using causal masking to prevent attending to future tokens.


## Dataset

- **Source:** MS COCO 2014
- **Split:** Karpathy split (standard in image captioning research)
    - Training: 113,287 images
    - Validation: 5,000 images
    - Test: 5,000 images
- **Image Features:** Pre-extracted VGG16 fc7 features (4096-dim)
- **Captions:** 5 reference captions per image

## Training Configuration

### Optimization

- **Optimizer:** RMSprop
- **Learning Rate:** 5e-5 (matching paper)
- **Scheduler:** CosineAnnealingLR
- **Epochs:** 70
- **Loss Function:** CrossEntropyLoss
- **Training Strategy:** Teacher forcing (ground truth previous words during training)

### Training Progress

- Initial loss: ~8.0
- Final loss: ~2.20 (after 30 epochs)
- Training conducted on standard Karpathy split

## Inference

#### Current token selection method : Greedy

## Evaluation Metrics

The model is evaluated using seven standard image captioning metrics:

- **BLEU-1, BLEU-2, BLEU-3, BLEU-4:** N-gram precision scores
- **METEOR:** Alignment-based metric considering synonyms and stemming
- **ROUGE:** Recall-oriented metric based on longest common subsequence
- **CIDEr:** Consensus-based metric (primary metric for comparison)
- **SPICE:** Semantic similarity based on scene graphs

**Baseline Target:** Aneja et al. achieve CIDEr score of ~0.881 with CNN baseline
## Results


Our transformer-based approach achieves competitive results compared to the CNN baseline from Aneja et al. (CVPR 2018).

#### VGG16 Features

|Metric|Our Transformer|Aneja et al. CNN Baseline|Difference|
|---|---|---|---|
|BLEU-1|**0.688**|0.695|-0.007|
|BLEU-2|**0.511**|0.521|-0.010|
|BLEU-3|**0.369**|0.380|-0.011|
|BLEU-4|**0.266**|0.276|-0.010|
|METEOR|**0.234**|0.241|-0.007|
|ROUGE-L|**0.505**|0.514|-0.009|
|**CIDEr**|**0.854**|**0.881**|**-0.027**|
|SPICE|**0.164**|0.171|-0.007|

#### ResNet-101 Features

|Metric|Score|
|---|---|
|BLEU-1|0.698|
|BLEU-2|0.522|
|BLEU-3|0.378|
|BLEU-4|0.273|
|METEOR|0.238|
|ROUGE-L|0.511|
|**CIDEr**|**0.883**|
|SPICE|0.167|
## Future Work

### 1. Beam Search Implementation

**Current Limitation:** The model uses greedy decoding (argmax), which selects only the single most probable word at each step. This can lead to suboptimal captions because locally optimal choices may not lead to globally optimal sequences.

**Planned Enhancement: Beam Search**