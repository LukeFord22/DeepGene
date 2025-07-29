# DeepSNP3D: Gene Detection and Classification System

A high-performance deep learning system for detecting and classifying genes in DNA sequences using DNABERT-2 transformer architecture with DeepSpeed optimization. The system processes raw gene sequences into tokenized datasets with realistic NGS noise patterns for robust gene detection.

## 🧬 Overview

DeepSNP3D leverages the state-of-the-art DNABERT-2 model (117M parameters) for gene detection and classification in genomic data. The system performs:

### Primary Tasks
- **Gene Classification**: Multi-class classification of gene sequences from NGS reads
- **Gene Detection**: Identification of gene sequences in noisy sequencing data
- **Sequence Processing**: Advanced tokenization with attention masks and realistic noise simulation

### Data Processing Pipeline
The system includes a comprehensive DNA sequence processing pipeline that:
- **Generates Realistic NGS Noise**: Simulates Illumina, PacBio, and ONT sequencing errors
- **Creates Consistent Gene Mappings**: Maintains persistent gene numbering across runs
- **Produces Balanced Datasets**: Intelligently augments underrepresented genes
- **Handles Variable-Length Sequences**: Dynamic windowing with attention masks

### Advanced Features
- **Transformer Architecture**: DNABERT-2 pre-trained on genomic data with enhanced attention mechanisms
- **Gene Classification**: Multi-class classification with consistent gene numbering system
- **NGS Noise Simulation**: Technology-specific error patterns for robust training
- **Multi-GPU Training**: DeepSpeed integration for distributed training with ZeRO optimization
- **Memory Optimization**: ZeRO optimizer with CPU offloading and gradient checkpointing
- **Enhanced Attention**: Multi-scale attention with rotary embeddings and gene-aware modulation
- **Temperature Scaling**: Learnable temperature parameters for all prediction heads
- **Token-Level Processing**: Fine-grained gene detection at nucleotide resolution
- **Curriculum Learning**: Progressive training from balanced to natural gene distributions
- **Data Augmentation**: Realistic sequencing noise and overlapping window generation

## 🚀 Quick Start

### Prerequisites
- Python 3.8+
- CUDA-compatible GPU(s)
- pytorch:2.8.0-py3.11-cuda12.8.1-cudnn-devel-ubuntu22.04

### DNA Sequence Processing Pipeline
Before training, you need to process your gene sequences using the DNA Processing Pipeline:

1. **Navigate to the DNA Processing directory**:
   ```bash
   cd ../Genes/DNAprocessing
   ```

2. **Configure your gene data**:
   Edit `config.json` to point to your gene sequence CSV file:
   ```json
   {
     "input": {
       "gene_data_path": "path/to/your/gene_sequences.csv"
     }
   }
   ```

3. **Run the processing pipeline**:
   ```bash
   python dna_sequence_processor.py
   ```

4. **Copy processed data to DeepSNP3D**:
   ```bash
   cp output/tokenized_dataset.pt ../DeepSNP3D/BERTModel/data/
   cp output/label_mapping.json ../DeepSNP3D/BERTModel/data/
   ```

### Installation
1. Clone the repository:
   ```bash
   git clone <repository-url>
   cd DeepSNP3D
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Interact with Git:
   ```bash
   git config --global user.name "LukeFord22"
   git config --global user.email "lhford1213@gmail.com"
   cd DeepGene
   git add .
   git commit -m "Update: DeepSNP DNA SNP detection system with DNABERT-2 and DeepSpeed"
   git push -u origin main

   # remove all staged files and start over
   git reset --soft HEAD~2
   git reset HEAD .
   ```

4. Send Data:
```Bash
runpodctl config --apiKey rpa_SOO146XCJMOTK6HOLXTJ9WLU7YPG2I3V17BVNXBFxhdz9x
runpodctl send Gene.pth
./runpodctl.exe receive
```


### Training
Navigate to the BERTModel/scripts directory and use the auto-launcher:
```bash
cd BERTModel/scripts
python launch_training.py
```

The launcher automatically:
- Detects your GPU configuration
- Chooses optimal training settings for gene classification
- Handles distributed setup
- Manages DeepSpeed configuration
- Loads processed gene data with attention masks

## 🏗️ Architecture

### Model Components
- **Base Model**: DNABERT-2-117M (zhihan1996/DNABERT-2-117M)
- **Gene Classification Architecture**: Multi-class classification for gene detection
- **Enhanced Attention**: 
  - **Token Attention**: Multi-scale windowed attention with rotary embeddings
  - **DNA Attention**: Gene-aware sequence-level attention with landmark-based pooling  
- **Classification Head**: Dedicated processing pipeline for gene classification
- **Temperature Scaling**: Learnable temperature parameters for calibrated predictions
- **Residual Processing**: Deep residual blocks with layer normalization and dropout
- **Attention Mask Integration**: Proper handling of variable-length sequences

### Gene Classification Architecture
The system uses a shared DNABERT-2 encoder with specialized gene detection pipeline:

1. **Gene Detection Pipeline**:
   - Token-level gene classification with attention masks
   - DNA-level sequence classification with gene-aware attention
   - Multi-class cross-entropy loss with temperature scaling
   - Consistent gene numbering across runs

2. **Data Processing Integration**:
   - Realistic NGS noise simulation for robust training
   - Dynamic windowing for variable-length sequences
   - Balanced dataset generation for underrepresented genes
   - Persistent gene mapping system

3. **Attention Mechanisms**:
   - **Multi-Scale Token Attention**: Motif (16bp), regulatory (64bp), and domain (256bp) windows
   - **Rotary Position Embeddings**: Critical for gene position sensitivity
   - **Gene-Aware Modulation**: Learnable attention gating based on gene predictions
   - **Landmark Attention**: Efficient processing of long sequences

### Enhanced Data Loading
- **Optimized PT Dataset**: 10-50x faster loading using pre-tokenized .pt files from DNA processing pipeline
- **Attention Mask Support**: Proper handling of variable-length sequences with attention masks
- **Gene Classification Data**: Multi-class labels with consistent gene numbering
- **Memory Efficient Collation**: Dense tensors for tokenized sequences and labels


### Enhanced Training Features
- **Optimized Data Loading**: 10-50x faster loading with pre-tokenized .pt files
- **Sparse Feature Processing**: Memory-efficient coordinate-based epigenetic data handling
- **Multi-Scale Attention**: Biologically-motivated attention windows (motif/regulatory/domain)
- **Advanced Loss Functions**: Focal loss, consistency regularization, adaptive weighting
- **Temperature Scaling**: Individual learnable temperatures for all 12+ prediction heads
- **Curriculum Learning**: Progressive transition from balanced to natural class distributions
- **Data Augmentation**: Overlapping window generation for improved robustness
- **Flash Attention Compatibility**: Automatic fallback with comprehensive patches
- **Mixed Precision**: FP16 training with automatic loss scaling and gradient clipping

## 📊 Performance Improvements

### DeepSpeed Optimizations
- **50% Memory Reduction**: ZeRO stage 2 with optimizer state partitioning
- **Multi-GPU Scaling**: Automatic data parallelism across available GPUs  
- **CPU Offloading**: Optimizer states offloaded to CPU memory
- **Mixed Precision**: FP16 training without manual AMP management
- **Gradient Optimization**: Automatic accumulation and clipping (max norm: 5.0)
- **Communication Efficiency**: Overlap computation with communication

### Training Enhancements
- **Adaptive Batch Sizing**: Automatic batch size based on GPU memory
- **Smart Checkpointing**: Regular model state saves with resume capability
- **Robust Error Handling**: Automatic fallback strategies for training failures
- **Real-time Monitoring**: Comprehensive Wandb integration for experiment tracking

## 📁 Project Structure

```
DeepSNP3D/
├── README.md                          # Project documentation
├── requirements.txt                   # Python dependencies
├── LICENSE                           # Project license
├── ../Genes/DNAprocessing/           # DNA sequence processing pipeline
│   ├── dna_sequence_processor.py     # Main processing pipeline with NGS noise simulation
│   ├── config.json                   # Processing configuration
│   ├── README.md                     # Processing pipeline documentation
│   └── output/                       # Processed data outputs
│       ├── tokenized_dataset.pt      # Tokenized gene sequences with attention masks
│       ├── label_mapping.json        # Gene-to-label mappings
│       └── gene_mapping.json         # Persistent gene numbering system
└── BERTModel/                        # Main training directory
    ├── scripts/
    │   └── launch_training.py        # Auto-launcher with GPU detection
    ├── configs/
    │   ├── config.py                 # Gene classification training configuration
    │   └── deepspeed_config.py       # DeepSpeed configuration for gene detection
    ├── model/
    │   └── Model.py                  # Enhanced DNABERTClassifier for gene classification
    ├── training/
    │   ├── train.py                  # Gene classification training with attention masks
    │   ├── setup_deepspeed.py        # DeepSpeed setup with memory optimization
    │   └── evaluate.py               # Gene classification evaluation
    └── data/
        ├── dataset_loading.py        # Gene classification dataset with attention mask support
        └── tokenized_dataset.pt      # Processed gene data from DNA pipeline
    
```

#### Key Components

#### dna_sequence_processor.py (DNA Processing Pipeline)
- Comprehensive gene sequence processing with realistic NGS noise simulation
- Multi-technology support (Illumina, PacBio, ONT) with specific error patterns
- Dynamic windowing for variable-length sequences with attention masks
- Consistent gene numbering system with persistent mappings
- Dataset balancing through intelligent augmentation
- DNABERT2-compatible tokenization output

#### dataset_loading.py (Gene Classification Dataset)
- 10-50x faster data loading using pre-tokenized .pt files from DNA processing pipeline
- Gene classification labels with attention mask support
- Memory-efficient batch collation for variable-length sequences
- Curriculum sampling with balanced-to-natural gene distribution progression

#### Model.py (Gene Classification Architecture)
- Multi-scale token attention with rotary embeddings for gene detection
- Gene-aware DNA attention with landmark pooling
- Dedicated classification head for multi-class gene prediction
- Individual temperature scaling for calibrated gene predictions

#### train.py (Gene Classification Training)
- Gene classification training with attention mask integration
- Comprehensive gene detection metric tracking
- Advanced visualization with gene-specific attention patterns
- Temperature parameter monitoring and optimization for gene classification

## 🧪 Dataset

### Data Format
- **Gene Sequences**: DNA sequences from various organisms with gene annotations
- **Multi-Class Labels**: Gene classification labels with consistent numbering across runs
- **Enhanced Data Augmentation**: 
  - **NGS Noise Simulation**: Realistic sequencing errors for Illumina, PacBio, and ONT technologies
  - **Multiple Noise Variants**: Different noise levels (low, medium, high) per gene sequence
  - **Balanced Dataset Generation**: Intelligent augmentation of underrepresented genes
- **Tokenization**: Pre-tokenized using DNABERT-2 tokenizer with attention masks
- **Sequence Length**: Up to 512 tokens with dynamic windowing for longer sequences
- **Gene Distribution**: Balanced through intelligent augmentation with curriculum progression

### Gene Classification Labels
Each sample includes gene-specific annotations:

#### Classification Labels
- **Gene Labels**: Multi-class labels for gene identification (consistent numbering)
- **Attention Masks**: Proper masking for variable-length sequences
- **Gene Mapping**: Persistent gene-to-label mappings across training runs

#### Input Features
- **Tokenized Sequences**: DNABERT-2 compatible token IDs
- **Attention Masks**: Binary masks indicating valid token positions
- **Sequence Metadata**: Gene names, organism information, sequence lengths

#### Data Augmentation Features
- **Noise Variants**: Multiple noisy versions of each gene sequence
- **Technology Simulation**: Error patterns specific to sequencing technologies
- **Balanced Sampling**: Intelligent augmentation for underrepresented genes

### Enhanced Data Processing Pipeline
1. **Optimized Loading**: Pre-tokenized .pt files for 10-50x faster data loading
2. **Gene Processing**: DNA sequence processing pipeline with realistic NGS noise
3. **Data Augmentation**: 
   - Multiple noise variants per gene sequence (low, medium, high noise levels)
   - Technology-specific error patterns (Illumina, PacBio, ONT)
   - Dynamic windowing for variable-length sequences
4. **Efficient Collation**: 
   - Dense tensors for tokenized sequences (input_ids, attention_mask, labels)
   - Memory-efficient batch assembly for variable-length sequences
   - Proper attention mask handling
5. **Curriculum Integration**: Balanced-to-natural gene distribution progression
6. **Multi-GPU Compatibility**: Distributed data loading with proper synchronization

## 📊 Monitoring & Evaluation

### Wandb Integration
Comprehensive experiment tracking including:

#### Gene Classification Metrics
- **Training Metrics**: Loss curves, learning rate schedules
- **Validation Metrics**: Accuracy, precision, recall, F1-score per gene class
- **Confusion Matrices**: Gene-wise classification performance analysis
- **Confidence Distributions**: Model calibration assessment for gene predictions

#### Gene Detection Performance
- **Per-Gene Analysis**: Individual gene classification accuracy
- **Noise Robustness**: Performance across different noise levels and technologies
- **Sequence Length Analysis**: Performance on different sequence lengths
- **Attention Pattern Analysis**: Gene-specific attention visualization
- **Dataset Balance Tracking**: Gene distribution and augmentation effectiveness

#### Visualization & Analysis
- **Enhanced Token Visualizations**: Gene-specific attention patterns and classification predictions
- **Interactive HTML Tables**: Token-by-token gene classification results
- **Gene-Specific Visualizations**: Per-gene attention maps and prediction confidence
- **Multi-Scale Attention Maps**: Motif, regulatory, and domain-level attention patterns
- **Hardware Utilization**: GPU memory and compute usage optimization
- **Temperature Analysis**: Individual temperature parameter evolution for gene classification heads

### Key Metrics

#### Gene Classification Performance
- **Primary**: Multi-class accuracy for gene identification
- **Secondary**: Per-gene precision, recall, F1-score
- **Calibration**: Expected Calibration Error (ECE) for gene predictions
- **Consistency**: Agreement between multiple gene classification predictions

#### Gene Detection Robustness
- **Noise Tolerance**: Performance across different NGS error rates and types
- **Technology Robustness**: Consistent performance across Illumina, PacBio, and ONT simulations
- **Sequence Length Handling**: Performance on variable-length sequences with attention masks
- **Gene Distribution**: Balanced performance across common and rare genes
- **Attention Quality**: Meaningful attention patterns for gene-specific regions

## 🔧 Advanced Features

### Flash Attention Compatibility
The system includes comprehensive patches for DNABERT-2's flash attention implementation:
- **Triton Kernel Fallback**: Automatic fallback when Triton version conflicts occur
- **Standard Attention Replacement**: PyTorch-native attention as backup
- **Import Interception**: Catches and redirects flash attention imports
- **Cache Patching**: Handles cached transformers modules

### Multi-GPU Support
- **Automatic Detection**: Detects available GPUs and optimal configuration
- **Distributed Training**: NCCL backend for efficient communication
- **Fault Tolerance**: Automatic fallback to single-GPU if distributed fails
- **Load Balancing**: Even data distribution across GPUs

## 🚀 Usage Examples

### Basic Training
```bash
cd BertModel/scripts
python launch_training.py
```

### View Configuration
```bash
cd BertModel/configs
python -c "from config import default_config; print(default_config)"
```

### Multi-GPU Training
The launcher automatically detects and uses all available GPUs:
```bash
cd BertModel/scripts
python launch_training.py  # Automatically uses all GPUs
```

## 🛠️ Troubleshooting

### Common Issues
1. **Flash Attention Errors**: Automatically handled by fallback mechanisms
2. **Memory Errors**: Reduce batch size or enable CPU offloading
3. **NCCL Errors**: Use fallback environment variables (automatically set)
4. **Triton Conflicts**: Automatic patching resolves version incompatibilities

### Performance Tuning
- **Batch Size**: Increase until GPU memory is ~90% utilized
- **Gradient Accumulation**: Increase if batch size must be small
- **CPU Offloading**: Enable for very large models or limited GPU memory
- **Mixed Precision**: Enable FP16 for 2x speedup (enabled by default)

## 📝 Citation

If you use DeepGene in your research, please cite:

```bibtex
@misc{deepGene2025,
  title={DeepSNP3D: Gene Detection and Classification System with Enhanced DNABERT-2},
  author={LukeFord},
  year={2025},
  note={Advanced gene detection system with realistic NGS noise simulation and robust classification using DNABERT-2 transformer architecture},
  url={https://github.com/LukeFord22/DeepGene}
}
```

## 📄 License

This project is licensed under the terms specified in the LICENSE file.

## 🤝 Contributing



