"""Setup function for DeepSpeed training initialization."""

import torch
import deepspeed
import os
import sys
from pathlib import Path
import numpy as np
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, AutoModel, BertConfig
import torch.distributed as dist
import time

# Add parent directories to Python path for absolute imports
script_dir = os.path.dirname(os.path.abspath(__file__))
training_dir = os.path.dirname(script_dir)
bert_model_dir = os.path.dirname(training_dir)
deepsnp_dir = os.path.dirname(bert_model_dir)
sys.path.insert(0, deepsnp_dir)
sys.path.insert(0, bert_model_dir)
sys.path.insert(0, training_dir)
sys.path.insert(0, script_dir)

# Import from packages using absolute imports
from BERTModel.model.Model import DNABERTClassifier
from BERTModel.data.dataset_loading import create_gene_dataset, collate_fn
from BERTModel.configs.deepspeed_config import create_deepspeed_config


def setup_training(config):
    """
    Prepares the model, optimizer, scheduler, dataloaders, device, and ranks.
    Initializes DeepSpeed model engine with proper configuration.
    """
    # Get distributed info from environment variables (set by launcher or main process)
    local_rank = int(os.environ.get('LOCAL_RANK', 0))
    world_size = int(os.environ.get('WORLD_SIZE', 1))
    rank = int(os.environ.get('RANK', 0))
    
    # Set device based on local rank
    if torch.cuda.is_available():
        device = torch.device(f'cuda:{local_rank}')
        torch.cuda.set_device(local_rank)
    else:
        device = torch.device('cpu')
    
    if local_rank <= 0:
        print(f"🔧 Setup training - Rank: {rank}, Local Rank: {local_rank}, World Size: {world_size}")
        print(f"🔧 Device: {device}")
    
    

    # === DATA LOADING SETUP  ===
    # Set path to the new gene classification data
    gene_data_dir = "DeepGene/BERTModel/data/SNPdata"
    pt_file_path = os.path.join(gene_data_dir, "tokenized_dataset.pt")
    gene_mapping_path = os.path.join(gene_data_dir, "gene_mapping.json")

    if local_rank <= 0:
        print(f"📁 Loading gene classification data from: {gene_data_dir}")
        print(f"📁 PT file: {pt_file_path}")
        print(f"📁 Gene mapping: {gene_mapping_path}")

    # Create the gene classification dataset
    dataset = create_gene_dataset(
        gene_data_dir,
        load_fraction=config.training.train_data_fraction,
        prefer_compressed=True
    )

    # Create test indices for evaluation
    test_indices = np.random.choice(
        range(len(dataset)), 
        size=int(len(dataset) * config.data.test_fraction),
        replace=False
    )
    
    # Create test dataset and loader
    test_dataset = torch.utils.data.Subset(dataset, test_indices)
    test_loader = DataLoader(
        test_dataset,
        batch_size=config.training.batch_size, 
        shuffle=False,
        pin_memory=config.data.pin_memory, 
        num_workers= 0,
        collate_fn=collate_fn,
        drop_last=True
    )

    # === LOCAL DNABERT-2 SETUP ===
    hf_model_name = "zhihan1996/DNABERT-2-117M"

    local_path = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 
        "model", "DNABERT-2-117M"
    )

    # Check if weights exist, download if not
    weight_files = ["pytorch_model.bin", "model.safetensors"]
    weights_exist = any(os.path.exists(os.path.join(local_path, f)) for f in weight_files)
    
    if not weights_exist:
        if local_rank <= 0:
            print(f"🔄 Model weights not found locally. Downloading from {hf_model_name}...")
            download_model_weights(hf_model_name, local_path, local_rank)
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            torch.distributed.barrier()

        # All ranks: wait until weights are present
        while not any(os.path.exists(os.path.join(local_path, f)) for f in weight_files):
            time.sleep(1)
    else:
        if local_rank <= 0:
            print(f"✅ Model weights found locally at {local_path}")


    # Load tokenizer and model configuration
    dna_tokenizer = AutoTokenizer.from_pretrained(local_path, trust_remote_code=True)
    bert_config = BertConfig.from_pretrained(local_path)

    # Step 1: Load local model architecture
    print(f"🔍 Loading model architecture from {local_path}...")
    base_model = AutoModel.from_pretrained(
        local_path,
        config=bert_config,
        trust_remote_code=True,
        torch_dtype=torch.float32
    )
    
    model = DNABERTClassifier(base_model, config=config)

    # Create DeepSpeed configuration
    ds_config = create_deepspeed_config(
        config=config,
        world_size=world_size
    )
    
    # Initialize DeepSpeed model engine
    print("🔧 Initializing DeepSpeed model engine...")
    model_engine, optimizer, _, scheduler = deepspeed.initialize(
        model=model,
        config=ds_config,
        model_parameters=model.parameters()
    )

    # Create train loader 
    train_indices = np.arange(len(dataset))
    
    # Remove test indices from train indices
    train_indices = np.setdiff1d(train_indices, test_indices)
    
    train_dataset = torch.utils.data.Subset(dataset, train_indices)
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.training.batch_size,
        shuffle=True,
        pin_memory=config.data.pin_memory,
        num_workers=0,
        collate_fn=collate_fn,
        drop_last=True
    )
    
    if local_rank <= 0:
        print(f"📊 Dataset split:")
        print(f"  - Training samples: {len(train_indices)}")
        print(f"  - Test samples: {len(test_indices)}")
        print(f"  - Total samples: {len(dataset)}")

    return (model_engine, optimizer, scheduler, train_loader, test_loader, 
            device, local_rank, dna_tokenizer, dataset, None, 
            world_size, rank)


def download_model_weights(hf_model_name, local_path, local_rank=0):
    """
    Download only the model weight files from Hugging Face to local directory
    """
    import os
    from huggingface_hub import hf_hub_download
    
    # Create local directory if it doesn't exist
    os.makedirs(local_path, exist_ok=True)
    
    # Files to download (model weights)
    weight_files = [
        "pytorch_model.bin",
    ]
    
    downloaded_files = []
    
    for filename in weight_files:
        try:
            if local_rank <= 0:
                print(f"📥 Downloading {filename}...")
            
            # Download file to local directory
            downloaded_path = hf_hub_download(
                repo_id=hf_model_name,
                filename=filename,
                local_dir=local_path,
                local_dir_use_symlinks=False  # Create actual files, not symlinks
            )
            downloaded_files.append(filename)
            
            if local_rank <= 0:
                print(f"✅ Downloaded {filename}")
                
        except Exception as e:
            if local_rank <= 0:
                print(f"⚠️ Could not download {filename}: {e}")
    
    if local_rank <= 0:
        print(f"📦 Downloaded {len(downloaded_files)} files to {local_path}")
        print(f"   Files: {downloaded_files}")
    
    return downloaded_files

