import torch
import json
import gzip
from pathlib import Path


#=======================================================================================================
#                                                                                        Dataset Loading
#======================================================================================================= 


class GeneClassificationDataset(torch.utils.data.Dataset):
    """
    Dataset for gene classification using single PT file from DNAprocessing output.
    Supports both compressed (.pt.gz) and uncompressed (.pt) files.
    """
    def __init__(self, pt_file_path, gene_mapping_path, load_fraction=1.0):
        """
        Dataset that loads gene classification data from a single PT file
        
        Args:
            pt_file_path: Path to the tokenized_dataset.pt or tokenized_dataset.pt.gz file
            gene_mapping_path: Path to the gene_mapping.json file
            load_fraction: Fraction of data to load (for debugging)
        """
        super().__init__()
        self.pt_file_path = pt_file_path
        self.load_fraction = load_fraction
        
        # Load the data (supports both .pt and .pt.gz files)
        print(f"Loading gene classification dataset from {pt_file_path}...")
        self.data = self._load_data(pt_file_path)
        
        # Load gene mapping
        with open(gene_mapping_path, 'r') as f:
            self.gene_mapping = json.load(f)
        
        self.num_classes = len(self.gene_mapping['gene_to_label'])
        
        # Apply load fraction if specified
        self.num_samples = len(self.data['input_ids'])
        if load_fraction < 1.0:
            keep_count = int(self.num_samples * load_fraction)
            self.num_samples = keep_count
            print(f"Applied load fraction {load_fraction}: keeping {keep_count}/{len(self.data['input_ids'])} samples")
        
        print(f"GeneClassificationDataset initialized:")
        print(f"  - Total samples: {self.num_samples}")
        print(f"  - Number of gene classes: {self.num_classes}")
        print(f"  - Sequence length: {self.data['input_ids'].shape[1]}")
        print(f"  - Attention mask available: {'attention_mask' in self.data}")
    
    def _load_data(self, file_path):
        """
        Load data from either compressed (.pt.gz) or uncompressed (.pt) files
        
        Args:
            file_path: Path to the data file (.pt or .pt.gz)
            
        Returns:
            dict: Loaded data dictionary
        """
        file_path = Path(file_path)
        
        if file_path.suffix == '.gz':
            # Handle compressed file
            print(f"  - Loading compressed file: {file_path}")
            with gzip.open(file_path, 'rb') as f:
                data = torch.load(f, map_location='cpu')
        elif file_path.suffix == '.pt':
            # Handle uncompressed file
            print(f"  - Loading uncompressed file: {file_path}")
            data = torch.load(file_path, map_location='cpu')
        else:
            # Try to detect based on file content or provide fallback
            try:
                # First try as compressed
                with gzip.open(file_path, 'rb') as f:
                    data = torch.load(f, map_location='cpu')
                print(f"  - Successfully loaded as compressed file: {file_path}")
            except (gzip.BadGzipFile, OSError):
                # Fall back to uncompressed
                data = torch.load(file_path, map_location='cpu')
                print(f"  - Successfully loaded as uncompressed file: {file_path}")
        
        return data
    
    def __len__(self):
        return self.num_samples
    
    def __getitem__(self, idx):
        """Get a single gene classification sample"""
        if idx >= self.num_samples:
            raise IndexError(f"Index {idx} out of range for dataset of size {self.num_samples}")
        
        # Extract sample data
        input_ids = self.data['input_ids'][idx]
        labels = self.data['labels'][idx]
        
        # Use saved attention mask if available, otherwise generate it
        if 'attention_mask' in self.data:
            attention_mask = self.data['attention_mask'][idx]
        else:
            # Fallback: Create attention mask (1 for non-padding tokens, 0 for padding)
            # Assuming padding token is 0
            attention_mask = (input_ids != 0).long()
        
        sample = {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'labels': labels
        }
        
        return sample



def collate_fn(batch):
    """Collate function for gene classification data"""
    batch_size = len(batch)
    
    if batch_size == 0:
        return {}
    
    # Stack dense data
    input_ids = torch.stack([item['input_ids'] for item in batch])
    attention_mask = torch.stack([item['attention_mask'] for item in batch])
    labels = torch.stack([item['labels'] for item in batch])
    
    return {
        'input_ids': input_ids,
        'attention_mask': attention_mask,
        'labels': labels
    }


def create_gene_dataset(data_dir, load_fraction=1.0, prefer_compressed=True):
    """
    Convenience function to create a GeneClassificationDataset with automatic file detection
    
    Args:
        data_dir: Directory containing the dataset files
        load_fraction: Fraction of data to load (for debugging)
        prefer_compressed: Whether to prefer compressed files if both exist
    
    Returns:
        GeneClassificationDataset: Initialized dataset
    """
    data_dir = Path(data_dir)
    
    # Look for dataset files
    compressed_file = data_dir / "tokenized_dataset.pt.gz"
    uncompressed_file = data_dir / "tokenized_dataset.pt"
    gene_mapping_file = data_dir / "gene_mapping.json"
    
    # Determine which file to use
    if prefer_compressed and compressed_file.exists():
        dataset_file = compressed_file
        print(f"Using compressed dataset: {dataset_file}")
    elif uncompressed_file.exists():
        dataset_file = uncompressed_file
        print(f"Using uncompressed dataset: {dataset_file}")
    elif compressed_file.exists():
        dataset_file = compressed_file
        print(f"Using compressed dataset: {dataset_file}")
    else:
        available_files = list(data_dir.glob("*.pt*"))
        raise FileNotFoundError(
            f"No dataset files found in {data_dir}. "
            f"Available files: {available_files}"
        )
    
    # Check for gene mapping
    if not gene_mapping_file.exists():
        raise FileNotFoundError(f"Gene mapping file not found: {gene_mapping_file}")
    
    return GeneClassificationDataset(
        pt_file_path=str(dataset_file),
        gene_mapping_path=str(gene_mapping_file),
        load_fraction=load_fraction
    )