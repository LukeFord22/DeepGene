import torch
import numpy as np
import torch.nn as nn
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix
import matplotlib.pyplot as plt

#=======================================================================================================
#                                                                               Evaluation Calculations
#=======================================================================================================


def evaluate(model, dataloader, device):
    model.eval()
    loss_total = 0

    # === Accumulators ===
    all_gene_preds, all_gene_labels = [], []
    all_conf_scores = []

    with torch.no_grad():
        for batch in dataloader:
            # Only move tensor data to device, handle non-tensor data separately
            device_batch = {}
            for k, v in batch.items():
                if isinstance(v, torch.Tensor):
                    device_batch[k] = v.to(device)
                else:
                    device_batch[k] = v  # Keep non-tensor data as is


            outputs = model(device_batch['input_ids'], device_batch['attention_mask'])

            # === Extract outputs ===
            gene_logits = outputs['gene']  # Gene classification logits
            conf_logits = outputs['conf']  # Confidence scores
            
            # === Process gene classification outputs ===
            labels = device_batch['labels']
            
            # Get predictions and labels
            gene_preds = gene_logits.argmax(dim=1).cpu().numpy()
            gene_labels = labels.cpu().numpy()
            
            # Extract confidence
            conf_scores = conf_logits.squeeze().cpu().numpy()
            
            # Append results
            all_gene_preds.extend(gene_preds.tolist())
            all_gene_labels.extend(gene_labels.tolist())
            all_conf_scores.extend(conf_scores.tolist())

            # Track loss
            gene_loss = nn.CrossEntropyLoss()(gene_logits, labels)
            loss_total += gene_loss.item()

    # === Compute Metrics ===
    val_loss = loss_total / len(dataloader)
    
    # Gene classification metrics
    gene_accuracy = accuracy_score(all_gene_labels, all_gene_preds)
    gene_precision, gene_recall, gene_f1, _ = precision_recall_fscore_support(
        all_gene_labels, all_gene_preds, average='weighted', zero_division=0
    )
    gene_cm = confusion_matrix(all_gene_labels, all_gene_preds)

    return {
        'val_loss': val_loss,
        'accuracy': gene_accuracy, 
        'precision': gene_precision, 
        'recall': gene_recall, 
        'f1': gene_f1,
        'confusion_matrix': gene_cm,
        'conf_scores': all_conf_scores,
        'gene_preds': all_gene_preds,
        'gene_labels': all_gene_labels,
    }




# Create custom confusion matrix plots with matplotlib
def plot_confusion_matrix(cm, classes, title='Confusion Matrix', cmap=plt.cm.Blues):
    fig, ax = plt.subplots(figsize=(8, 6))
    im = ax.imshow(cm, interpolation='nearest', cmap=cmap)
    ax.figure.colorbar(im, ax=ax)
    
    # We want to show all ticks and label them with the respective list entries
    ax.set(xticks=np.arange(cm.shape[1]),
        yticks=np.arange(cm.shape[0]),
        xticklabels=classes, yticklabels=classes,
        title=title,
        ylabel='True Label',
        xlabel='Predicted Label')
    
    # Rotate the tick labels and set their alignment
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
    
    # Loop over data dimensions and create text annotations
    thresh = cm.max() / 2
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], 'd'),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black")
            
    fig.tight_layout()
    return fig




# == confidence box plots ==
def create_confidence_boxplots(conf_correct, conf_incorrect):
    fig, ax1 = plt.subplots(1, 1, figsize=(10, 6))
    
    # Gene-level confidence box plot
    data = [conf_correct, conf_incorrect]
    labels = ['Correct', 'Incorrect']
    colors = ['#2ecc71', '#e74c3c']
    
    box1 = ax1.boxplot(data, labels=labels, patch_artist=True)
    for patch, color in zip(box1['boxes'], colors):
        patch.set_facecolor(color)
    
    ax1.set_title('Gene Classification Confidence', fontsize=14)
    ax1.set_ylabel('Confidence Score', fontsize=12)
    ax1.grid(True, linestyle='--', alpha=0.7)
    
    # Add individual points with jitter for better visualization
    for i, conf_data in enumerate(data):
        # Add some random jitter to the x-position
        x = np.random.normal(i+1, 0.08, size=len(conf_data))
        ax1.scatter(x, conf_data, alpha=0.3, s=10, c=colors[i])
    
    # Add statistics as text
    stats = f"Correct: mean={np.mean(conf_correct):.3f}, median={np.median(conf_correct):.3f}, n={len(conf_correct)}\n"
    stats += f"Incorrect: mean={np.mean(conf_incorrect):.3f}, median={np.median(conf_incorrect):.3f}, n={len(conf_incorrect)}"
    ax1.text(0.5, -0.15, stats, transform=ax1.transAxes, ha='center', fontsize=10)
    
    plt.tight_layout(pad=3.0)
    plt.suptitle('Confidence Distribution: Correct vs. Incorrect Gene Predictions', fontsize=16, y=1.05)
    
    return fig



# == histograms for more detailed distribution ==
def create_confidence_histograms(conf_correct, conf_incorrect):
    fig, ax1 = plt.subplots(1, 1, figsize=(10, 6))
    
    # Gene confidence histogram
    bins = np.linspace(0, 1, 21)
    ax1.hist([conf_correct, conf_incorrect], bins=bins, stacked=True, 
                color=['#2ecc71', '#e74c3c'], alpha=0.7, label=['Correct', 'Incorrect'])
    ax1.set_title('Gene Classification Confidence Distribution', fontsize=14)
    ax1.set_xlabel('Confidence Score', fontsize=12)
    ax1.set_ylabel('Count', fontsize=12)
    ax1.legend()
    ax1.grid(True, linestyle='--', alpha=0.7)
    
    return fig



# == Gene Classification Sample Visualization ==



def create_gene_samples(model, dataloader, device, tokenizer, dataset, max_samples=5):
    """Create gene classification sample data for visualization"""
    model.eval()
    samples = []
    
    sample_count = 0
    with torch.no_grad():
        for batch in dataloader:
            if sample_count >= max_samples:
                break
                
            # Move batch to device
            device_batch = {}
            for k, v in batch.items():
                if isinstance(v, torch.Tensor):
                    device_batch[k] = v.to(device)
                else:
                    device_batch[k] = v
            
            # Get predictions
            outputs = model(device_batch['input_ids'], device_batch['attention_mask'])
            
            # Extract predictions and confidence
            gene_preds = outputs['gene'].argmax(dim=-1)
            conf_scores = outputs['conf'].squeeze(-1)
            
            # Process each sequence in the batch
            for i in range(len(device_batch['input_ids'])):
                if sample_count >= max_samples:
                    break
                    
                sample_count += 1
                
                # Get sample data
                input_ids = device_batch['input_ids'][i]
                attention_mask = device_batch['attention_mask'][i]
                true_label = device_batch['labels'][i].item()
                pred_label = gene_preds[i].item()
                confidence = conf_scores[i].item()
                
                # Get gene names from label mapping if available
                if hasattr(dataset, 'label_mapping'):
                    true_gene = dataset.label_mapping['label_to_gene'].get(str(true_label), f'Gene_{true_label}')
                    pred_gene = dataset.label_mapping['label_to_gene'].get(str(pred_label), f'Gene_{pred_label}')
                else:
                    true_gene = f'Gene_{true_label}'
                    pred_gene = f'Gene_{pred_label}'
                
                # Get sequence tokens (first 50 tokens for readability)
                valid_tokens = input_ids[attention_mask.bool()][:50]
                sequence = tokenizer.decode(valid_tokens, skip_special_tokens=True)
                
                samples.append({
                    "id": f"sample_{sample_count}",
                    "sequence": sequence[:200] + "..." if len(sequence) > 200 else sequence,
                    "true_gene": true_gene,
                    "predicted_gene": pred_gene,
                    "confidence": confidence,
                    "correct": true_label == pred_label
                })
    
    return samples


def create_gene_html_tables(samples):
    """Create HTML table visualizations for gene classification samples"""
    html_tables = []
    
    for sample in samples:
        # Determine background color based on correctness
        bg_color = "#ccffcc" if sample['correct'] else "#ffcccc"  # Light green or light red
        
        html = f"""
        <div style="margin-bottom: 20px; padding: 10px; border: 1px solid #ddd; border-radius: 5px; background-color: {bg_color};">
            <h3>Sample {sample['id']}</h3>
            <table style="border-collapse: collapse; width: 100%; font-size: 12px;">
                <tr>
                    <td style="padding: 8px; font-weight: bold;">True Gene:</td>
                    <td style="padding: 8px;">{sample['true_gene']}</td>
                </tr>
                <tr>
                    <td style="padding: 8px; font-weight: bold;">Predicted Gene:</td>
                    <td style="padding: 8px;">{sample['predicted_gene']}</td>
                </tr>
                <tr>
                    <td style="padding: 8px; font-weight: bold;">Confidence:</td>
                    <td style="padding: 8px;">{sample['confidence']:.3f}</td>
                </tr>
                <tr>
                    <td style="padding: 8px; font-weight: bold;">Correct:</td>
                    <td style="padding: 8px;">{'✓' if sample['correct'] else '✗'}</td>
                </tr>
                <tr>
                    <td style="padding: 8px; font-weight: bold; vertical-align: top;">Sequence:</td>
                    <td style="padding: 8px; font-family: monospace; font-size: 10px; word-break: break-all;">{sample['sequence']}</td>
                </tr>
            </table>
        </div>
        """
        html_tables.append(html)
    
    return "".join(html_tables)

def save_model(model_engine, path='Gene.pth'):
    # Save the model to the current directory
    import os
    
    # Make sure we use a path in the current working directory
    if os.path.isabs(path):
        # Extract just the filename from the absolute path
        path = os.path.basename(path)
    
    # Save the underlying model (not the DeepSpeed engine)
    torch.save(model_engine.module.state_dict(), path)
    print(f"Model saved locally to {path}")