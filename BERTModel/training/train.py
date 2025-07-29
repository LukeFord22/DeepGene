import torch
import time
from tqdm import tqdm
import numpy as np
import torch.nn.functional as F
import torch.nn as nn
import wandb
import os
import sys
import matplotlib.pyplot as plt

# Add parent directories to Python path for absolute imports
script_dir = os.path.dirname(os.path.abspath(__file__))
training_dir = os.path.dirname(script_dir)
sys.path.insert(0, training_dir)
sys.path.insert(0, script_dir)

# Import from packages using absolute imports
from BERTModel.training.evaluate import evaluate, plot_confusion_matrix, create_confidence_boxplots, create_confidence_histograms, create_gene_samples, create_gene_html_tables, save_model
from BERTModel.training.setup_deepspeed import setup_training



def train_model(config=None):
    """Main training function for Gene Identification with DeepSpeed."""

    # Setup training components with DeepSpeed
    setup_results = setup_training(config)
    (model_engine, optimizer, _, train_loader, test_loader, 
     device, local_rank, dna_tokenizer, dataset, _, 
     world_size, _) = setup_results

    # Store runtime info in config for compatibility
    config.deepspeed_runtime = {
        'model_engine': model_engine,
        'device': device,
        'rank': local_rank,
        'world_size': world_size,
        'dna_tokenizer': dna_tokenizer,
        'test_loader': test_loader,
        'dataset': dataset,
    }



    # Only initialize wandb on rank 0
    if local_rank <= 0:
        wandb.init(
            project="gene-classification-deepspeed",
            config={
                "batch_size": config.training.batch_size,
                "epochs": config.training.num_epochs,
                "learning_rate": config.training.learning_rate,
                "grad_accum_steps": config.training.grad_accum_steps,
                "train_data_fraction": config.training.train_data_fraction,
                "num_gene_classes": dataset.num_classes,
                "deepspeed": True,
                "local_rank": local_rank,
                "world_size": torch.distributed.get_world_size() if torch.distributed.is_initialized() else 1
            }
        )

    #=======================================================================================================
    #                                                                                          Training Loop
    #=======================================================================================================

    global_step = 0
    best_eval_metrics = None
    best_val_f1 = -float('inf') 
    total_steps = 0  

    if local_rank <= 0:
        print("Training Model with DeepSpeed...")
        global_start_time = time.time()
        steps_done = 0
        steps_per_epoch = len(train_loader)
        total_steps = steps_per_epoch * config.training.num_epochs


    
    for epoch in range(config.training.num_epochs):
        model_engine.train()
        epoch_loss = 0    

        # Training loop for this epoch  
        if local_rank <= 0:
            progress_bar = tqdm(total=steps_per_epoch, desc=f"Epoch {epoch+1}/{config.training.num_epochs}", leave=True)

        for batch in train_loader:
      
            # === Compute gene classification and confidence losses ===
            outputs = model_engine(batch['input_ids'].to(device), batch['attention_mask'].to(device))
            gene_logits = outputs['gene']  # Gene classification logits
            conf_logits = outputs['conf']  # Confidence scores
            labels = batch['labels'].to(device).long()
            
            # Calculate gene classification loss
            gene_loss_fn = nn.CrossEntropyLoss()
            gene_loss = gene_loss_fn(gene_logits, labels)
            
            # Calculate confidence loss (MSE with correctness)
            with torch.no_grad():
                predictions = gene_logits.argmax(dim=1)
                correctness = (predictions == labels).float()  # 1.0 for correct, 0.0 for incorrect
            
            # Confidence loss: model should be confident when correct, uncertain when wrong
            conf_loss_fn = nn.MSELoss()
            conf_loss = conf_loss_fn(conf_logits.squeeze(), correctness)
            
            # Combined loss
            total_loss = (gene_loss + 0.1 * conf_loss) / config.training.grad_accum_steps

            # === Accuracy and confidence monitoring ===
            with torch.no_grad():
                correct = (predictions == labels).sum().item()
                accuracy = correct / labels.size(0)
                avg_confidence = conf_logits.mean().item()

            # === Log Step Data ===
            if local_rank <= 0:
                steps_done += 1
                elapsed_global = time.time() - global_start_time
                avg_steps_per_sec_global = steps_done / elapsed_global if elapsed_global > 0 else 1
                steps_left_global = total_steps - steps_done
                eta_seconds_global = int(steps_left_global / avg_steps_per_sec_global) if avg_steps_per_sec_global > 0 else 0
                eta_h = int(eta_seconds_global // 3600)
                eta_m = int((eta_seconds_global % 3600) // 60)
                eta_s = int(eta_seconds_global % 60)
                eta_string = f"{eta_h:02d}:{eta_m:02d}:{eta_s:02d}"
                eta_h = eta_seconds_global / 3600.0
              
                progress_bar.set_postfix({
                    'total_loss': f'{total_loss.item() * config.training.grad_accum_steps:.3f}',
                    'gene_loss': f'{gene_loss.item():.3f}',
                    'conf_loss': f'{conf_loss.item():.3f}',
                    'acc': f'{accuracy:.3f}',
                    'conf': f'{avg_confidence:.3f}',
                    'ETA': eta_string
                })

                progress_bar.update(1)

                # Log to wandb
                wandb.log({
                    'loss/total': total_loss.detach().item() * config.training.grad_accum_steps,
                    'loss/gene_classification': gene_loss.detach().item(),
                    'loss/confidence': conf_loss.detach().item(),
                    'training/eta_hours': eta_h,
                    'training/learning_rate': optimizer.param_groups[0]['lr'],
                    'training/step': global_step,
                    'training/accuracy': accuracy,
                    'training/avg_confidence': avg_confidence,
                })

            # === BACKWARD PASS ===
            model_engine.backward(total_loss)

            # === Model engine Step ===
            model_engine.step()
            global_step += 1
            epoch_loss += total_loss.detach().item()        
            

        # === Epoch Validation & Evaluation ===
        eval_metrics = evaluate(model_engine.module, test_loader, device)
        
        # Log validation metrics to wandb (only on rank 0)
        if local_rank <= 0:
            wandb.log({
                'epoch/loss': eval_metrics['val_loss'],
                'epoch/accuracy': eval_metrics['accuracy'],
                'epoch/epoch': epoch + 1
            })

            # === Save best model checkpoint ===
            current_accuracy = eval_metrics['accuracy']
            current_loss = eval_metrics['val_loss']

            # === Save based on accuracy ===
            if (epoch + 1) % 10 == 0 and current_accuracy > best_val_f1:  # Reusing best_val_f1 variable for best accuracy
                best_val_f1 = current_accuracy
                checkpoint_path = f"GeneClassification_epoch{epoch+1}_acc{current_accuracy:.4f}.pth"
                torch.save(model_engine.module.state_dict(), checkpoint_path)
                print(f"Saved new best model to {checkpoint_path} (Accuracy: {current_accuracy:.4f} Loss: {current_loss:.4f})")


            progress_bar.close()

            # Print epoch summary
            avg_loss = epoch_loss / steps_per_epoch
            tqdm.write(f"Epoch {epoch+1}/{config.training.num_epochs} | Train Loss: {avg_loss:.4f} | Val Loss: {eval_metrics['val_loss']:.4f} | Val Accuracy: {eval_metrics['accuracy']:.4f}")


    # === Final Evaluation ===
    if local_rank <= 0:
        print("Training complete! Logging final results...")

        # Run a final evaluation after training is complete
        print("Running final evaluation to collect metrics...")
        final_eval_metrics = evaluate(model_engine.module, test_loader, device)
        best_eval_metrics = final_eval_metrics  # Use the final metrics

        # Log final metrics and visualizations
        print("Logging final evaluation metrics and visualizations...")
        
        # Basic metrics
        wandb.log({
            "final/accuracy": best_eval_metrics['accuracy'],
            "final/precision": best_eval_metrics['precision'],
            "final/recall": best_eval_metrics['recall'],
            "final/f1": best_eval_metrics['f1'],
            "final/loss": best_eval_metrics['val_loss']
        })
        
        # Create and log confusion matrix
        gene_cm = best_eval_metrics['confusion_matrix']
        # Use first 50 gene labels for readability if there are many classes
        if gene_cm.shape[0] > 50:
            print(f"Confusion matrix too large ({gene_cm.shape[0]} classes), logging first 50x50 subset")
            cm_subset = gene_cm[:50, :50]
            gene_class_names = [f"Gene_{i}" for i in range(50)]
            cm_fig = plot_confusion_matrix(
                cm_subset,
                classes=gene_class_names,
                title='Gene Classification Confusion Matrix (Top 50 Classes)'
            )
        else:
            gene_class_names = [f"Gene_{i}" for i in range(gene_cm.shape[0])]
            cm_fig = plot_confusion_matrix(
                gene_cm,
                classes=gene_class_names,
                title='Gene Classification Confusion Matrix'
            )
        
        wandb.log({"final/confusion_matrix": wandb.Image(cm_fig)})
        plt.close(cm_fig)
        
        # Confidence analysis: separate correct vs incorrect predictions
        gene_preds = best_eval_metrics['gene_preds']
        gene_labels = best_eval_metrics['gene_labels']
        conf_scores = best_eval_metrics['conf_scores']
        
        # Separate confidence scores by correctness
        conf_correct = [conf for conf, pred, label in zip(conf_scores, gene_preds, gene_labels) if pred == label]
        conf_incorrect = [conf for conf, pred, label in zip(conf_scores, gene_preds, gene_labels) if pred != label]
        
        if len(conf_correct) > 0 and len(conf_incorrect) > 0:
            # Create confidence analysis plots
            conf_fig = create_confidence_boxplots(conf_correct, conf_incorrect)
            wandb.log({"final/confidence_analysis": wandb.Image(conf_fig)})
            plt.close(conf_fig)
            
            # Create confidence histograms
            hist_fig = create_confidence_histograms(conf_correct, conf_incorrect)
            wandb.log({"final/confidence_histograms": wandb.Image(hist_fig)})
            plt.close(hist_fig)
        
        # Create sample visualization (simplified for gene classification)
        try:
            gene_samples = create_gene_samples(
                model_engine.module,
                test_loader,
                device,
                dna_tokenizer,
                dataset,
                max_samples=5
            )
            if gene_samples:
                html_tables = create_gene_html_tables(gene_samples)
                wandb.log({"final/sample_predictions": wandb.Html(html_tables)})
        except Exception as e:
            print(f"Warning: Could not create sample visualization: {e}")

        # Save final model
        save_model(model_engine)        # Finish the wandb run
        print("All metrics logged successfully!")
        wandb.finish()
