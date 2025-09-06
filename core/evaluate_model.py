import torch
import torch.nn as nn
import os
import json
import argparse 
from datetime import datetime 


# --- Evaluation ---
def evaluate_model(model, X_eval_tensor, y_eval_tensor, core_dir='core', output_dir='results', log_details=False):
    """
    Evaluates a trained PyTorch LSTM model, saves accuracy to core_dir, and optionally logs details to output_dir.
    Returns evaluation accuracy.
    Args:
        model (torch.nn.Module): Trained PyTorch model.
        X_eval_tensor (torch.Tensor): Evaluation features tensor.
        y_eval_tensor (torch.Tensor): Evaluation targets tensor.
        core_dir (str): Directory to save core metrics (e.g., accuracy).
        output_dir (str): Directory to save optional logs.
        log_details (bool): If True, log detailed evaluation results to output_dir.
    Returns:
        float: Evaluation accuracy.
    """
    # Ensure directories exist
    os.makedirs(core_dir, exist_ok=True)
    os.makedirs(output_dir, exist_ok=True)

    device = next(model.parameters()).device # Get model's device
    model.eval()
    with torch.no_grad():
        outputs = model(X_eval_tensor.to(device))
        predicted_classes = (outputs > 0.5).squeeze().float()
        correct_predictions = (predicted_classes == y_eval_tensor.to(device).squeeze()).sum().item()
        total_predictions = y_eval_tensor.size(0)
        accuracy = correct_predictions / total_predictions if total_predictions > 0 else 0.0

    print(f"Evaluation Accuracy: {accuracy:.4f}")

    # Save metrics to core folder
    metrics_path = os.path.join(core_dir, 'evaluation_metrics.json')
    metrics_data = {'accuracy': accuracy}
    try:
        with open(metrics_path, 'w') as f:
            json.dump(metrics_data, f)
        print(f"Evaluation metrics saved to {metrics_path}")
    except Exception as e:
        print(f"Warning: Could not save evaluation metrics to {metrics_path}: {e}")

    # Optional: Log evaluation details to output folder
    if log_details:
        log_path = os.path.join(output_dir, 'evaluation_log.txt')
        try:
            with open(log_path, 'a') as f:
                f.write(f"Evaluation run on {datetime.now()}:\n")
                f.write(f"Accuracy: {accuracy:.4f}\n")
                # Add more detailed logging here if needed
                # Example: f.write(f"Number of correct predictions: {correct_predictions}\n")
                # Example: f.write(f"Total predictions: {total_predictions}\n")
                f.write("-" * 20 + "\n")
            print(f"Evaluation log updated in {log_path}")
        except Exception as e:
            print(f"Warning: Could not write to evaluation log {log_path}: {e}")


    return accuracy