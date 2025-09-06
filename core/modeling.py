import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import json
import os
import numpy as np 

# --- Modeling ---
class LSTMModel(nn.Module):
    """
    Optimized LSTM architecture for sequence classification (next-day up/down).
    - BiLSTM stack
    - LayerNorm on inputs and sequence reps
    - Temporal attention over sequence
    - Feature-wise squeeze-excitation gating
    - Residual MLP head with GELU
    Output: sigmoid probability in [0,1].
    """
    def __init__(self, input_dim, hidden_dim, output_dim, dropout_prob):
        super().__init__()
        d_model = hidden_dim  # use hidden_dim as model width

        # Feature-wise gating (squeeze-excitation over time)
        self.feature_gate = nn.Sequential(
            nn.Linear(input_dim, max(8, input_dim // 2)),
            nn.GELU(),
            nn.Linear(max(8, input_dim // 2), input_dim),
            nn.Sigmoid()
        )

        # Project inputs to model width + LayerNorm
        self.in_proj = nn.Linear(input_dim, d_model)
        self.in_ln = nn.LayerNorm(d_model)

        # BiLSTM stack
        self.lstm1 = nn.LSTM(d_model, hidden_dim, batch_first=True, bidirectional=True)
        self.lstm2 = nn.LSTM(2 * hidden_dim, hidden_dim, batch_first=True, bidirectional=True)
        self.seq_ln = nn.LayerNorm(2 * hidden_dim)  # bi-directional output width

        # Temporal attention (learn what timesteps matter)
        self.attn_proj = nn.Linear(2 * hidden_dim, hidden_dim)
        self.attn_vec = nn.Linear(hidden_dim, 1, bias=False)  # context vector

        # Classification head (residual MLP)
        self.head_ln = nn.LayerNorm(2 * hidden_dim)
        self.fc1 = nn.Linear(2 * hidden_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim // 2)
        self.out = nn.Linear(hidden_dim // 2, output_dim)

        self.dropout = nn.Dropout(dropout_prob)
        self.sigmoid = nn.Sigmoid()

        self._reset_parameters()

    def _reset_parameters(self):
        # Kaiming init for Linear layers
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_uniform_(m.weight, a=0.2)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
        # Orthogonal init for LSTM weights, zeros for biases
        for lstm in [self.lstm1, self.lstm2]:
            for name, param in lstm.named_parameters():
                if 'weight_ih' in name or 'weight_hh' in name:
                    for p in param.chunk(4, 0):
                        nn.init.orthogonal_(p)
                elif 'bias' in name:
                    nn.init.zeros_(param)

    def forward(self, x):
        """
        x: (batch, seq_len, input_dim)
        returns: (batch, 1) sigmoid probability
        """
        B, T, feat_dim = x.shape   # <-- renamed

        # Feature-wise gating
        feat_stat = x.mean(dim=1)                      # (B, feat_dim)
        gate = self.feature_gate(feat_stat)            # (B, feat_dim)
        x = x * gate.unsqueeze(1)                      # broadcast over time

        # Input projection + norm
        x = self.in_proj(x)                            # (B, T, d_model)
        x = self.in_ln(x)

        # BiLSTM stack
        h, _ = self.lstm1(x)                           # (B, T, 2H)
        h = self.dropout(h)
        h, _ = self.lstm2(h)                           # (B, T, 2H)
        h = self.seq_ln(h)
        h = self.dropout(h)

        # Temporal attention
        attn_hidden = torch.tanh(self.attn_proj(h))    # (B, T, H)
        scores = self.attn_vec(attn_hidden).squeeze(-1)  # (B, T)
        weights = F.softmax(scores, dim=1)             # ✅ now works correctly
        context = torch.bmm(weights.unsqueeze(1), h).squeeze(1)  # (B, 2H)

        # Residual MLP head
        z = self.head_ln(context)
        z = self.fc1(z)
        z = F.gelu(z)
        z = self.dropout(z)
        z = self.fc2(z)
        z = F.gelu(z)
        z = self.dropout(z)

        out = self.out(z)                              # (B, output_dim)
        out = self.sigmoid(out)
        return out


def train_model(X_train_tensor, y_train_tensor, X_eval_tensor, y_eval_tensor,
                X_finetune_tensor=None, y_finetune_tensor=None, # Added fine-tuning data parameters
                input_dim, hidden_dim, output_dim, dropout_prob,
                lr, epochs, batch_size, model_path="models/best_model.pth", config_path="models/model_config.json"):
    """
    Trains the PyTorch LSTM model, saves the best model state dictionary based on evaluation accuracy.
    Loads existing model weights if available to continue training.
    Also saves model configuration. Can optionally fine-tune on a separate dataset.
    """
    # Ensure the models directory exists
    models_dir = os.path.dirname(model_path)
    os.makedirs(models_dir, exist_ok=True)


    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    model = LSTMModel(input_dim, hidden_dim, output_dim, dropout_prob).to(device)

    # Load existing model state if file exists
    if os.path.exists(model_path):
        print(f"Loading model state from {model_path}")
        try:
            # Optional: Load config to check compatibility before loading state dict
            if os.path.exists(config_path):
                with open(config_path, 'r') as f:
                    config = json.load(f)
                    # You might want to add checks here, e.g., if input_dim matches

            model.load_state_dict(torch.load(model_path, map_location=device))
            print("Model state loaded successfully.")
        except Exception as e:
            print(f"Error loading model state: {e}. Starting training from scratch.")
            # Consider adding more robust error handling or checks


    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    num_batches = max(1, len(X_train_tensor) // batch_size)

    best_accuracy = 0.0 # Track best accuracy for saving the best model
    training_losses = []

    # Move data to the same device as the model
    X_train_tensor = X_train_tensor.to(device)
    y_train_tensor = y_train_tensor.to(device)
    X_eval_tensor = X_eval_tensor.to(device)
    y_eval_tensor = y_eval_tensor.to(device)

    # Move fine-tuning data to device if provided
    if X_finetune_tensor is not None and y_finetune_tensor is not None:
        X_finetune_tensor = X_finetune_tensor.to(device)
        y_finetune_tensor = y_finetune_tensor.to(device)
        num_finetune_batches = max(1, len(X_finetune_tensor) // batch_size)
        print(f"Fine-tuning data available: {len(X_finetune_tensor)} samples.")
    else:
        num_finetune_batches = 0
        print("No fine-tuning data provided.")


    for epoch in range(epochs):
        model.train()
        total_loss = 0

        # --- Main Training Loop ---
        for i in range(num_batches):
            start = i * batch_size
            end = start + batch_size
            batch_X = X_train_tensor[start:end]
            batch_y = y_train_tensor[start:end]
            outputs = model(batch_X)
            loss = criterion(outputs.squeeze(), batch_y.float()) # Ensure outputs and targets have same shape and dtype
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        # --- Fine-tuning Loop (if data is provided) ---
        if num_finetune_batches > 0:
             print(f"Epoch [{epoch+1}/{epochs}] - Fine-tuning...")
             total_finetune_loss = 0
             for i in range(num_finetune_batches):
                  start = i * batch_size
                  end = start + batch_size
                  batch_X_ft = X_finetune_tensor[start:end]
                  batch_y_ft = y_finetune_tensor[start:end]
                  outputs_ft = model(batch_X_ft)
                  loss_ft = criterion(outputs_ft.squeeze(), batch_y_ft.float())
                  optimizer.zero_grad()
                  loss_ft.backward()
                  optimizer.step()
                  total_finetune_loss += loss_ft.item()
             avg_finetune_loss = total_finetune_loss / num_finetune_batches
             print(f"Epoch [{epoch+1}/{epochs}] - Fine-tune Loss: {avg_finetune_loss:.4f}")


        avg_loss = total_loss / num_batches
        training_losses.append(avg_loss)

        # Evaluate model after each epoch
        eval_accuracy = evaluate_model(model, X_eval_tensor, y_eval_tensor) # Pass tensors
        print(f'Epoch [{epoch+1}/{epochs}], Main Train Loss: {avg_loss:.4f}, Eval Accuracy: {eval_accuracy:.4f}')

        # Save best model based on evaluation accuracy
        if eval_accuracy > best_accuracy:
            best_accuracy = eval_accuracy
            print(f"Saving best model with accuracy {best_accuracy:.4f} to {model_path}")
            torch.save(model.state_dict(), model_path)

            # Save model configuration and training history
            config = {
                'input_dim': input_dim,
                'hidden_dim': hidden_dim,
                'output_dim': output_dim,
                'dropout_prob': dropout_prob,
                'lr': lr,
                'epochs': epochs, # Save total epochs planned
                'batch_size': batch_size,
                'best_accuracy': best_accuracy,
                'last_epoch_loss': avg_loss, # Loss of the epoch that achieved best accuracy
                'completed_epochs': epoch + 1, # Number of epochs completed
                # You could also save training_losses, but might be large
            }
            with open(config_path, 'w') as f:
                json.dump(config, f)


    print(f"PyTorch LSTM model training complete. Best evaluation accuracy: {best_accuracy:.4f}")
    return model, training_losses


def evaluate_model(model, X_eval_tensor, y_eval_tensor):
    """
    Evaluates a trained PyTorch LSTM model and returns accuracy.
    This version is for internal use within modeling.py (e.g., by train_model).
    It does NOT save metrics or logs to files.
    """
    device = next(model.parameters()).device # Get model's device
    model.eval()
    with torch.no_grad():
        outputs = model(X_eval_tensor.to(device))
        predicted_classes = (outputs > 0.5).squeeze().float()
        correct_predictions = (predicted_classes == y_eval_tensor.to(device).squeeze()).sum().item()
        total_predictions = y_eval_tensor.size(0)
        accuracy = correct_predictions / total_predictions if total_predictions > 0 else 0.0

    return accuracy