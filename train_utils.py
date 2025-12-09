import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import config
import models
from torch.utils.data import TensorDataset, DataLoader

def get_next_layer_inputs(encoder, dataloader, device):
    """
    Passes the entire dataset through the encoder in batches and returns the encoded features
    as a single PyTorch Tensor on the CPU.
    """
    encoder.eval()
    features_list = []
    
    with torch.no_grad():
        for inputs, _ in dataloader:
            inputs = inputs.to(device)
            # Get encoded features
            features = torch.sigmoid(encoder(inputs))
            features_list.append(features.cpu().numpy())
            
    # Concatenate all batches into a single numpy array (on CPU!)
    X_features_numpy = np.concatenate(features_list, axis=0)
    
    # Convert back to a float tensor on CPU
    return torch.FloatTensor(X_features_numpy) 

def train_sae(sae_layer, dataloader, device):
    """
    Trains a single Sparse Autoencoder layer and returns the loss history.
    """
    criterion = nn.MSELoss()
    optimizer = optim.Adam(sae_layer.parameters(), lr=config.LEARNING_RATE, weight_decay=config.WEIGHT_DECAY)
    
    sae_layer.train()
    loss_history = []
    
    for epoch in range(config.EPOCHS):
        total_loss = 0
        for batch_idx, (inputs, _) in enumerate(dataloader):
            inputs = inputs.to(device)
            
            # Forward pass
            # FIX IS HERE: Swapped order to (encoded, outputs) to match models.py return
            encoded, outputs = sae_layer(inputs)
            
            # Calculate Reconstruction Loss (MSE between Output and Input)
            reconstruction_loss = criterion(outputs, inputs)
            
            # Calculate Sparsity Loss (KL Divergence on Encoded activations)
            rho_hat = torch.mean(encoded, dim=0) # Average activation over the batch
            rho = config.SPARSITY_PARAM
            
            # Clamp rho_hat to avoid log(0)
            rho_hat = torch.clamp(rho_hat, 1e-6, 1 - 1e-6)
            
            sparsity_loss = rho * torch.log(rho / rho_hat) + (1 - rho) * torch.log((1 - rho) / (1 - rho_hat))
            sparsity_loss = sparsity_loss.sum()
            
            # Total Loss
            loss = reconstruction_loss + config.SPARSITY_WEIGHT * sparsity_loss
            
            # Backward and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            
        epoch_loss = total_loss / len(dataloader)
        loss_history.append(epoch_loss)
        print(f"Epoch {epoch + 1}/{config.EPOCHS}, Loss: {epoch_loss:.6f}")

    return loss_history

def train_stacked_sae(ssae, initial_train_loader, device):
    """
    Performs layer-wise pre-training of the Stacked Sparse Autoencoder.
    """
    current_loader = initial_train_loader 
    loss_history_stacked = {}
    
    for i, sae_layer in enumerate(ssae.saes):
        # 1. Pre-train current SAE layer
        # Access dimensions via 'sae_layer.encoder'
        current_sae = models.SparseAutoencoder(
            sae_layer.encoder.in_features, 
            sae_layer.encoder.out_features
        )
        current_sae.to(device)
        
        # 2. Train the current layer
        print(f"\n=== Pre-training Layer {i + 1} ===")
        print(f"Training SAE: Input {sae_layer.encoder.in_features} -> Hidden {sae_layer.encoder.out_features}")
        
        # Capture loss history
        layer_loss_history = train_sae(current_sae, current_loader, device) 
        loss_history_stacked[f"Layer_{i+1}"] = layer_loss_history
        
        # Update the trained weights in the Stacked Autoencoder model
        ssae.saes[i] = current_sae
        
        # 3. Prepare inputs for the next layer 
        if i < len(ssae.saes) - 1:
            print(f"Extracting features for Layer {i + 2} input (Batch-wise)...", end=' ')
            
            X_features_cpu = get_next_layer_inputs(current_sae.encoder, current_loader, device)
            
            # Create dummy targets for dataloader compatibility
            dummy_targets = torch.zeros(X_features_cpu.shape[0], dtype=torch.long)
            new_dataset = TensorDataset(X_features_cpu, dummy_targets)
            
            current_loader = DataLoader(
                new_dataset, 
                batch_size=config.BATCH_SIZE, 
                shuffle=True
            )
            print("Done.")
            
    print("\nLayer-wise pre-training completed.")
    return ssae, loss_history_stacked