# models.py
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import config

class SparseAutoencoder(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super(SparseAutoencoder, self).__init__()
        # Encoder: h = s(Wx + b1) - Eq. 1
        self.encoder = nn.Linear(input_dim, hidden_dim)
        
        # Decoder: y = s(W'h + b2) - Eq. 2
        self.decoder = nn.Linear(hidden_dim, input_dim)
        
        # Tie weights (optional but common in AEs, paper doesn't explicitly forbid it)
        # self.decoder.weight.data = self.encoder.weight.data.t() 

    def forward(self, x):
        # Sigmoid activation as specified in 3.1
        encoded = torch.sigmoid(self.encoder(x))
        decoded = torch.sigmoid(self.decoder(encoded))
        return encoded, decoded

def kl_divergence(rho, rho_hat):
    """
    Calculates KL Divergence (Eq. 4)
    KL(rho || rho_hat) = rho * log(rho / rho_hat) + (1-rho) * log((1-rho) / (1-rho_hat))
    """
    # Add small epsilon to prevent log(0)
    rho_hat = torch.clamp(rho_hat, 1e-7, 1 - 1e-7) 
    term1 = rho * torch.log(rho / rho_hat)
    term2 = (1 - rho) * torch.log((1 - rho) / (1 - rho_hat))
    return term1 + term2

def loss_function(recon_x, x, hidden_activations, model):
    """
    Total Cost function (Eq. 6)
    C_sparse = MSE + Beta * KL + Lambda/2 * ||W||^2
    """
    # 1. Reconstruction Error (MSE) - Eq. 3
    # 1/2m * sum(||x - y||^2)
    # PyTorch MSELoss calculates mean, so we divide by 2 to match the paper's 1/2 factor
    mse = F.mse_loss(recon_x, x, reduction='mean') * 0.5

    # 2. Sparsity Penalty (KL Divergence) - Eq. 5
    # Average activation of hidden nodes over the batch
    rho_hat = torch.mean(hidden_activations, dim=0)
    kl_div = torch.sum(kl_divergence(config.SPARSITY_PARAM, rho_hat))
    sparsity_term = config.SPARSITY_WEIGHT * kl_div

    # 3. Weight Decay (L2 Regularization)
    # Handled by optimizer in PyTorch usually, but implementing explicitly for completeness with Eq. 6
    l2_reg = 0.0
    for param in model.parameters():
        l2_reg += torch.norm(param, 2) ** 2
    
    weight_decay_term = (config.WEIGHT_DECAY / 2.0) * l2_reg

    return mse + sparsity_term + weight_decay_term

class StackedSparseAutoencoder(nn.Module):
    def __init__(self, input_dim, layer_sizes):
        super(StackedSparseAutoencoder, self).__init__()
        self.saes = nn.ModuleList()
        self.trained_encoders = nn.ModuleList()
        
        # Create individual SAEs for layer-wise training
        current_dim = input_dim
        for hidden_dim in layer_sizes:
            self.saes.append(SparseAutoencoder(current_dim, hidden_dim))
            current_dim = hidden_dim

    def forward_features(self, x):
        """
        Passes input through the stack of trained encoders to get low-dimensional features.
        """
        out = x
        for sae in self.saes:
            # We only use the encoder part of each SAE
            out = torch.sigmoid(sae.encoder(out))
        return out