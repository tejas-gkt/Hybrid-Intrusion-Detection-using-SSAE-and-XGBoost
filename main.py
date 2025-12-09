# main.py
import torch
import numpy as np
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from sklearn.preprocessing import LabelEncoder
import config
import data_utils
import models
import train_utils
import sys
import matplotlib.pyplot as plt # NEW IMPORT
import seaborn as sns # NEW IMPORT

def extract_features_in_batches(model, dataloader, device):
    # ... (function body remains the same) ...
    model.eval()
    features_list = []
    labels_list = []
    
    print(f"Processing {len(dataloader)} batches...", end=' ')
    
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(dataloader):
            inputs = inputs.to(device)
            
            features = model.forward_features(inputs)
            
            features_list.append(features.cpu().numpy())
            labels_list.append(targets.numpy())
            
            if batch_idx % 1000 == 0:
                print(".", end='', flush=True)
                
    print(" Done.")
    
    X_features = np.concatenate(features_list, axis=0)
    y_labels = np.concatenate(labels_list, axis=0)
    
    return X_features, y_labels

# NEW FUNCTION: Plot SSAE Training Loss
def plot_sae_loss(loss_history_stacked):
    """Plots the training loss curve for each SAE layer."""
    plt.figure(figsize=(10, 6))
    
    for layer, history in loss_history_stacked.items():
        epochs = range(1, len(history) + 1)
        plt.plot(epochs, history, label=layer)
        
    plt.title('SSAE Layer-wise Pre-training Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Average MSE Loss')
    plt.legend()
    plt.grid(True)
    plt.savefig('ssae_training_loss.png')
    plt.close()
    print("Saved SSAE training loss plot to ssae_training_loss.png")

# NEW FUNCTION: Plot Confusion Matrix
def plot_confusion_matrix(cm, classes, title='Confusion Matrix', cmap=plt.cm.Blues):
    """Plots the confusion matrix."""
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap=cmap, 
                xticklabels=classes, yticklabels=classes, 
                cbar=True, linewidths=.5, linecolor='black')
    
    # Ensure labels are readable
    plt.tick_params(axis='x', rotation=45)
    plt.tick_params(axis='y', rotation=0)

    plt.title(title)
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.tight_layout()
    plt.savefig('confusion_matrix.png')
    plt.close()
    print("Saved Confusion Matrix plot to confusion_matrix.png")

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # 1. Data Preprocessing
    TRAIN_FILE = '/home/hdn/Documents/NITK/sem1/IPC/proj2/train_compressed.csv'
    X_train_tensor, y_train_tensor, X_test_tensor, y_test_tensor, label_encoder =  data_utils.load_and_preprocess_separate_data(TRAIN_FILE, config.TEST_FILE)
    
    # Create DataLoaders
    train_loader, test_loader = data_utils.create_dataloaders(
        X_train_tensor, y_train_tensor, X_test_tensor, y_test_tensor
    )
    
    # 2. Initialize SSAE
    input_dim = X_train_tensor.shape[1]
    ssae = models.StackedSparseAutoencoder(input_dim, config.HIDDEN_LAYERS)
    
    # 3. Train SSAE (Layer-wise)
    # MODIFIED: CAPTURE LOSS HISTORY
    ssae, loss_history_stacked = train_utils.train_stacked_sae(ssae, train_loader, device)
    
    # NEW: Plot Training Loss
    plot_sae_loss(loss_history_stacked)

    # 4. Feature Extraction
    print("\nExtracting low-dimensional features...")
    
    print("Extracting Training Set Features:")
    X_train_features, y_train = extract_features_in_batches(ssae, train_loader, device)
    
    print("Extracting Test Set Features:")
    X_test_features, y_test = extract_features_in_batches(ssae, test_loader, device)
        
    print(f"Original Dimension: {input_dim}")
    print(f"Reduced Dimension: {X_train_features.shape[1]}")
    print(f"Training Samples: {X_train_features.shape[0]}")
    
    # 5. Train XGBoost Classifier
    print("\nTraining XGBoost Classifier...")
    
    le_xgb = LabelEncoder()
    y_train_xgb = le_xgb.fit_transform(y_train)
    
    known_classes_train = set(le_xgb.classes_)
    
    mask_known_classes = np.isin(y_test, list(known_classes_train))
    
    X_test_features_safe = X_test_features[mask_known_classes]
    y_test_safe = y_test[mask_known_classes]
    
    y_test_xgb = le_xgb.transform(y_test_safe)
    
    print(f"Encoded {len(le_xgb.classes_)} unique classes for XGBoost.")
    if len(y_test) != len(y_test_safe):
        print(f"Warning: Dropped {len(y_test) - len(y_test_safe)} test samples with labels not present in the training set.")

    # Instantiate and train XGBoost
    xgb_clf = XGBClassifier(
        objective='multi:softmax', 
        num_class=len(le_xgb.classes_),
        eval_metric='logloss',
        n_jobs=-1
    )
    
    xgb_clf.fit(X_train_features, y_train_xgb)
    
    # 6. Evaluation
    print("Evaluating model...")
    y_pred_xgb = xgb_clf.predict(X_test_features_safe)
    
    # Metrics
    acc = accuracy_score(y_test_xgb, y_pred_xgb)
    prec = precision_score(y_test_xgb, y_pred_xgb, average='macro', zero_division=0)
    rec = recall_score(y_test_xgb, y_pred_xgb, average='macro', zero_division=0)
    f1 = f1_score(y_test_xgb, y_pred_xgb, average='macro', zero_division=0)
    
    cm = confusion_matrix(y_test_xgb, y_pred_xgb)
    
    # Get original class names from the initial LabelEncoder
    original_classes = label_encoder.inverse_transform(le_xgb.classes_)

    # NEW: Plot Confusion Matrix
    plot_confusion_matrix(cm, classes=original_classes, title='XGBoost Confusion Matrix on Reduced Features')
    
    print("\n=== Experimental Results ===")
    print(f"Accuracy (AC): {acc*100:.2f}%")
    print(f"Precision (P): {prec*100:.2f}%")
    print(f"Recall (R):    {rec*100:.2f}%")
    print(f"F-measure (F): {f1*100:.2f}%")
    print("Confusion Matrix:")
    print(cm)

if __name__ == "__main__":
    main()