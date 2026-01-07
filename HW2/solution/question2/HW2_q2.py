from utils_w_masking import load_rnacompete_data, masked_mse_loss, masked_spearman_correlation, configure_seed
from torch.utils.data import DataLoader
from HW2.solution.question2.TrasnformerEncoder import RNATransformer
import torch
import itertools

configure_seed(42)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

train_d = load_rnacompete_data('RBFOX1', 'train')
val_d   = load_rnacompete_data('RBFOX1', 'val')
test_d  = load_rnacompete_data('RBFOX1', 'test')

train_loader = DataLoader(train_d, batch_size=64, shuffle=True)
val_loader   = DataLoader(val_d, batch_size=256, shuffle=False)
test_loader  = DataLoader(test_d, batch_size=256, shuffle=False)


"""Training and Evaluation Functions"""
def train_epoch(loader, model, optimizer):
    model.train()
    total_loss = 0.0
    for sequence, affinity, mask in loader:
        if sequence.dim() == 3: 
            is_ambiguous = (sequence.max(dim=-1).values == sequence.min(dim=-1).values)
            sequence = sequence.argmax(dim=-1)
            sequence[is_ambiguous] = 4

        sequence = sequence.to(device).long()
        affinity = affinity.to(device)
        mask = mask.to(device)

        optimizer.zero_grad()
        outputs = model(sequence).squeeze()
        loss = masked_mse_loss(outputs, affinity, mask)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(loader)

def evaluate(loader, model):
    model.eval()
    total_loss = 0.0
    spearman_vals = []
    with torch.no_grad():
        for sequence, affinity, mask in loader:
            if sequence.dim() == 3: 
                is_ambiguous = (sequence.max(dim=-1).values == sequence.min(dim=-1).values)
                sequence = sequence.argmax(dim=-1)
                sequence[is_ambiguous] = 4  

            sequence = sequence.to(device).long()
            affinity = affinity.to(device)
            mask = mask.to(device)

            outputs = model(sequence).squeeze()
            total_loss += masked_mse_loss(outputs, affinity, mask).item()
            spearman_vals.append(masked_spearman_correlation(outputs, affinity, mask).item())

    avg_loss = total_loss / len(loader)
    avg_spearman = sum(spearman_vals) / max(len(spearman_vals), 1)
    return avg_loss, avg_spearman


def run_grid_search_transformer():
    # Define hyperparameter grid for Transformer
    param_grid = {
        'learning_rate': [1e-3, 1e-4],
        'd_model': [64, 128],
        'nhead': [4, 8],
        'num_layers': [2, 3],
        'dim_feedforward': [128, 256],
        'dropout': [0.1, 0.2]
    }
    
    num_epochs = 20
    best_spearman = -float('inf')
    best_params = None
    results = []
    
    # Generate all combinations
    keys = list(param_grid.keys())
    combinations = list(itertools.product(*param_grid.values()))
    
    print(f"Total Transformer configurations to test: {len(combinations)}")
    
    for i, values in enumerate(combinations):
        params = dict(zip(keys, values))
        
        # Ensure d_model is divisible by nhead
        if params['d_model'] % params['nhead'] != 0:
            print(f"\n[{i+1}/{len(combinations)}] Skipping (d_model not divisible by nhead): {params}")
            continue
            
        print(f"\n[{i+1}/{len(combinations)}] Testing: {params}")
        
        # Create model with current hyperparameters
        model = RNATransformer(
            vocab_size=5,
            d_model=params['d_model'],
            nhead=params['nhead'],
            num_layers=params['num_layers'],
            dim_feedforward=params['dim_feedforward'],
            output_dim=1,
            dropout=params['dropout']
        ).to(device)
        
        optimizer = torch.optim.Adam(model.parameters(), lr=params['learning_rate'])
        
        # Train for num_epochs
        best_val_spearman_epoch = -float('inf')
        for epoch in range(num_epochs):
            train_loss = train_epoch(train_loader, model, optimizer)
            val_loss, val_spearman = evaluate(val_loader, model)
            
            if val_spearman > best_val_spearman_epoch:
                best_val_spearman_epoch = val_spearman
            
            if (epoch + 1) % 5 == 0:
                print(f"  Epoch {epoch+1}: Train Loss={train_loss:.4f}, Val Loss={val_loss:.4f}, Val Spearman={val_spearman:.4f}")
        
        results.append((params, best_val_spearman_epoch))
        print(f"  Best Val Spearman: {best_val_spearman_epoch:.4f}")
        
        # Track overall best
        if best_val_spearman_epoch > best_spearman:
            best_spearman = best_val_spearman_epoch
            best_params = params.copy()
    
    # Print summary
    print("\n" + "="*60)
    print("TRANSFORMER GRID SEARCH RESULTS SUMMARY")
    print("="*60)
    for params, spearman in sorted(results, key=lambda x: x[1], reverse=True)[:5]:
        print(f"Spearman={spearman:.4f} | {params}")
    
    print("\n" + "="*60)
    print(f"BEST TRANSFORMER HYPERPARAMETERS (Val Spearman: {best_spearman:.4f}):")
    print("="*60)
    for k, v in best_params.items():
        print(f"  {k}: {v}")
    
    return best_params, best_spearman


if __name__ == "__main__":    
    print("\n\n" + "="*60)
    print("TRANSFORMER GRID SEARCH")
    print("="*60)
    best_transformer_params, best_transformer_spearman = run_grid_search_transformer()
    
    print("\n\n" + "="*60)
    print("FINAL COMPARISON")
    print("="*60)
    print(f"Best Transformer Spearman: {best_transformer_spearman:.4f}")