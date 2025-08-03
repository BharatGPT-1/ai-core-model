import torch
from torch.utils.data import DataLoader
from core.model import TransformerModel
from configs.base import Config

def train():
    # Load config
    config = Config()
    
    # Initialize model
    model = TransformerModel(
        vocab_size=config.vocab_size,
        d_model=config.d_model,
        n_layers=config.n_layers,
        n_heads=config.n_heads,
        d_ff=config.d_ff,
        dropout=config.dropout
    ).to(config.device)
    
    # Optimizer and loss
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.lr)
    criterion = nn.CrossEntropyLoss()
    
    # Training loop
    for epoch in range(config.epochs):
        model.train()
        for batch in train_loader:
            inputs, targets = batch
            inputs, targets = inputs.to(config.device), targets.to(config.device)
            
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs.view(-1, config.vocab_size), 
                           targets.view(-1))
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            
        # Validation
        val_loss = evaluate(model, val_loader, config)
        print(f"Epoch {epoch}: Train Loss {loss.item():.4f}, Val Loss {val_loss:.4f}")
        
        # Save checkpoint
        torch.save(model.state_dict(), f"checkpoints/model_epoch{epoch}.pt")