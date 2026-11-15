import torch
import torch.nn as nn
import torch.optim as optim
from model import STGNNModel

def train_model(model, train_loader, adj_mx, epochs=10, learning_rate=0.001, device='cpu'):
    # Move the neural network and the spatial adjacency matrix to the target device
    # All tensors need to be on the same devie
    model.to(device)
    adj_mx = adj_mx.to(device)

    # We define the loss function, in this case, we use Mean Squared Error (MSE)
    # This is standard for continuous regression tasks
    # MSEL looks at the model's predicted future traffic speed, subtracts the actual future
    # traffic speed, and squares the difference. Squaring heavily penalizes massive mistakes over small ones 
    criterion = nn.MSELoss()

    # We define the optimizer, Adam (Adaptive Movement Estimation). Adam adaptively adjusts the learning rate during training
    # The optimizer is the engine that actually changes the model's weights. 
    # Adam is standard because it dynamically accelerates leraning on simple parameters and slows down on complex ones,
    # preventing the model from getting stuck in local minimums.
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # We set the training mode. model.train() flags the network to enable specific behaviors
    # and ensures pytorch begins activley tracking gradients for backpropogation 
    model.train()
    print(f'Training on {device} for {epochs} epochs')

    # Epoch loop
    # An epoch represents one compelte pass ove rthe entire training dataset.
    # Models need multiple passes to incrementally learn the complex patterns without memorizing the data.
    for epoch in range(epochs):
        total_loss = 0

        # Batch loop: The DataLoader feeds memory-efficient chunks of 64 sequences to the model
        for batch_ix, (x_batch, y_batch) in enumerate(train_loader):

            # Move the batch data to the target device
            x_batch = x_batch.to(device)
            y_batch = y_batch.to(device)

            # Zero gradients: Pytorch accumulates (adds up) gradients by default
            # Without wiping them to zero at the start of every single batch, the model will apply the mathematically corrections from batch 1 onto batch 2.
            # This crashes the learning process
            optimizer.zero_grad()

            # Forward Pass: Push historical data (x_batch) and the spatial graph through the model architecture.
            predictions = model(x_batch, adj_mx)

            # Calculate Loss: Measure how far the predictions are form the actual future (y_batch).
            loss = criterion(predictions, y_batch)

            # Backward Pass: The core backpropogation. Computes the partial derivative of the loss with respect
            # to every single learnable weight in the network, figuring out eaclty which parameters casused the 
            # bad predictions and how they need to change. 
            loss.backward()

            # Optimizer Step: This applies the corrections calculated by .backward(), officially adjusting
            # the model's weights to make it slightly more accurate for the next batch 
            optimizer.step()

            # Accumulate the batch loss to calculate the average for the epoch
            # .item() extracts the raw Python float from the PyTorch tensor to prevent memory leaks
            total_loss += loss.item()

        # End of Epoch Reporting: Calculate and print the average loss at the end of the epoch
        avg_loss = total_loss / len(train_loader)
        print(f'Epoch {epoch + 1}/{epochs} | Training Loss: {avg_loss:.4f}')
    print('Training complete')
    return model