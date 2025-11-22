import torch
from tqdm import tqdm

def train_model(model, train_loader, criterion, optimizer, device, val_loader=None, num_epochs=10, save_path=None):
    """
    Function to train the model, including validation if a validation loader is provided.
    """
    history = {
        "train_losses": [],
        "val_losses": [],
        "accuracy": []
    }

    for epoch in range(num_epochs):
        model.train()  # Set model to training mode
        running_loss = 0.0
        correct = 0
        total = 0

        # Training loop
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)

            # Zero the parameter gradients
            optimizer.zero_grad()

            # Forward pass
            outputs = model(inputs)
            loss = criterion(outputs, labels)

            # Backward pass and optimization
            loss.backward()
            optimizer.step()

            # Update metrics
            running_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        # Calculate training accuracy and loss
        accuracy = 100 * correct / total
        history["train_losses"].append(running_loss / len(train_loader))
        history["accuracy"].append(accuracy)

        # If validation loader is provided, calculate validation loss
        if val_loader:
            model.eval()  # Set model to evaluation mode
            val_loss = 0.0
            correct_val = 0
            total_val = 0
            with torch.no_grad():
                for inputs, labels in val_loader:
                    inputs, labels = inputs.to(device), labels.to(device)
                    outputs = model(inputs)
                    loss = criterion(outputs, labels)
                    val_loss += loss.item()
                    _, predicted = torch.max(outputs, 1)
                    total_val += labels.size(0)
                    correct_val += (predicted == labels).sum().item()
            val_accuracy = 100 * correct_val / total_val
            history["val_losses"].append(val_loss / len(val_loader))
            print(f"Epoch [{epoch+1}/{num_epochs}], "
                  f"Train Loss: {running_loss / len(train_loader):.4f}, "
                  f"Train Accuracy: {accuracy:.2f}%, "
                  f"Val Loss: {val_loss / len(val_loader):.4f}, "
                  f"Val Accuracy: {val_accuracy:.2f}%")
        else:
            print(f"Epoch [{epoch+1}/{num_epochs}], "
                  f"Train Loss: {running_loss / len(train_loader):.4f}, "
                  f"Train Accuracy: {accuracy:.2f}% (no validation)")

        # Save model checkpoint after each epoch
        if save_path:
            torch.save(model.state_dict(), f"{save_path}_model_epoch_{epoch+1}.pth")

    return model, history
