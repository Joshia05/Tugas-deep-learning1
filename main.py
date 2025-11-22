import torch
import matplotlib.pyplot as plt
import seaborn as sns
import os
from sklearn.metrics import confusion_matrix
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from models.vit_model import ViTModel
from models.deit_model import DeiTModel
from utils.eval import evaluate_model
from utils.train import train_model

import matplotlib.pyplot as plt

def visualize_comparison(vit_loss, vit_accuracy, deit_loss, deit_accuracy, output_dir):
    """
    Visualisasikan perbandingan antara model ViT dan DeiT berdasarkan loss dan accuracy.
    """
    # Plot loss
    plt.figure(figsize=(8, 6))
    plt.plot(['ViT', 'DeiT'], [vit_loss, deit_loss], marker='o', label='Loss')
    plt.title('Model Comparison: Loss')
    plt.xlabel('Model')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig(os.path.join(output_dir, 'model_loss_comparison.png'))
    plt.close()

    # Plot accuracy
    plt.figure(figsize=(8, 6))
    plt.plot(['ViT', 'DeiT'], [vit_accuracy, deit_accuracy], marker='o', label='Accuracy')
    plt.title('Model Comparison: Accuracy')
    plt.xlabel('Model')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.savefig(os.path.join(output_dir, 'model_accuracy_comparison.png'))
    plt.close()

# Fungsi untuk visualisasi learning curve
def plot_learning_curve(train_losses, val_losses, accuracy, output_dir):
    """
    Visualisasikan learning curve untuk training loss, validation loss, dan akurasi.
    """
    plt.figure(figsize=(12, 5))
    
    # Plot kurva loss
    plt.subplot(1, 2, 1)
    plt.plot(train_losses, label='Training Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.title('Loss Curve')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()

    # Plot kurva akurasi
    plt.subplot(1, 2, 2)
    plt.plot(accuracy, label='Accuracy')
    plt.title('Accuracy Curve')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'learning_curve.png'))
    plt.close()

# Fungsi untuk visualisasi confusion matrix
# Fungsi untuk visualisasi confusion matrix
def plot_confusion_matrix(y_true, y_pred, output_dir):
    """
    Visualisasikan confusion matrix untuk evaluasi.
    """
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='g', cmap='Blues', xticklabels=['Class 1', 'Class 2', 'Class 3', 'Class 4', 'Class 5'], yticklabels=['Class 1', 'Class 2', 'Class 3', 'Class 4', 'Class 5'])
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    plt.savefig(os.path.join(output_dir, 'confusion_matrix.png'))
    plt.close()

# Update fungsi utama untuk mengumpulkan `y_true` dan `y_pred`
def main():
    # Tentukan device (GPU jika tersedia, jika tidak, CPU)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Tentukan transformasi gambar
    transform = transforms.Compose([ 
        transforms.Resize((224, 224)),  # Ukuran gambar sesuai input model
        transforms.ToTensor(),  # Konversi gambar menjadi tensor
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # Normalisasi sesuai pre-trained model
    ])

    # Muat dataset train dan test dari folder yang sudah ada
    train_dataset = datasets.ImageFolder('data/train', transform=transform)
    test_dataset = datasets.ImageFolder('data/test', transform=transform)

    # DataLoader untuk training dan testing
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

    # Inisialisasi model ViT dan DeiT
    vit_model = ViTModel(num_classes=len(train_dataset.classes)).to(device)
    deit_model = DeiTModel(num_classes=len(train_dataset.classes)).to(device)

    # Tentukan loss function
    criterion = torch.nn.CrossEntropyLoss()

    # Optimizer untuk ViT dan DeiT
    vit_optimizer = torch.optim.Adam(vit_model.parameters(), lr=1e-4)
    deit_optimizer = torch.optim.Adam(deit_model.parameters(), lr=1e-4)

    # Memuat checkpoint ViT model dan optimizer jika ada
    try:
        vit_model.load_state_dict(torch.load('vit_model_model_epoch_10.pth'))
        vit_optimizer.load_state_dict(torch.load('vit_optimizer_epoch_10.pth'))
        print("Loaded ViT model and optimizer from checkpoint.")
    except FileNotFoundError:
        print("No ViT checkpoint found, skipping ViT evaluation.")
    
    # Evaluasi Model ViT sebelum melatih DeiT
    print("Evaluating ViT model before DeiT training...")
    vit_loss, vit_accuracy = evaluate_model(vit_model, test_loader, criterion, device)
    print(f"ViT - Loss: {vit_loss:.4f}, Accuracy: {vit_accuracy:.4f}")

    # Memuat checkpoint DeiT model dan optimizer jika ada
    try:
        deit_model.load_state_dict(torch.load('deit_model_model_epoch_10.pth'))
        deit_optimizer.load_state_dict(torch.load('deit_optimizer_epoch_10.pth'))
        print("Loaded DeiT model and optimizer from checkpoint.")
    except FileNotFoundError:
        print("No DeiT checkpoint found. Skipping DeiT training.")
        # Jika checkpoint tidak ditemukan, maka lakukan pelatihan untuk DeiT
        print("Training DeiT model...")
        deit_model, deit_history = train_model(deit_model, train_loader, criterion, deit_optimizer, device, num_epochs=10, save_path='deit_model')

    # Evaluasi Model DeiT
    print("Evaluating DeiT model...")
    deit_loss, deit_accuracy = evaluate_model(deit_model, test_loader, criterion, device)
    print(f"DeiT - Loss: {deit_loss:.4f}, Accuracy: {deit_accuracy:.4f}")

    # Visualisasi perbandingan
    print("Visualizing comparison between ViT and DeiT models...")
    output_dir = 'output'
    visualize_comparison(vit_loss, vit_accuracy, deit_loss, deit_accuracy, output_dir)

    # Visualisasi Learning Curve untuk DeiT
    print("Visualizing learning curve for DeiT model...")
    if 'deit_history' in locals():
        plot_learning_curve(deit_history['train_losses'], deit_history['val_losses'], deit_history['accuracy'], output_dir)
    else:
        print("No training history for DeiT model, skipping learning curve visualization.")
    
    # Visualisasi Confusion Matrix untuk DeiT
    print("Visualizing confusion matrix for DeiT model...")
    deit_y_true = []  # Ganti dengan data true labels
    deit_y_pred = []  # Ganti dengan data predicted labels
    for inputs, labels in test_loader:
        inputs, labels = inputs.to(device), labels.to(device)
        outputs = deit_model(inputs)
        _, predicted = torch.max(outputs, 1)
        deit_y_true.extend(labels.cpu().numpy())
        deit_y_pred.extend(predicted.cpu().numpy())

    plot_confusion_matrix(deit_y_true, deit_y_pred, output_dir)

if __name__ == '__main__':
    main()
