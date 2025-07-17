import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import torch.optim as optim
import os

from simple_cnn import SimpleCNN

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")

# Veriyi pytorchun beklediği hale dönüştürür ve -1 ile 1 arasında normalize eder (
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

# data klasörü yoksa download = True yapılmalı.
train_set = torchvision.datasets.MNIST(root='./data', train=True, download=False, transform=transform)
train_loader = torch.utils.data.DataLoader(train_set, batch_size=64, shuffle=True)

test_set = torchvision.datasets.MNIST(root='./data', train=False, download=False, transform=transform)
test_loader = torch.utils.data.DataLoader(test_set, batch_size=1000, shuffle=False)

def evaluate_model(model, data_loader, criterion, device):
    # Modeli hesap moduna geçirir, dropout katmanlarını kapatır.
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0

    # Gradyan hesaplama mekanizması olmadan çalışır (Gradyan hesaplama sadece eğitimde gerekli)
    with torch.no_grad():
        for inputs, labels in data_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            total_loss += loss.item()

            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    avg_loss = total_loss / len(data_loader)
    accuracy = 100 * correct / total
    return avg_loss, accuracy


def save_model(model, filepath):
    torch.save(model.state_dict(), filepath)
    print(f"Model kaydedildi: {filepath}")


def load_model(filepath):
    modelx = torch.load(filepath, weights_only= True)
    return modelx


model = SimpleCNN().to(device)

# Modelin öğretmeni gibi düşünülebilir, ne kadar yanlış olduğunu değerlendirir.
criterion = nn.CrossEntropyLoss()

# Modelin gelen yanıttan nasıl iyileşeceğini belirleyen yöntem.
optimizer = optim.Adam(model.parameters(), lr=0.001)

num_epochs = 25
patience = 4
best_val_accuracy = 0.0
patience_counter = 0

train_losses = []
val_losses = []
train_accuracies = []
val_accuracies = []

if not os.path.exists('models'):
    os.makedirs('models')

print("Training Starts...")
print("************")

for epoch in range(num_epochs):
    # Modeli eğitim moduna geçirir, dropout katmanları aktifleştirilir.
    model.train()

    running_loss = 0.0
    correct = 0
    total = 0

    for inputs, labels in train_loader:
        inputs, labels = inputs.to(device), labels.to(device)

        # Bir önceki adımdan kalan gradyanları sıfırlar.
        optimizer.zero_grad()

        # Modelde tanımlı forward metodunu çalıştırır.
        outputs = model(inputs)

        # Kaybı hesaplar.
        loss = criterion(outputs, labels)

        # Geri yayılım yapar, gradyanları hesaplar
        loss.backward()

        # Model parametrelerini günceller
        optimizer.step()

        running_loss += loss.item()

        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    train_loss = running_loss / len(train_loader)
    train_accuracy = 100 * correct / total

    val_loss, val_accuracy = evaluate_model(model, test_loader, criterion, device)

    train_losses.append(train_loss)
    val_losses.append(val_loss)
    train_accuracies.append(train_accuracy)
    val_accuracies.append(val_accuracy)

    print(f"Epoch [{epoch + 1}/{num_epochs}]")
    print(f"  Train Loss: {train_loss:.4f}, Train Acc: {train_accuracy:.2f}%")
    print(f"  Val Loss: {val_loss:.4f}, Val Acc: {val_accuracy:.2f}%")

    if val_accuracy > best_val_accuracy:
        best_val_accuracy = val_accuracy
        patience_counter = 0

        # En iyi modeli kaydet
        save_model(model, './models/best_model.pt')
        print(f"Yeni en iyi model kaydedildi, Doğruluk: {best_val_accuracy:.2f}%")
    else:
        patience_counter += 1
        print(f"  Sayaç: {patience_counter}/{patience}")

    print("************")

    if patience_counter >= patience:
        print(f"Eğitim duruyor, {patience} epoch boyunca gelişme olmadı.")
        break

print("Eğitim tamamlandı")