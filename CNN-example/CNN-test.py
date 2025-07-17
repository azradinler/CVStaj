import torch
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt

from simple_cnn import SimpleCNN

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")

# Transform tanımı (eğitimde kullanılan ile aynı olmalı)
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

# Test dataseti yükle
test_set = torchvision.datasets.MNIST(root='./data', train=False, download=True, transform=transform)
test_loader = torch.utils.data.DataLoader(test_set, batch_size=1000, shuffle=False)



def detailed_test(model, test_loader, device):
    model.eval()
    correct = 0
    total = 0

    print("Test başlıyor...")

    with torch.no_grad():
        for batch_idx, (images, labels) in enumerate(test_loader):
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)

            total += labels.size(0)
            correct += (predicted == labels).sum().item()


            # İlerleme gösterimi
            if batch_idx % 2 == 0:
                current_acc = 100 * correct / total
                print(f"Batch {batch_idx + 1}/{len(test_loader)}, "
                      f"Accuracy: {current_acc:.2f}%")

    final_accuracy = 100 * correct / total
    print(f"\n{'=' * 50}")
    print(f"TEST SONUÇLARI:")
    print(f"{'=' * 50}")
    print(f"Örnek Sayısı: {total}")
    print(f"Doğru tahmin: {correct}")
    print(f"Yanlış tahmin: {total - correct}")
    print(f"Test Doğruluğu: {final_accuracy:.2f}%")
    print(f"{'=' * 50}")

    return

def show_misclassified_examples(model, test_loader, device, num_examples=10):

    # ilk 10 yanlış tahmini göster
    model.eval()
    misclassified = []

    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)

            wrong_idx = (predicted != labels).nonzero(as_tuple=True)[0]

            for idx in wrong_idx[:num_examples]:
                misclassified.append({
                    'image': images[idx].cpu(),
                    'predicted': predicted[idx].cpu().item(),
                    'actual': labels[idx].cpu().item()
                })

            if len(misclassified) >= num_examples:
                break

    if misclassified:
        fig, axes = plt.subplots(2, 5, figsize=(12, 6))
        fig.suptitle('Yanlış Sınıflandırılan Örnekler', fontsize=16)

        for i, example in enumerate(misclassified[:10]):
            row = i // 5
            col = i % 5

            # Görüntüyü normalize et (gösterim için)
            img = example['image'].squeeze()
            img = (img + 1) / 2  # Baştaki normalizasyonun tersi

            axes[row, col].imshow(img, cmap='gray')
            axes[row, col].set_title(f'Gerçek: {example["actual"]}\nTahmin: {example["predicted"]}')
            axes[row, col].axis('off')

        plt.tight_layout()
        #plt.savefig('misclassified_examples.png', dpi=150, bbox_inches='tight')
        plt.show()


def main():
    model = SimpleCNN()

    #  Modeli yükle
    model_path = 'models/mnist_model.pt'
    model.load_state_dict(torch.load(model_path, map_location= device))
    model.to(device)
    model.eval()

    detailed_test(model, test_loader, device)
    show_misclassified_examples(model, test_loader, device)


if __name__ == "__main__":
    main()