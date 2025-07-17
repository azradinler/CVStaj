import torch.nn as nn
import torch.nn.functional as F

# MNIST (28x28 görsellerdeki rakamları sınıflandırma) için rastgele bir model yapısı.


class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()

        # 2x2 kernellerdeki en büyük elemanı al ve 2 adım ilerle (Kerneller kesişmez)
        self.pool = nn.MaxPool2d(2, 2)


        # Dropout : Eğitim sırasında rastgele nöronların çıktılarını sıfırlar.
        self.dropout1 = nn.Dropout(0.25)
        self.dropout2 = nn.Dropout(0.5)

        # 5x5 kernelde padding 2 olursa çıkış matrisinin boyutu giriş ile aynı olur. 3x3 kernelde padding 1 olmalı.
        self.conv1 = nn.Conv2d(1, 32, 5, padding=2)

        # Batch normalization, tam anlamadım ancak her katmanın girişini normalleştiriyor ve ağı daha kararlı yapıyormuş.
        self.bn1 = nn.BatchNorm2d(32)

        self.conv2 = nn.Conv2d(32, 32, 3, padding=1)

        self.conv3 = nn.Conv2d(32, 64, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)

        self.conv4 = nn.Conv2d(64, 64, 3, padding=1)

        # Linear, Fully connected layer oluyor, giriş matrisinden belli sayıda değer üretiyor.

        self.fc1 = nn.Linear(64 * 7 * 7, 256)
        self.bn3 = nn.BatchNorm1d(256)

        # Son katmanda gelen 256 değerden 10 adet (10 rakamın olasılıklarını temsil edecek ilerde) değere dönüştürüyor.
        self.fc2 = nn.Linear(256, 10)

    def forward(self, x):
        # Her convolution layerdan sonra Relu gibi aktivasyon katmanları uygulanır, Lineerliğin bozulmasını sağlar.
        # Relu negatif değerleri 0 a çevirir, pozitif değerleri değiştirmez

        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.conv2(x))
        x = self.dropout1(self.pool(x))

        x = F.relu(self.bn2(self.conv3(x)))
        x = F.relu(self.conv4(x))
        x = self.dropout1(self.pool(x))

        # 4 boyutlu veriyi (batch_size, channels, yükseklik, genişlik) 2 boyuta çevirir, fully connected layer için.
        x = x.view(-1, 64 * 7 * 7)

        x = F.relu(self.bn3(self.fc1(x)))
        x = self.fc2(self.dropout2(x))

        return x
