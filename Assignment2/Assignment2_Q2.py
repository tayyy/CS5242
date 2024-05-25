import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
from torchvision import transforms
import matplotlib.pyplot as plt
import numpy as np

# Load the MNIST dataset
transform = transforms.Compose([transforms.ToTensor()])

train_dataset = torchvision.datasets.MNIST(root='./data', train=True, transform=transform, download=True)
test_dataset = torchvision.datasets.MNIST(root='./data', train=False, transform=transform, download=True)
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=128, shuffle=True)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=128, shuffle=False)

# Define the architecture class
class Autoencoder(nn.Module):
    def __init__(self, encoding_dim):
        super(Autoencoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Flatten(),
            nn.Linear(28 * 28, 196),
            nn.ReLU(),
            nn.Linear(196, 49),
            nn.ReLU(),
            nn.Linear(49, encoding_dim),
            nn.ReLU()
        )
        self.decoder = nn.Sequential(
            nn.Linear(encoding_dim, 49),
            nn.ReLU(),
            nn.Linear(49, 196),
            nn.ReLU(),
            nn.Linear(196, 28 * 28),
            nn.ReLU()
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x

device = torch.device("cpu")
# Define the autoencoders for λ = 2 and λ = 32
autoencoder_lambda_2 = Autoencoder(encoding_dim=2).to(device)
autoencoder_lambda_32 = Autoencoder(encoding_dim=32).to(device)

# Define the loss function (Mean Absolute Error) and optimizer (Adam)
criterion = nn.L1Loss()  # Mean Absolute Error
optimizer_lambda_2 = optim.Adam(autoencoder_lambda_2.parameters(), lr=0.001)
optimizer_lambda_32 = optim.Adam(autoencoder_lambda_32.parameters(), lr=0.001)

# Training loop for λ = 2
num_epochs = 30
losses_lambda_2 = []

for epoch in range(num_epochs):
    total_loss = 0
    for data in train_loader:
        inputs, _ = data        
        inputs = inputs.to(device)
        inputs = inputs.view(inputs.size(0), -1)  # Flatten the input
        optimizer_lambda_2.zero_grad()
        outputs = autoencoder_lambda_2(inputs)
        loss = criterion(outputs, inputs)
        loss.backward()
        optimizer_lambda_2.step()
        total_loss += loss.item()
    losses_lambda_2.append(total_loss / len(train_loader))
    print(f'lamda 2: Epoch {epoch} Loss: {total_loss / len(train_loader):.4f}')

# Training loop for λ = 32
losses_lambda_32 = []

for epoch in range(num_epochs):
    total_loss = 0
    for data in train_loader:
        inputs, _ = data
        inputs = inputs.to(device)
        inputs = inputs.view(inputs.size(0), -1)  # Flatten the input
        optimizer_lambda_32.zero_grad()
        outputs = autoencoder_lambda_32(inputs)
        loss = criterion(outputs, inputs)
        loss.backward()
        optimizer_lambda_32.step()
        total_loss += loss.item()
    losses_lambda_32.append(total_loss / len(train_loader))
    print(f'lamda 32: Epoch {epoch} Loss: {total_loss / len(train_loader):.4f}')

# Convert test_images to tensor and flatten for later use
# sample_input = torch.tensor(test_images[0:1].reshape(1, -1))
# Display sample input and reconstructed output images for both scenarios
test_dataset = torchvision.datasets.MNIST(root='./data', train=False, transform=transform, download=True)

# Convert test_images to tensor and flatten for later use
sample_loader = torch.utils.data.DataLoader(test_dataset, batch_size=1, shuffle=True)
sample_images, _ = next(iter(sample_loader))
sample_images = sample_images.to(device)

# Pass the sample images through the autoencoders
sample_output_lambda_2 = autoencoder_lambda_2(sample_images.view(1, -1))
sample_output_lambda_32 = autoencoder_lambda_32(sample_images.view(1, -1))

# Plot training loss vs. epochs for both scenarios
plt.figure()
############### lambda 2
plt.subplot(2, 3, 1)
plt.plot(losses_lambda_2, label='Training Loss (λ=2)')
plt.xlabel('Epochs')
plt.ylabel('Mean Absolute Error')
plt.legend()
plt.title('Autoencoder Loss (λ=2)')

plt.subplot(2, 3, 2)
plt.imshow(sample_images[0][0].numpy(), cmap='gray')
plt.title('Sample Input')

plt.subplot(2, 3, 3)
plt.imshow(sample_output_lambda_2.view(28, 28).detach().numpy(), cmap='gray')
plt.title('Reconstructed (λ=2)')

############### lambda 32
plt.subplot(2, 3, 4)
plt.plot(losses_lambda_32, label='Training Loss (λ=32)')
plt.xlabel('Epochs')
plt.ylabel('Mean Absolute Error')
plt.legend()
plt.title('Autoencoder Loss (λ=32)')

plt.subplot(2, 3, 5)
plt.imshow(sample_images[0][0].numpy(), cmap='gray')
plt.title('Sample Input')

plt.subplot(2, 3, 6)
plt.imshow(sample_output_lambda_32.view(28, 28).detach().numpy(), cmap='gray')
plt.title('Reconstructed (λ=32)')

plt.tight_layout()
plt.show()
