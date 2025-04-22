import torch
from PIL import Image
import torchvision.transforms as transforms
from pathlib import Path
import torch.nn as nn 
import torch.nn.functional as F
from torchvision.transforms import CenterCrop
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import torch.optim as optim
from sklearn.model_selection import train_test_split

# Inspired by:
# https://pyimagesearch.com/2021/11/08/u-net-training-image-segmentation-models-in-pytorch/


class Block(nn.Module):
    def __init__(self, inChannels, outChannels):
        super().__init__()
        self.conv1 = nn.Conv2d(inChannels, outChannels, kernel_size=3)
        self.conv2 = nn.Conv2d(outChannels, outChannels, kernel_size=3)

    def forward(self, x):
        return self.conv2(F.relu(self.conv1(x)))


class Encoder(nn.Module):
    # Input: (1,1,256,256)
    # Block1 → (1,64,256,256) → Pool → (1,64,128,128)
    # Block2 → (1,128,128,128) → Pool → (1,128,64,64)
    # Block3 → (1,256,64,64) → Pool → (1,256,32,32)
    # Block4 → (1,512,32,32) → Pool → (1,512,16,16)
    def __init__(self, channels=(1, 64, 128, 256, 512, 1024)):
        super().__init__()
        self.encBlocks = nn.ModuleList(
            [Block(channels[i], channels[i + 1])
             for i in range(len(channels) - 1)])
        self.pool = nn.MaxPool2d(kernel_size=2)

    def forward(self, x):
        features = []
        for block in self.encBlocks:
            x = block(x)
            features.append(x)
            x = self.pool(x)
        return features


class Decoder(nn.Module):
    # Upconv1: (1,512,16,16) → (1,256,32,32)
    # Concat → (1,512,32,32) → Block → (1,256,32,32)
    # Upconv2: → (1,128,64,64)
    # Concat → (1,256,64,64) → Block → (1,128,64,64)
    # Upconv3: → (1,64,128,128)
    # Concat → (1,128,128,128) → Block → (1,64,128,128)
    # Upconv4: → (1,32,256,256)
    # Final Conv: → (1,2,256,256)
    def __init__(self, channels=(1024, 512, 256, 128, 64)):
        super().__init__()
        self.upconvs = nn.ModuleList([
            nn.ConvTranspose2d(channels[i], channels[i+1], kernel_size=2, stride=2)
            for i in range(len(channels) - 1)
        ])
        self.dec_blocks = nn.ModuleList([
            Block(2 * channels[i+1], channels[i+1])  # Corrected line
            for i in range(len(channels) - 1)
        ])
        self.channels = channels

    def forward(self, x, encFeatures):
        for i in range(len(self.channels) - 1):
            # Upsampling
            x = self.upconvs[i](x)
            # Cropping
            encFeat = self.crop(encFeatures[i], x)
            # Concatenating encoder and decoder channels (along the
            # 1st dimension)
            x = torch.cat([x, encFeat], dim=1)
            x = self.dec_blocks[i](x)
        return x

    def crop(self, encFeatures, x):
        (_, _, H, W) = x.shape
        encFeatures = CenterCrop([H, W])(encFeatures)
        return encFeatures


class UNet(nn.Module):
    def __init__(self, inChannels=1, outChannels=2):
        super().__init__()
        self.encoder = Encoder()
        self.decoder = Decoder()
        self.finalConv = nn.Conv2d(64, outChannels, kernel_size=1)

    def forward(self, x):
        encFeatures = self.encoder(x)
        x = encFeatures[-1]
        # Reverse the order of the encoder features to match the decoder
        x = self.decoder(x, encFeatures[-2::-1])
        return self.finalConv(x)


# Inspired by:
# https://www.geeksforgeeks.org/converting-an-image-to-a-torch-tensor-in-python/
def resize_image(image, size=(256, 256)):
    transform = transforms.Compose([
        transforms.Resize(size),
        transforms.ToTensor(),
        transforms.Lambda(lambda x: x[:1, :, :] if x.shape[0] > 1 else x)  # Ensure 1 channel
    ])
    return transform(image)


class ImageDataset(Dataset):
    def __init__(self, image_files, mask_folder, transform=None):
        self.image_files = image_files
        self.mask_folder = mask_folder
        self.transform = transform

        # Precompute valid pairs during initialization
        self.valid_pairs = []
        for image_path in self.image_files:
            mask_name = image_path.name.replace(".png", "_mask.png")
            mask_path = self.mask_folder / mask_name
            if image_path.exists() and mask_path.exists():
                self.valid_pairs.append((image_path, mask_path))

    def __len__(self):
        return len(self.valid_pairs)

    def __getitem__(self, idx):
        image_path, mask_path = self.valid_pairs[idx]
        image = Image.open(image_path)
        mask = Image.open(mask_path)

        if self.transform:
            image = self.transform(image)
            mask = self.transform(mask)

        return image, mask


def train_network(net, train_loader, criterion, optimizer):
    for epoch in range(2):  # Use full 32 epochs
        net.train()
        running_loss = 0.0

        for inputs, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}"):
            optimizer.zero_grad()

            labels = labels.squeeze(1).long()  # Remove channel dimension

            # Forward pass
            outputs = net(inputs)

            # Interpolate outputs to match labels size
            if outputs.shape[-2:] != labels.shape[-2:]:
                outputs = F.interpolate(outputs, labels.shape[-2:], mode='bilinear')

            # Calculate loss (outputs: [N, 2, H, W], labels: [N, H, W])
            loss = criterion(outputs, labels)

            # Backward pass
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        print(f"Epoch {epoch+1} Loss: {running_loss/len(train_loader):.4f}")


def test_network(net, test_loader):
    net.eval()
    true_positives, true_negatives, false_positives, false_negatives = 0, 0, 0, 0
    correct, total = 0, 0

    with torch.no_grad():
        for inputs, labels in tqdm(test_loader, desc="Testing"):
            # Forward pass
            print(f"Inputs shape: {inputs.shape}")
            print(f"Labels shape: {labels.shape}")

            labels = labels.squeeze(1).long()  # Remove channel dimension

            outputs = net(inputs)

            # Interpolate outputs to match labels size
            if outputs.shape[-2:] != labels.shape[-2:]:
                outputs = F.interpolate(outputs, labels.shape[-2:], mode='bilinear')

            # __, predicted = torch.max(outputs.data, 1)
            foreground_probs = torch.softmax(outputs, dim=1)[:, 1, :, :]  # shape [N, H, W]
            threshold = 0.55
            predicted = (foreground_probs > threshold).long()

            matches = (predicted == labels)
            correct += matches.sum().item()
            total += labels.numel()

            true_positives += ((predicted == 1) & (labels == 1)).sum().item()
            true_negatives += ((predicted == 0) & (labels == 0)).sum().item()
            false_positives += ((predicted == 1) & (labels == 0)).sum().item()
            false_negatives += ((predicted == 0) & (labels == 1)).sum().item()

        precision = (
            true_positives / (true_positives + false_positives)
            if true_positives + false_positives > 0
            else 0
        )
        sensitivity = (
            true_positives / (true_positives + false_negatives)
            if true_positives + false_negatives > 0
            else 0
        )
        specificity = (
            true_negatives / (true_negatives + false_positives)
            if true_negatives + false_positives > 0
            else 0
        )

        print(f"true_positives: {true_positives}")
        print(f"true_negatives: {true_negatives}")
        print(f"false_positives: {false_positives}")
        print(f"false_negatives: {false_negatives}")
        print(f"Accuracy: {correct / total:.2f}")
        print("Confusion Matrix")
        print(f"Precision: {precision:.4f}")
        print(f"Sensitivity: {sensitivity:.4f}")
        print(f"Specificity: {specificity:.4f}")


if __name__ == "__main__":
    image_folder = Path("chest_xray_dataset/Lung Segmentation/CXR_png")
    mask_folder = Path("chest_xray_dataset/Lung Segmentation/masks")
    image_files = list(image_folder.iterdir())

    train_files, val_files = train_test_split(image_files, test_size=0.3, random_state=42)

    print(f"Train files: {len(train_files)}")
    print(f"Val files: {len(val_files)}")
   
    # Create dataset instance
    train_dataset = ImageDataset(train_files, mask_folder, transform=resize_image)
    val_dataset = ImageDataset(val_files, mask_folder, transform=resize_image)
   
    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    test_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

    # Define the network
    net = UNet()

    # Class weights 
    class_weights = torch.tensor([0.2392, 0.7608])  # [Negative, Positive]
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    
    # Define the loss function and optimizer
    optimizer = optim.Adam(net.parameters(), lr=0.001)  # Use Adam optimizer

    # #Train the network
    train_network(net, train_loader, criterion, optimizer)
    # Test the network
    test_network(net, test_loader)
