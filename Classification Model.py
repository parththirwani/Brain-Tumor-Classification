#Data Pre-Procsessing

# Define data transformations
transform = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.Resize((64, 64)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# Specify the paths to your custom dataset
train_dataset_path = '/kaggle/input/brain-tumor-mri-dataset/Training/'
test_dataset_path = '/kaggle/input/brain-tumor-mri-dataset/Testing/'

# Load custom dataset
train_dataset = torchvision.datasets.ImageFolder(root=train_dataset_path, transform=transform)
test_dataset = torchvision.datasets.ImageFolder(root=test_dataset_path, transform=transform)



# Define batch size and create data loaders
batch_size = 128
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size)

# Define the VGG16-in-VGG16 model
class InnerVGG16(nn.Module):
    def __init__(self, num_classes):
        super(InnerVGG16, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(512, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        self.avgpool = nn.AdaptiveAvgPool2d((7, 7))
        self.classifier = nn.Sequential(
            nn.Linear(3136, 32),
            nn.ReLU(inplace=True),
            nn.Linear(32, num_classes),
        )

    def forward(self, x):
        #print(x.shape)
        x = self.features(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        #print("flat",x.shape)
        x = self.classifier(x)
        return x

# Define the outer VGG16 model
class OuterVGG16(nn.Module):
    def __init__(self, num_classes):
        super(OuterVGG16, self).__init__()
        self.outer_vgg = models.vgg16(pretrained=True)  # Load a pretrained VGG16
        self.inner_vgg = InnerVGG16(num_classes)  # Create the inner VGG16

    def forward(self, x):
        x = self.outer_vgg.features(x)  # Pass through the outer VGG16 layers
        #print(x.shape)
        x = self.inner_vgg(x)  # Pass through the inner VGG16
        return x

# Initialize the model, loss function, and optimizer
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = OuterVGG16(num_classes=10).to(device)  # Adjust the number of classes as needed
for param in model.outer_vgg.parameters():
    param.requires_grad = False
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)

# Training loop
num_epochs = 100
for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    for inputs, labels in train_loader:
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
    print(f"Epoch {epoch+1}/{num_epochs}, Loss: {running_loss/len(train_loader)}")

# Evaluate the model on the test set

model.eval()
true_labels = []
predicted_labels = []

with torch.no_grad():
    for inputs, labels in test_loader:
        inputs, labels = inputs.to(device), labels.to(device)
        outputs = model(inputs)
        _, predicted = torch.max(outputs.data, 1)
        true_labels.extend(labels.cpu().numpy())
        predicted_labels.extend(predicted.cpu().numpy())

# Calculate accuracy
accuracy = accuracy_score(true_labels, predicted_labels)
print(f"Accuracy: {accuracy * 100:.2f}%")

# Calculate precision, recall, and F1 score
precision = precision_score(true_labels, predicted_labels, average='weighted')
recall = recall_score(true_labels, predicted_labels, average='weighted')
f1 = f1_score(true_labels, predicted_labels, average='weighted')

print(f"Precision: {precision :.2f}%")
print(f"Recall: {recall :.2f}%")
print(f"F1 Score: {f1 :.2f}%")