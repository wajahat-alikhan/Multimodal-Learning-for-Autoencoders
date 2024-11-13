import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from transformers import CLIPProcessor, CLIPModel, VisionEncoderDecoderModel, ViTImageProcessor, AutoTokenizer
from tqdm import tqdm

# CIFAR-10 dataset setup
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

cifar10 = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
dataloader = DataLoader(cifar10, batch_size=32, shuffle=True)

# CIFAR-10 classes and descriptions
cifar10_classes = [
    "airplane", "automobile", "bird", "cat", "deer",
    "dog", "frog", "horse", "ship", "truck"
]

class_descriptions = {
    "airplane": "a sleek flying vehicle with large wings",
    "automobile": "a shiny four-wheeled motor vehicle for transportation",
    "bird": "a vibrant feathered flying animal with a colorful plumage",
    "cat": "a fluffy small domesticated feline pet",
    "deer": "a graceful wild animal with large antlers",
    "dog": "a playful domesticated canine pet with a shiny coat",
    "frog": "a small green amphibious jumping animal with smooth skin",
    "horse": "a majestic large four-legged domesticated animal with a flowing mane",
    "ship": "a large sturdy boat for traveling on water",
    "truck": "a heavy-duty large motor vehicle for transporting goods"
}

# Multi-GPU setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
num_gpus = torch.cuda.device_count()

# Model setup
clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
image_to_text_model = VisionEncoderDecoderModel.from_pretrained("nlpconnect/vit-gpt2-image-captioning")
image_processor = ViTImageProcessor.from_pretrained("nlpconnect/vit-gpt2-image-captioning")
tokenizer = AutoTokenizer.from_pretrained("nlpconnect/vit-gpt2-image-captioning")

# Distribute models across GPUs if available
if num_gpus > 1:
    clip_model = nn.DataParallel(clip_model)
    image_to_text_model = nn.DataParallel(image_to_text_model)

clip_model = clip_model.to(device)
image_to_text_model = image_to_text_model.to(device)

# Freeze CLIP and image-to-text models
for param in clip_model.parameters():
    param.requires_grad = False
for param in image_to_text_model.parameters():
    param.requires_grad = False

class MultimodalAutoEncoder(nn.Module):
    def __init__(self, clip_dim):
        super(MultimodalAutoEncoder, self).__init__()
        self.fusion = nn.Sequential(
            nn.Linear(clip_dim * 2, clip_dim),
            nn.ReLU(),
            nn.Linear(clip_dim, clip_dim)
        )
        self.decoder = nn.Sequential(
            nn.Linear(clip_dim, 512 * 7 * 7),
            nn.ReLU(),
            nn.Unflatten(1, (512, 7, 7)),
            nn.ConvTranspose2d(512, 256, 4, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(256, 128, 4, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(128, 64, 4, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 32, 4, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(32, 3, 4, stride=2, padding=1),
            nn.Tanh()
        )

    def forward(self, clip_image_features, clip_text_features):
        fused_features = self.fusion(torch.cat([clip_image_features, clip_text_features], dim=1))
        generated_image = self.decoder(fused_features)
        return generated_image

def encode_text(text):
    with torch.no_grad():
        inputs = clip_processor(text=text, return_tensors="pt", padding=True, truncation=True).to(device)
        if isinstance(clip_model, nn.DataParallel):
            return clip_model.module.get_text_features(**inputs)
        else:
            return clip_model.get_text_features(**inputs)

def generate_text_from_image(images):
    images = torch.clamp(images * 0.5 + 0.5, 0, 1)
    images = [transforms.ToPILImage()(img.cpu()) for img in images]
    pixel_values = image_processor(images=images, return_tensors="pt").pixel_values.to(device)
    if isinstance(image_to_text_model, nn.DataParallel):
        generated_ids = image_to_text_model.module.generate(pixel_values)
    else:
        generated_ids = image_to_text_model.generate(pixel_values)
    generated_text = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
    return generated_text

# Initialize model
clip_dim = clip_model.module.config.projection_dim if num_gpus > 1 else clip_model.config.projection_dim
model = MultimodalAutoEncoder(clip_dim).to(device)
if num_gpus > 1:
    model = nn.DataParallel(model)

# Loss functions and optimizer
criterion_mse = nn.MSELoss()
criterion_cosine = nn.CosineEmbeddingLoss()
optimizer = optim.Adam(model.parameters(), lr=1e-4)

# Training loop
num_epochs = 300
for epoch in range(num_epochs):
    total_loss = 0
    progress_bar = tqdm(dataloader, desc=f'Epoch {epoch+1}/{num_epochs}')
    
    for batch_idx, (data, targets) in enumerate(progress_bar):
        data = data.to(device)
        
        # Get text descriptions for the batch
        text_inputs = [class_descriptions[cifar10_classes[t.item()]] for t in targets]
        
        # Get CLIP features
        with torch.no_grad():
            clip_text_features = encode_text(text_inputs)
            if isinstance(clip_model, nn.DataParallel):
                clip_image_features = clip_model.module.get_image_features(data)
            else:
                clip_image_features = clip_model.get_image_features(data)
        
        # Forward pass
        optimizer.zero_grad()
        generated_image = model(clip_image_features, clip_text_features)
        
        # Calculate losses
        reconstruction_loss = criterion_mse(generated_image, data)
        image_text_loss = criterion_cosine(clip_image_features, clip_text_features, torch.ones(data.shape[0]).to(device))
        
        generated_text = generate_text_from_image(generated_image)
        generated_text_features = encode_text(generated_text)
        text_similarity_loss = criterion_cosine(generated_text_features, clip_text_features, torch.ones(data.shape[0]).to(device))
        
        # Combined loss
        loss = reconstruction_loss + 0.2 * image_text_loss + 0.3 * text_similarity_loss
        
        # Backward pass and optimization
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        progress_bar.set_postfix({'loss': loss.item()})
    
    avg_loss = total_loss / len(dataloader)
    print(f'Epoch [{epoch+1}/{num_epochs}], Average Loss: {avg_loss:.4f}')

    # Save checkpoint every 50 epochs
    if (epoch + 1) % 50 == 0:
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': avg_loss,
        }
        torch.save(checkpoint, f'checkpoint_epoch_{epoch+1}.pth')

 #Save final model
torch.save(model.state_dict(), 'final_model.pth')