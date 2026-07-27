import sys
from pathlib import Path

import torch
from PIL import Image
from torchvision import models, transforms

if len(sys.argv) != 2:
    print("Usage: python3 predict_car.py <image_path>")
    raise SystemExit(1)

image_path = Path(sys.argv[1])
model_path = Path("/workspace/models/car_classifier_best.pth")

if not image_path.exists():
    raise FileNotFoundError(f"Image not found: {image_path}")

if not model_path.exists():
    raise FileNotFoundError(f"Trained model not found: {model_path}")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

checkpoint = torch.load(model_path, map_location=device)
classes = checkpoint["classes"]

model = models.mobilenet_v3_small(weights=None)
input_features = model.classifier[3].in_features
model.classifier[3] = torch.nn.Linear(input_features, len(classes))

model.load_state_dict(checkpoint["model_state_dict"])
model = model.to(device)
model.eval()

image_transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225],
    ),
])

image = Image.open(image_path).convert("RGB")
image_tensor = image_transform(image).unsqueeze(0).to(device)

with torch.no_grad():
    outputs = model(image_tensor)
    probabilities = torch.softmax(outputs, dim=1)[0]

top_probabilities, top_indices = torch.topk(
    probabilities,
    min(3, len(classes)),
)

print("\nTop predictions:")

for rank, (probability, index) in enumerate(
    zip(top_probabilities, top_indices),
    start=1,
):
    print(
        f"{rank}. {classes[index.item()]}: "
        f"{probability.item() * 100:.2f}%"
    )
