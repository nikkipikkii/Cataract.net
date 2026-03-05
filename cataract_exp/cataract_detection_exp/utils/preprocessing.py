import torch
from torchvision import transforms
from PIL import Image

transform_224 = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485,0.456,0.406],
                         std=[0.229,0.224,0.225])
])

transform_256 = transforms.Compose([
    transforms.Resize((256,256)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485,0.456,0.406],
                         std=[0.229,0.224,0.225])
])

def preprocess(image: Image.Image):
    img_224 = transform_224(image).unsqueeze(0)
    img_256 = transform_256(image).unsqueeze(0)
    return img_224, img_256
