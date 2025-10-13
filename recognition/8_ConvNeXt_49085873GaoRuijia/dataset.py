from torchvision import transforms

# === load the dataset ===========================================================

def build_transforms(img_size=224, mean=None, std=None):
    if mean is None or std is None:
        mean = [0.485, 0.456, 0.406]
        std  = [0.229, 0.224, 0.225]

    train_tf = transforms.Compose([
        transforms.Resize(int(img_size*1.15)),
        transforms.RandomResizedCrop(img_size, scale=(0.8, 1.0)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(degrees=10), 
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std),  
    ])

    test_tf = transforms.Compose([
        transforms.Resize(int(img_size*1.15)),
        transforms.CenterCrop(img_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std),
    ])
    return train_tf, test_tf



