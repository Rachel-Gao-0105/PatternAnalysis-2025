import random
from torchvision import datasets, transforms
from torch.utils.data import Subset

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

def subset_by_class(dataset, samples_per_class=100, seed=42):
    random.seed(seed)
    indices = []
    for cls_idx in range(len(dataset.classes)):
        cls_indices = [i for i, (_, label) in enumerate(dataset.samples) if label == cls_idx]
        cls_indices = random.sample(cls_indices, min(samples_per_class, len(cls_indices)))
        indices.extend(cls_indices)
    subset = Subset(dataset, indices)
    subset.classes = dataset.classes          
    subset.class_to_idx = dataset.class_to_idx  
    print(f"Extracted {samples_per_class} samples per class, {len(indices)} in total.")
    return subset

def build_datasets(root_dir: str, img_size=224, samples_per_class=None, mean=None, std=None, seed=42):
    if mean is None or std is None:
        mean = [0.485, 0.456, 0.406]
        std  = [0.229, 0.224, 0.225]
    train_tf, test_tf = build_transforms(img_size,mean=mean, std=std)
    train_ds = datasets.ImageFolder(f"{root_dir}/train", transform=train_tf)
    test_ds  = datasets.ImageFolder(f"{root_dir}/test",  transform=test_tf)

    if samples_per_class is not None:
        train_ds = subset_by_class(train_ds, samples_per_class, seed)
        #test_ds  = subset_by_class(test_ds,  samples_per_class, seed)
        print(f"Using subset mode: up to {samples_per_class} samples per class. ")

    return train_ds, test_ds

