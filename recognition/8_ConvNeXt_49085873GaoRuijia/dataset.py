import random
from torchvision import datasets, transforms
from torch.utils.data import Subset
from torch.utils.data import DataLoader

# === load the dataset ===========================================================================================================

def build_transforms(img_size=224, mean=None, std=None):
    if mean is None or std is None:
        mean = [0.485, 0.456, 0.406]
        std  = [0.229, 0.224, 0.225]
    '''
    data augmentation for training set

    Resize() -- resize original image
    RandomResizedCrop() -- randomly crop and resize to target size
    RandomHorizontalFlip() -- horizontally flip the image
    ColorJitter() -- adjust brightness and contrast
    ToTensor() -- convert image to a PyTorch tensor and normalize pixel values to [0,1]
    Normalize() -- standardize using ImageNet mean and std for compatibility with pretrained models
    '''
    train_tf = transforms.Compose([
        transforms.Resize(int(img_size*1.15)),
        transforms.RandomResizedCrop(img_size, scale=(0.8, 1.0)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(degrees=10), 
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std),  
    ])
    '''
    data preprocessing for test set

    use fixed cropping only to ensure stable testing results
    CenterCrop() -- center crop to target size
    '''
    val_tf = transforms.Compose([
        transforms.Resize(int(img_size*1.15)),
        transforms.CenterCrop(img_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std),
    ])
    return train_tf, val_tf

def subset_by_class(dataset, samples_per_class=100, seed=42):
    """
    randomly sample a fixed number of images from each class to create a subset dataset
    
    dataset: a torchvision.datasets.ImageFolder object
    samples_per_class: number of samples to extract per class
    seed: random seed for reproducibility
    """
    random.seed(seed)
    indices = []
    for cls_idx in range(len(dataset.classes)):
        cls_indices = [i for i, (_, label) in enumerate(dataset.samples) if label == cls_idx]
        cls_indices = random.sample(cls_indices, min(samples_per_class, len(cls_indices)))
        indices.extend(cls_indices)
    subset = Subset(dataset, indices)
    subset.classes = dataset.classes          
    subset.class_to_idx = dataset.class_to_idx  
    print(f"Extracted {samples_per_class} samples per class, {len(indices)} in total. ")
    return subset

def build_datasets(root_dir: str, img_size=224, samples_per_class=None, mean=None, std=None, seed=42):
    """
    build training and testing datasets, with optional random subset sampling

    root_dir: root directory of dataset
    img_size: target image size
    samples_per_class: if not None, enable per-class sampling
    mean, std: normalization parameters
    seed: random seed for reproducibility
    """
    if mean is None or std is None:
        mean = [0.485, 0.456, 0.406]
        std  = [0.229, 0.224, 0.225]

    train_tf, val_tf = build_transforms(img_size,mean=mean, std=std)
    train_ds = datasets.ImageFolder(f"{root_dir}/train", transform=train_tf)
    val_ds  = datasets.ImageFolder(f"{root_dir}/validation",  transform=val_tf)

    if samples_per_class is not None:
        train_ds = subset_by_class(train_ds, samples_per_class, seed)
        #test_ds  = subset_by_class(test_ds,  samples_per_class, seed)
        print(f"Using subset mode: up to {samples_per_class} samples per class. ")

    return train_ds, val_ds

def data_loader (root_dir: str, img_size=448, samples_per_class=None, mean=None, std=None, seed=42, batch_size=64, num_workers=0):
    """
    build dataloaders for training and validation sets

    root_dir: root directory of dataset
    img_size: target image size
    samples_per_class: if not None, enable per-class sampling
    mean, std: normalization parameters
    seed: random seed for reproducibility
    batch_size: number of samples per batch
    num_workers: number of subprocesses for data loading

    train_loader: dataloader for training set
    val_loader: dataloader for validation set
    """
    train_ds, val_ds = build_datasets(
        root_dir=root_dir,
        img_size=img_size,
        samples_per_class=samples_per_class,
        seed=seed,
        mean=mean, 
        std=std
    )
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                              num_workers=num_workers, pin_memory=True)
    val_loader   = DataLoader(val_ds,  batch_size=batch_size, shuffle=False,
                              num_workers=num_workers, pin_memory=True)
    return train_loader, val_loader