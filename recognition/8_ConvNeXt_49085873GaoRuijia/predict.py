import argparse
import os
import torch
import torch.nn.functional as F
import numpy as np
from torchvision import transforms, datasets
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, roc_auc_score, confusion_matrix, classification_report
from PIL import Image
import matplotlib.pyplot as plt
from modules import ConvNeXt  

def parse_args():
    ap = argparse.ArgumentParser(description="Single-image inference & folder evaluation")
    ap.add_argument("--ckpt", type=str, required=True, help="checkpoint path, e.g. runs/best_acc_tuned.pth")
    ap.add_argument("--data_root", type=str, default=None, help="folder root that contains a split folder (e.g. validation)")
    ap.add_argument("--split", type=str, default="validation", help="subfolder name under data_root for evaluation")
    ap.add_argument("--image", type=str, default=None, help="single image path to predict (skip folder eval if provided)")
    ap.add_argument("--img_size", type=int, default=448, help="inference resolution")
    ap.add_argument("--batch_size", type=int, default=64)
    ap.add_argument("--num_workers", type=int, default=0)
    ap.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    return ap.parse_args()

def build_transform(img_size, mean, std):
    return transforms.Compose([
        transforms.Resize(int(img_size * 1.15)),
        transforms.CenterCrop(img_size),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])

def load_model_from_ckpt(ckpt_path, device):
    ckpt = torch.load(ckpt_path, map_location=device)

    ckpt_args = ckpt.get("args", {}) or {}
    num_classes = ckpt_args.get("num_classes", 2)
    drop_path_rate = ckpt_args.get("drop_path_rate", 0.0)
    dropout_rate = ckpt_args.get("dropout_rate", 0.0)

    mean      = ckpt.get("mean", [0.485, 0.456, 0.406])
    std       = ckpt.get("std",  [0.229, 0.224, 0.225])
    img_size  = ckpt.get("img_size", 448)
    classes   = ckpt.get("classes", ['AD','NC'])
    class_to_idx = ckpt.get("class_to_idx", {c:i for i,c in enumerate(classes)})
    best_thr  = ckpt.get("best_thr", 0.5)

    model = ConvNeXt(num_classes=num_classes, drop_path_rate=drop_path_rate, dropout_rate=dropout_rate)
    model.load_state_dict(ckpt["model"])
    model = model.to(device).eval()

    return model, mean, std, best_thr, ckpt_args, classes

@torch.no_grad()
def predict_single_image(model, img_path, device, tf, class_names=None, thr=0.5, show=False):
    """
    Predict a single image with a trained model.

    model: PyTorch model in eval mode.
    img_path: Path to the input image.
    device: 'cuda' or 'cpu'.
    tf: Preprocessing transform that maps PIL.Image -> Tensor.
    class_names: Optional list of class names aligned with class indices.
    thr: Decision threshold for binary classification (applies to class-1 probability).
    show: If True, display the image with the prediction title.

    pred_idx: Predicted class index.
    prob1: Confidence of the predicted class (for binary: prob of the chosen class).
    probs: Softmax probabilities over classes.
    """
    img = Image.open(img_path).convert("RGB")
    x = tf(img).unsqueeze(0).to(device)

    # forward pass with AMP
    with torch.amp.autocast("cuda", enabled=(device == "cuda")):
        logits = model(x)
        probs = F.softmax(logits, dim=1)[0].cpu().numpy()

    # apply threshold on probability of class 1
    pred_idx = int(probs[1] >= thr) if probs.shape[0] == 2 else int(np.argmax(probs))
    prob1 = float(probs[pred_idx])

    if class_names is not None and 0 <= pred_idx < len(class_names):
        print(f"[IMAGE] {img_path}")
        print(f"Pred index: {pred_idx} | Pred class: {class_names[pred_idx]} | Confidence: {prob1:.4f}")
    else:
        print(f"[IMAGE] {img_path}")
        print(f"Pred index: {pred_idx} | Confidence: {prob1:.4f}")

    # visualize
    if show:
        if class_names is not None:
            title = f"Pred: class {class_names[pred_idx]}, index {pred_idx} ({prob1*100:.1f}%)"
        else:
            title = f"Pred: index {pred_idx} ({prob1*100:.1f}%)"
        plt.imshow(img)
        plt.title(title)
        plt.axis("off")
        plt.show()

    return pred_idx, prob1, probs


@torch.no_grad()
def evaluate_folder(model, root_dir, split, device, tf, batch_size=64, num_workers=0, thr=0.5):
    """
    Evaluate a folder-structured dataset (ImageFolder style) with a trained model.

    model: torch.nn.Module already moved to `device` and set to eval() by caller.
    root_dir: Root directory containing subfolders: train/validation/test
    split: Which subfolder under root_dir to evaluate
    device: 'cuda' or 'cpu'.
    tf: torchvision transform applied to each image
    batch_size: DataLoader batch size.
    num_workers: DataLoader workers.
    thr: Decision threshold for binary classification on the probability of class index 1.
    """
    folder = os.path.join(root_dir, split)
    if not os.path.isdir(folder):
        raise FileNotFoundError(f"Folder not found: {folder}")

    # create dataset & loader
    ds = datasets.ImageFolder(folder, transform=tf) #ImageFolder infers labels from subfolder names
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False,
                        num_workers=num_workers, pin_memory=True)
    class_names = ds.classes

    preds, gts = [], [] #preds hold probabilities for positive class; gts holds integer labels
    for x, y in loader:
        x = x.to(device, non_blocking=True).to(memory_format=torch.channels_last)
        y = y.to(device, non_blocking=True)

        # AMP forward pass
        with torch.amp.autocast("cuda", enabled=(device == "cuda")):
            out = model(x)
        p = out.softmax(1)[:, 1].detach().cpu().numpy() if out.shape[1] == 2 else out.softmax(1).detach().cpu().numpy().max(axis=1)

        # take probability of class index 1 as positive
        preds.extend(p)  # probabilities for positive class
        gts.extend(y.cpu().numpy())

    preds = np.array(preds)
    gts = np.array(gts, dtype=int)

    # AUC & ACC
    try:
        auc = roc_auc_score(gts, preds)
    except Exception:
        auc = float("nan")
    acc_thr = accuracy_score(gts, (preds >= thr).astype(int))
    acc05   = accuracy_score(gts, (preds >= 0.5).astype(int))
    y_hat = (preds >= thr).astype(int)

    cm = confusion_matrix(gts, y_hat, labels=list(range(len(class_names))))
    report = classification_report(gts, y_hat, target_names=class_names, digits=3)

    print("\nResults")
    print(f"AUC: {auc:.3f}")
    print(f"ACC@thr({thr:.3f}): {acc_thr:.3f}")
    print(f"ACC@0.5: {acc05:.3f}")
    print("Confusion Matrix (rows=true, cols=pred):")
    print(cm)
    print("\nClassification Report:")
    print(report)


def main():
    args = parse_args()
    device = args.device
    print(f"Device: {device}")

    # load ckpt & model
    model, mean, std, best_thr, ckpt_args, class_names = load_model_from_ckpt(args.ckpt, device)

    # transform
    img_size = args.img_size
    tf = build_transform(img_size, mean, std)

    # single image inference
    if args.image is not None:
        predict_single_image(model, args.image, device, tf, class_names=class_names, thr=best_thr, show=True)
        return

    # dataset evaluation
    if args.data_root is None:
        print("Please provide --data_root for folder evaluation, or use --image for single-image inference.")
    else:
        evaluate_folder(model, args.data_root, args.split, device, tf,
                    batch_size=args.batch_size, num_workers=args.num_workers, thr=best_thr)

if __name__ == "__main__":
    main()

