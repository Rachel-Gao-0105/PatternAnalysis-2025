import os
import torch
import torch.nn.functional as F
import numpy as np
from torchvision import transforms, datasets
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, roc_auc_score, confusion_matrix, classification_report
from PIL import Image
import matplotlib.pyplot as plt

# load model definition
ckpt_path = "best_acc_tuned_copy.pth"   
device = "cuda" if torch.cuda.is_available() else "cpu"

ckpt = torch.load(ckpt_path, map_location=device)
mean = ckpt.get("mean", [0.485, 0.456, 0.406])
std  = ckpt.get("std",  [0.229, 0.224, 0.225])
best_thr = ckpt.get("best_thr", 0.5)
ckpt_args = ckpt.get("args", {})  

from modules import convnext_tiny
model = convnext_tiny(weights=None, num_classes=2, stochastic_depth_prob=0.1)
model.load_state_dict(ckpt["model"])
model = model.to(device).eval()

# prepare test dataset & loader
data_root = ckpt_args.get("data_root", "/root/autodl-tmp/ADNI/AD_NC")
img_size = ckpt_args.get("img_size", 384)

test_tf = transforms.Compose([
    transforms.Resize(int(img_size * 1.15)),
    transforms.CenterCrop(img_size),
    transforms.ToTensor(),
    transforms.Normalize(mean, std),
])

test_dir = os.path.join(data_root, "test")
test_ds = datasets.ImageFolder(test_dir, transform=test_tf)
test_loader = DataLoader(test_ds, batch_size=64, shuffle=False, num_workers=0, pin_memory=True)
classes = test_ds.classes                     
class_to_idx = test_ds.class_to_idx           # {'AD':0, 'NC':1}

# single image prediction demo
def predict_image(img_path, gt_from_path=True):
    """返回：probs(2,), pred_class(0/1), img(PIL), (可选)是否预测正确"""
    img = Image.open(img_path).convert("RGB")
    x = test_tf(img).unsqueeze(0).to(device)

    with torch.no_grad():
        logits = model(x)
        probs = F.softmax(logits, dim=1)[0].cpu().numpy()

    pred_class = int(probs[1] >= best_thr)

    correct = None
    if gt_from_path:
        parts = os.path.normpath(img_path).split(os.sep)
        try:
            idx = parts.index("test")
            gt_name = parts[idx+1]  # 'AD' / 'NC'
            if gt_name in class_to_idx:
                gt = class_to_idx[gt_name]
                correct = (pred_class == gt)
        except ValueError:
            pass

    return probs, pred_class, img, correct

def show_prediction(img_path):
    probs, pred_class, img, correct = predict_image(img_path, gt_from_path=True)
    conf = probs[pred_class]
    title = f"Pred: {classes[pred_class]} ({conf*100:.1f}%)"
    if correct is not None:
        title += f" | {'✔️ Correct' if correct else '❌ Wrong'}"
    plt.imshow(img)
    plt.title(title)
    plt.axis("off")
    plt.show()

demo_images = [
    "/root/autodl-tmp/ADNI/AD_NC/test/AD/388206_79.jpeg",
    "/root/autodl-tmp/ADNI/AD_NC/test/AD/1439611_106.jpeg",
]
for p in demo_images:
    if os.path.exists(p):
        print(f"\n[DEMO] {p}")
        show_prediction(p)
    else:
        print(f"[WARN] Not found: {p}")

# evaluate on the entire test set
@torch.no_grad()
def test_on_loader(model, loader, device, thr=0.5):
    model.eval()
    preds, gts = [], []
    for x, y in loader:
        x = x.to(device, non_blocking=True).to(memory_format=torch.channels_last)
        y = y.to(device, non_blocking=True)
        with torch.amp.autocast("cuda", enabled=(device == "cuda")):
            out = model(x)
        p = out.softmax(1)[:, 1].detach().cpu().numpy()
        preds.extend(p)
        gts.extend(y.cpu().numpy())

    preds = np.array(preds)
    gts = np.array(gts, dtype=int)

    try:
        auc = roc_auc_score(gts, preds)
    except Exception:
        auc = float("nan")

    acc_thr = accuracy_score(gts, (preds >= thr).astype(int))
    acc05   = accuracy_score(gts, (preds >= 0.5).astype(int))

    y_hat = (preds >= thr).astype(int)
    cm = confusion_matrix(gts, y_hat, labels=[0, 1])
    report = classification_report(gts, y_hat, target_names=classes, digits=3)

    return acc_thr, acc05, auc, cm, report

print("\n[TEST] Evaluating on test set...")
acc_thr, acc05, auc, cm, report = test_on_loader(model, test_loader, device, thr=best_thr)
print(f"[TEST] AUC = {auc:.3f}")
print(f"[TEST] ACC@thr({best_thr:.3f}) = {acc_thr:.3f}")
print(f"[TEST] ACC@0.5 = {acc05:.3f}")
print("[TEST] Confusion Matrix (rows=true [AD,NC], cols=pred [AD,NC]):")
print(cm)
print("\n[TEST] Classification Report (thr):")
print(report)
