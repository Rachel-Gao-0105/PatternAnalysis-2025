import os, argparse, torch, torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, roc_auc_score
import matplotlib.pyplot as plt
from modules import ConvNeXt
from dataset import build_datasets, data_loader
from torch.optim.lr_scheduler import LinearLR, CosineAnnealingLR, SequentialLR
import numpy as np
from sklearn.metrics import roc_curve

# === MixUp ===========================================================================================================

def mixup_data(x, y, alpha=0.2):
    """
    Perform the MixUp data augmentation technique.

    x (Tensor): Input batch, typically shaped [batch_size, C, H, W].
    y (Tensor): Corresponding labels, shaped [batch_size] or [batch_size, num_classes].
    alpha: controls the strength of interpolation.
        If alpha > 0: lambda is sampled from Beta(alpha, alpha).
        If alpha <= 0: MixUp is disabled (lambda = 1).

    mixed_x (Tensor): Mixed input batch.
    y_a (Tensor): Original labels.
    y_b (Tensor): Labels of shuffled samples.
    lam: Mixing ratio (lambda) representing the weight of the original sample.
    """
    
    if alpha <= 0: #MixUp is disabled
        lam = 1.0
        return x, y, y, lam   
    
    lam = np.random.beta(alpha, alpha)  #randomly sample lam from Beta distribution
    bs = x.size(0)  #batch size
    index = torch.randperm(bs, device=x.device)
    mixed_x = lam * x + (1 - lam) * x[index]
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam


def mixup_criterion(criterion, pred, y_a, y_b, lam):
    """
    Compute the MixUp loss.

    criterion: Base loss function.
    pred (Tensor): Model predictions.
    y_a (Tensor): Labels of the first (original) samples.
    y_b (Tensor): Labels of the second (shuffled) samples.
    lam: Mixing ratio (lambda) representing contribution of each label.

    loss (Tensor): Weighted MixUp loss.
    """
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)  #final loss

# === train_one_epoch ===========================================================================================================

def train_one_epoch(model, loader, criterion, optimizer, device, scaler, epoch=None, total_epochs=None):
    """
    Train the model for one epoch with optional MixUp augmentation and automatic mixed precision (AMP).

    model: The model to train.
    loader: PyTorch DataLoader that provides batches of training data
    criterion: Loss function
    optimizer: Optimizer used to update model parameters
    device: 'cuda' or 'cpu'
    scaler: Scaler for mixed precision training to prevent gradient underflow.
    epoch: Current training epoch number (used for dynamic MixUp adjustment).
    total_epochs: Total number of epochs (used for scheduling MixUp alpha).

    avg_loss: Average loss across all batches.
    acc: Accuracy 
    auc: Area Under the ROC Curve 
    """
    model.train()
    losses, preds, gts = [], [], [] #batch losses, predictions, ground truths

    for x, y in loader:
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        x = x.to(memory_format=torch.channels_last)

        optimizer.zero_grad(set_to_none=True)

        #dynamically adjust MixUp strength
        if epoch is not None and total_epochs is not None:
            current_alpha = 0.2 * max(0.0, 1.0 - epoch / (0.7 * total_epochs))
        else:
            current_alpha = 0.2  #fallback
        
        #apply MixUp augmentation
        x, y_a, y_b, lam = mixup_data(x, y, alpha=current_alpha)

        #AMP 
        with torch.amp.autocast("cuda", enabled=(device == "cuda")):
            #x, y_a, y_b, lam = mixup_data(x, y, alpha=0.2)
            out = model(x)
            loss = mixup_criterion(criterion, out, y_a, y_b, lam)

        #backward pass with gradient scaling to avoid underflow
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        losses.append(loss.item())

        #convert logits to probabilities for class 1
        p = out.softmax(1)[:, 1].detach().cpu().tolist()
        preds.extend(p)
        gts.extend(y.cpu().tolist())

    #accuracy with threshold 0.5
    acc = accuracy_score(gts, [p >= 0.5 for p in preds])

    try:
        auc = roc_auc_score(gts, preds)
    except Exception:
        auc = float("nan")

    return sum(losses)/len(losses), acc, auc

# === evaluate ===========================================================================================================

@torch.no_grad()
def evaluate(model, loader, criterion, device):
    """
    Evaluate the model on a validation or test dataset.

    model: trained model to be evaluated
    loader: PyTorch DataLoader providing evaluation data batches
    criterion: loss function
    device: 'cuda' or 'cpu'

    avg_loss: average loss across all evaluation batches
    acc: accuracy at a fixed threshold (0.5)
    auc: Area Under the ROC Curve
    acc_tuned: accuracy computed at the optimal threshold derived from ROC.
    best_thr: best threshold value maximizing
    """
    model.eval()
    losses, preds, gts = [], [], [] #batch losses, predictions, ground truth labels
    for x, y in loader:
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        x = x.to(memory_format=torch.channels_last)

        with torch.amp.autocast("cuda", enabled=(device == "cuda")):
            out = model(x)
            loss = criterion(out, y)

        losses.append(loss.item())
        preds.extend(out.softmax(1)[:, 1].cpu().tolist())
        gts.extend(y.cpu().tolist())

    #compute accuracy using a fixed decision threshold of 0.5
    acc = accuracy_score(gts, [p >= 0.5 for p in preds]) 
    
    #compute AUC, best threshold, accuracy at that threshold 
    try:
        #the area under ROC curve
        auc = roc_auc_score(gts, preds) 
        #the ROC curve (false positive rate, true positive rate, thresholds)
        fpr, tpr, thr = roc_curve(gts, preds) 
        j = tpr - fpr
        best_idx = int(np.argmax(j))
        best_thr = float(thr[best_idx])
        #accuracy at the optimal threshold
        acc_tuned = accuracy_score(gts, (np.array(preds) >= best_thr).astype(int)) 
    except Exception:
        auc = float("nan")
        best_thr = 0.5
        acc_tuned = float("nan")

    return sum(losses)/len(losses), acc, auc, acc_tuned, best_thr

# === Parse Arguments ===========================================================================================================

def parse_args():
    """
    Parse command-line arguments for model training configuration.
    
    Example:
        python train.py --data_root "D:/Datasets/ADNI/AD_NC" --epochs 50 --lr 1e-4
    """
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_root", type=str, required=True, help="root directory of the dataset, e.g. D:/.../ADNI/AD_NC") 
    #ap.add_argument("--model_name", type=str, default="convnext_tiny")
    ap.add_argument("--img_size", type=int, default=448, help="image resolution (width and height)")                                          
    ap.add_argument("--num_classes", type=int, default=2, help="number of output classes")
    ap.add_argument("--drop_path_rate", type=float, default=0.1, help="stochastic depth (DropPath) rate")
    ap.add_argument("--dropout_rate", type=float, default=0.2, help="dropout rate to prevent overfitting")
    ap.add_argument("--weight_decay", type=float, default=0.07, help="L2 regularization factor for optimizer to control model complexity")
    ap.add_argument("--batch_size", type=int, default=64, help="number of samples per batch")
    ap.add_argument("--epochs", type=int, default=50, help="total number of training epochs")
    ap.add_argument("--lr", type=float, default=1.5e-4, help="initial learning rate for optimizer")
    ap.add_argument("--out_dir", type=str, default="runs", help="directory to save outputs")
    ap.add_argument("--num_workers", type=int, default=0, help="number of subprocesses for data loading")  
    ap.add_argument("--samples_per_class_train", type=int, default=None, help="e.g. 100 to quick pilot, None to use full set")
    ap.add_argument("--seed", type=int, default=42)
    #early stopping, None to disable min_delta = 1e-3
    ap.add_argument("--patience", type=int, default=10, help="epochs to wait without val_acc improvement, 0 to disable")
    ap.add_argument("--target_acc", type=float, default=0.8, help="early stop if val_acc >= target_acc, 0 to disable")
    ap.add_argument("--min_delta", type=float, default=1e-3, help="minimum improvement to reset stagnation counter")
    return ap.parse_args()

def main():
    '''
    Main entry point for training and evaluating.

    1.  Parsing configuration arguments
    2.  Dataset construction
    3.  Model, optimizer, and scheduler setup
    4.  Training and evaluation loops
    5.  Model checkpoint saving and early stopping
    6.  Visualization of performance metrics
    '''
    # Parse arguments and environment setup
    args = parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    # Pre-computed ImageNet normalization values
    mean, std = None, None
    mean = [0.485, 0.456, 0.406]
    std  = [0.229, 0.224, 0.225]

    # Build training and validation datasets
    train_loader, val_loader = data_loader (root_dir=args.data_root, 
                                                img_size=args.img_size, 
                                                samples_per_class=args.samples_per_class_train, 
                                                mean=mean, 
                                                std=std, 
                                                seed=args.seed, 
                                                batch_size=args.batch_size, 
                                                num_workers=args.num_workers)

    # Build model
    model = ConvNeXt(num_classes=args.num_classes, drop_path_rate=args.drop_path_rate, dropout_rate = args.dropout_rate) 
    model = model.to(device)
    model = model.to(memory_format=torch.channels_last)

    print(">> Training")

    # Loss function and optimizer
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    criterion = criterion.to(device)

    # AdamW optimizer: adaptive learning + weight decay
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    
    # Learning rate scheduling
    warmup_steps = max(1, int(0.05 * args.epochs))       
    t_max = max(1, args.epochs - warmup_steps)           
    
    warmup = LinearLR(optimizer, start_factor=0.1, total_iters=warmup_steps)
    cosine = CosineAnnealingLR(optimizer, T_max= t_max)
    scheduler = SequentialLR(optimizer, schedulers=[warmup, cosine], milestones=[warmup_steps])

    # Mixed Precision Training (AMP)
    scaler = torch.amp.GradScaler("cuda", enabled=(device == "cuda"))

    '''
    Metric tracking

    tr_loss        Average training loss over all batches in the current epoch
    tr_acc         Training accuracy at a fixed threshold of 0.5
    tr_auc         Training AUC (Area Under the ROC Curve)
    val_loss       Average validation loss over all batches
    val_acc        Validation accuracy at threshold 0.5
    val_auc        Validation AUC
    val_acc_tuned  Validation accuracy computed at the optimal threshold (maximizing TPR - FPR)
    val_thr        Best decision threshold found from the ROC curve
    '''
    history = {"tr_loss":[], "tr_acc":[], "tr_auc":[], "val_loss":[], "val_acc":[], "val_auc":[], "val_acc_tuned":[], "val_thr":[]}
    best_acc = 0.0
    best_auc = 0.0
    best_acc_tuned = 0.0

    '''
    Paths for saving best models

    best_acc.pth         Model achieving the highest validation accuracy at threshold = 0.5
    best_auc.pth         Model achieving the highest validation AUC
    best_acc_tuned.pth   Model achieving the highest validation accuracy at the optimal threshold
    '''
    best_acc_path = os.path.join(args.out_dir, "best_acc.pth")
    best_auc_path = os.path.join(args.out_dir, "best_auc.pth")
    best_acc_tuned_path = os.path.join(args.out_dir, "best_acc_tuned.pth")

    patience = args.patience  # early-stopping patience (tolerated stagnation epochs)
    no_improve_epochs = 0 # counter for stagnation detection
    min_delta = args.min_delta  # minimum improvement to reset stagnation counter

    # Training & Evaluation Loop
    for epoch in range(1, args.epochs + 1):
        tr_loss, tr_acc, tr_auc = train_one_epoch(
                model, train_loader, criterion, optimizer, device, scaler,
                epoch=epoch, total_epochs=args.epochs
            )
        val_loss, val_acc, val_auc, val_acc_tuned, best_thr = evaluate(model, val_loader, criterion, device) #修改

        history["tr_loss"].append(tr_loss)
        history["tr_acc"].append(tr_acc)
        history["tr_auc"].append(tr_auc)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)
        history["val_auc"].append(val_auc)
        history["val_acc_tuned"].append(val_acc_tuned)
        history["val_thr"].append(best_thr) 

        print(f"Epoch {epoch:03d} | "
          f"train loss {tr_loss:.4f} acc {tr_acc:.3f} auc {tr_auc:.3f} | "
          f"val loss {val_loss:.4f} acc@0.5 {val_acc:.3f} auc {val_auc:.3f} | "
          f"val acc@bestThr {val_acc_tuned:.3f} thr={best_thr:.3f}")

        improved_acc = (val_acc_tuned - best_acc_tuned) > min_delta
        if improved_acc:
            no_improve_epochs = 0
        else:
            no_improve_epochs += 1

        # Save model with the best tuned accuracy (based on ROC-optimal threshold)
        if val_acc_tuned > best_acc_tuned:
            best_acc_tuned = val_acc_tuned
            torch.save({"model": model.state_dict(), "args": vars(args), "mean": mean, "std": std, "best_thr": best_thr}, best_acc_tuned_path)

        # Save model with the highest accuracy at threshold 0.5
        if val_acc > best_acc:
            best_acc = val_acc
            torch.save({"model": model.state_dict(), "args": vars(args), "mean": mean, "std": std, "best_thr": best_thr}, best_acc_path)

        # Save model with the highest AUC
        if val_auc > best_auc:
            best_auc = val_auc
            torch.save({"model": model.state_dict(), "args": vars(args), "mean": mean, "std": std, "best_thr": best_thr}, best_auc_path)
                
        scheduler.step()

        # Early Stopping
        if args.target_acc > 0 and val_acc >= args.target_acc:
            print(f"\nEarly Stopping Triggered: Validation accuracy reached {val_acc:.3f} ≥ {args.target_acc:.3f}.")
            break

        if patience > 0 and no_improve_epochs >= patience:
            print(f"\nEarly stopping due to no improvement for {patience} epochs.")
            break

    # Visualization
    plt.figure(); 
    plt.plot(history["tr_loss"]) 
    plt.plot(history["val_loss"]) 
    plt.legend(["train_loss","val_loss"]) 
    plt.title("Loss") 
    plt.savefig(os.path.join(args.out_dir,"loss.png"))

    plt.figure()
    plt.plot(history["tr_acc"])
    plt.plot(history["val_acc"])
    plt.plot(history["val_acc_tuned"])
    plt.legend(["train_acc","val_acc@0.5","val_acc@bestThr"])
    plt.title("Accuracy")
    plt.savefig(os.path.join(args.out_dir,"acc.png"))

    plt.figure()
    plt.plot(history["tr_auc"])
    plt.plot(history["val_auc"])
    plt.legend(["train_auc","val_auc"])
    plt.title("AUC")
    plt.savefig(os.path.join(args.out_dir,"auc.png"))

    print(f"Best val acc@0.5: {best_acc:.3f} | best_acc_ckpt: {best_acc_path}")
    print(f"Best val AUC: {best_auc:.3f} | best_auc_ckpt: {best_auc_path}")
    print(f"Best val acc@bestThr: {best_acc_tuned:.3f}  | best_acc_tuned_ckpt: {best_acc_tuned_path}")

if __name__ == "__main__":
    main()

