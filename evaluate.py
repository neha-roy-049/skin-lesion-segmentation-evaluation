import torch
import numpy as np
import cv2
import os
from modules.model import SkinLesionSegNet
from modules.preprocess import SkinLesionDataset, val_transform
from torch.utils.data import DataLoader

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

def compute_metrics(pred_mask, true_mask):
    TP = ((pred_mask == 1) & (true_mask == 1)).sum()
    TN = ((pred_mask == 0) & (true_mask == 0)).sum()
    FP = ((pred_mask == 1) & (true_mask == 0)).sum()
    FN = ((pred_mask == 0) & (true_mask == 1)).sum()
    return TP, TN, FP, FN

def get_tta_probs(model, images):
    """Run TTA and return averaged probability map"""
    probs_orig   = torch.sigmoid(model(images))
    probs_hflip  = torch.flip(torch.sigmoid(model(torch.flip(images, [3]))), [3])
    probs_vflip  = torch.flip(torch.sigmoid(model(torch.flip(images, [2]))), [2])
    probs_hvflip = torch.flip(torch.sigmoid(model(torch.flip(images, [2,3]))), [2,3])
    return (probs_orig + probs_hflip + probs_vflip + probs_hvflip) / 4.0

def apply_morphology(preds, device, kernel_size=3):
    """Clean up noisy predictions using morphological operations"""
    preds_np = preds.cpu().numpy()
    kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE, (kernel_size, kernel_size)
    )
    for i in range(preds_np.shape[0]):
        mask = preds_np[i, 0].astype(np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        preds_np[i, 0] = mask
    return torch.from_numpy(preds_np).to(device)

def find_best_threshold(model, val_loader, device):
    """Find optimal threshold using TTA predictions"""
    print("\nFinding optimal threshold using TTA...")
    thresholds = [0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7]
    best_thresh = 0.5
    best_iou = 0

    for thresh in thresholds:
        all_iou = []
        with torch.no_grad():
            for images, masks in val_loader:
                images = images.to(device)
                masks  = masks.to(device)

                # Use TTA here too — consistent with main evaluation
                probs = get_tta_probs(model, images)
                preds = (probs > thresh).float()
                preds = apply_morphology(preds, device)

                for i in range(images.size(0)):
                    pred_flat = preds[i].cpu().numpy().flatten()
                    mask_flat = masks[i].cpu().numpy().flatten()
                    TP = ((pred_flat == 1) & (mask_flat == 1)).sum()
                    FP = ((pred_flat == 1) & (mask_flat == 0)).sum()
                    FN = ((pred_flat == 0) & (mask_flat == 1)).sum()
                    iou = TP / (TP + FP + FN + 1e-6)
                    all_iou.append(iou)

        mean_iou = np.mean(all_iou)
        print(f"  Threshold {thresh:.2f} → IoU: {mean_iou*100:.2f}%")
        if mean_iou > best_iou:
            best_iou = mean_iou
            best_thresh = thresh

    print(f"\n  Best threshold: {best_thresh} → IoU: {best_iou*100:.2f}%")
    return best_thresh

def main():
    print("=" * 55)
    print("   SKIN LESION SEGMENTATION — MODEL EVALUATION")
    print("=" * 55)

    # ── Load model ───────────────────────────────────────────
    print("\nLoading model...")
    model = SkinLesionSegNet().to(DEVICE)
    model.load_state_dict(torch.load('models/best_model.pth',
                                      map_location=DEVICE))
    model.eval()
    print("Model loaded successfully.")

    # ── Load validation dataset ──────────────────────────────
    print("Loading validation dataset...")
    dataset = SkinLesionDataset(
        image_dir='data/images',
        mask_dir='data/masks',
        transform=val_transform
    )
    val_size    = int(len(dataset) * 0.2)
    val_indices = list(range(len(dataset) - val_size, len(dataset)))
    val_dataset = torch.utils.data.Subset(dataset, val_indices)
    val_loader  = DataLoader(val_dataset, batch_size=4,
                             shuffle=False, num_workers=0)
    print(f"Validation samples: {len(val_dataset)}")

    # ── Find best threshold ──────────────────────────────────
    best_threshold = find_best_threshold(model, val_loader, DEVICE)

    # ── Run final evaluation ─────────────────────────────────
    total_TP = total_TN = total_FP = total_FN = 0
    all_iou  = []
    all_dice = []

    print("\nRunning final evaluation...")
    print("-" * 55)

    with torch.no_grad():
        for batch_idx, (images, masks) in enumerate(val_loader):
            images = images.to(DEVICE)
            masks  = masks.to(DEVICE)

            # TTA predictions
            probs = get_tta_probs(model, images)
            preds = (probs > best_threshold).float()

            # Morphological cleanup
            preds = apply_morphology(preds, DEVICE)

            # Compute metrics per image
            for i in range(images.size(0)):
                pred_flat = preds[i].cpu().numpy().flatten()
                mask_flat = masks[i].cpu().numpy().flatten()

                TP, TN, FP, FN = compute_metrics(
                    pred_flat.astype(int),
                    mask_flat.astype(int)
                )
                total_TP += TP
                total_TN += TN
                total_FP += FP
                total_FN += FN

                intersection = TP
                union        = TP + FP + FN
                iou  = intersection / (union + 1e-6)
                dice = (2 * TP) / (2 * TP + FP + FN + 1e-6)
                all_iou.append(iou)
                all_dice.append(dice)

            if (batch_idx + 1) % 10 == 0:
                print(f"  Processed {(batch_idx+1)*4} / "
                      f"{len(val_dataset)} samples...")

    # ── Compute final metrics ────────────────────────────────
    TP = total_TP
    TN = total_TN
    FP = total_FP
    FN = total_FN

    accuracy    = (TP + TN) / (TP + TN + FP + FN + 1e-6)
    precision   = TP / (TP + FP + 1e-6)
    recall      = TP / (TP + FN + 1e-6)
    f1          = 2 * precision * recall / (precision + recall + 1e-6)
    specificity = TN / (TN + FP + 1e-6)
    mean_iou    = float(np.mean(all_iou))
    mean_dice   = float(np.mean(all_dice))

    # ── Print results ────────────────────────────────────────
    print("\n" + "=" * 55)
    print("   EVALUATION RESULTS")
    print("=" * 55)
    print(f"  Best Threshold  : {best_threshold}")
    print(f"  Accuracy        : {accuracy    * 100:.2f}%")
    print(f"  Precision       : {precision   * 100:.2f}%")
    print(f"  Recall          : {recall      * 100:.2f}%")
    print(f"  F1 Score        : {f1          * 100:.2f}%")
    print(f"  Specificity     : {specificity * 100:.2f}%")
    print(f"  Mean IoU        : {mean_iou    * 100:.2f}%")
    print(f"  Mean Dice Score : {mean_dice   * 100:.2f}%")
    print("=" * 55)
    print("\n  Confusion Matrix (pixel-level totals):")
    print(f"  TP = {TP:,}   FP = {FP:,}")
    print(f"  FN = {FN:,}   TN = {TN:,}")
    print("=" * 55)

    # ── Save results ─────────────────────────────────────────
    os.makedirs('models', exist_ok=True)
    with open('models/eval_metrics.txt', 'w') as f:
        f.write("SKIN LESION SEGMENTATION — EVALUATION METRICS\n")
        f.write("=" * 50 + "\n")
        f.write(f"Best Threshold  : {best_threshold}\n")
        f.write(f"Accuracy        : {accuracy    * 100:.2f}%\n")
        f.write(f"Precision       : {precision   * 100:.2f}%\n")
        f.write(f"Recall          : {recall      * 100:.2f}%\n")
        f.write(f"F1 Score        : {f1          * 100:.2f}%\n")
        f.write(f"Specificity     : {specificity * 100:.2f}%\n")
        f.write(f"Mean IoU        : {mean_iou    * 100:.2f}%\n")
        f.write(f"Mean Dice Score : {mean_dice   * 100:.2f}%\n")
        f.write("=" * 50 + "\n")
        f.write(f"TP={TP}  FP={FP}  FN={FN}  TN={TN}\n")

    print("\n  Results saved to: models/eval_metrics.txt")
    print("=" * 55)

if __name__ == '__main__':
    main()