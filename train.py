import torch
import torch.nn as nn
from modules.model import SkinLesionSegNet
from modules.preprocess import get_dataloaders
import os

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
EPOCHS = 30
LR = 1e-4
BATCH_SIZE = 4

print(f"Using device: {DEVICE}")

def dice_loss(pred, target, smooth=1.0):
    pred = torch.sigmoid(pred)
    intersection = (pred * target).sum(dim=(2, 3))
    dice = (2 * intersection + smooth) / (
        pred.sum(dim=(2, 3)) + target.sum(dim=(2, 3)) + smooth
    )
    return 1 - dice.mean()

def combined_loss(pred, target):
    bce = nn.BCEWithLogitsLoss()(pred, target)
    dl = dice_loss(pred, target)
    return 0.5 * bce + 0.5 * dl

def compute_iou(pred, target, threshold=0.5):
    pred = (torch.sigmoid(pred) > threshold).float()
    intersection = (pred * target).sum()
    union = pred.sum() + target.sum() - intersection
    return (intersection / (union + 1e-6)).item()

print("Loading dataset...")
train_loader, val_loader = get_dataloaders(
    'data/images', 'data/masks', BATCH_SIZE, subset=500
)
print(f"Train batches: {len(train_loader)} | Val batches: {len(val_loader)}")

print("Loading model...")
model = SkinLesionSegNet().to(DEVICE)
optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, EPOCHS)

best_iou = 0
print("Starting training...")
for epoch in range(EPOCHS):
    model.train()
    total_loss = 0

    for batch_idx, (images, masks) in enumerate(train_loader):
        images, masks = images.to(DEVICE), masks.to(DEVICE)
        optimizer.zero_grad()
        preds = model(images)
        loss = combined_loss(preds, masks)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

        if batch_idx % 50 == 0:
            print(f"  Epoch {epoch+1} | Batch {batch_idx}/{len(train_loader)} | Loss: {loss.item():.4f}")

    scheduler.step()

    model.eval()
    val_iou = 0
    with torch.no_grad():
        for images, masks in val_loader:
            images, masks = images.to(DEVICE), masks.to(DEVICE)
            preds = model(images)
            val_iou += compute_iou(preds, masks)
    val_iou /= len(val_loader)

    print(f"Epoch {epoch+1}/{EPOCHS} | Loss: {total_loss/len(train_loader):.4f} | Val IoU: {val_iou:.4f}")

    if val_iou > best_iou:
        best_iou = val_iou
        os.makedirs('models', exist_ok=True)
        torch.save(model.state_dict(), 'models/best_model.pth')
        # Save IoU score to a text file
        with open('models/best_iou.txt', 'w') as f:
            f.write(str(round(best_iou, 4)))
        print(f"  ✓ Best model saved! IoU: {best_iou:.4f}")

print("Training complete!")