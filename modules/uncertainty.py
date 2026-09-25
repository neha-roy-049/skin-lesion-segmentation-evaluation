import torch
import numpy as np

def enable_dropout(model):
    for m in model.modules():
        if isinstance(m, torch.nn.Dropout2d):
            m.train()

def estimate_uncertainty(model, image_tensor, n_passes=15, device='cpu'):
    model.eval()
    enable_dropout(model)

    predictions = []
    with torch.no_grad():
        for _ in range(n_passes):
            logits = model(image_tensor.to(device))
            prob = torch.sigmoid(logits).cpu().numpy()
            predictions.append(prob)

    predictions = np.stack(predictions, axis=0)
    mean_pred = predictions.mean(axis=0)[0, 0]
    uncertainty = predictions.std(axis=0)[0, 0]
    mean_mask = (mean_pred > 0.5).astype(np.uint8)

    confidence = float(1.0 - uncertainty.mean() * 4)
    confidence = max(0.0, min(1.0, confidence))

    return mean_mask, uncertainty, confidence