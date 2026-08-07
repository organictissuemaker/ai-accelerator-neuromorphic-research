#!/usr/bin/env python3
"""Task 3 - AI-based semiconductor failure classification (transfer learning).

Adapts a pretrained CNN (ResNet18 / MobileNetV2) to the 7 failure classes by
freezing the ImageNet backbone and training only a new classification head on the
small prepared dataset. Reports per-image predicted class, confidence, correct/
incorrect, and overall test accuracy, and saves the trained weights for Task 4.

NOTE ON DATASET SIZE: only ~24 images across 7 classes. This is far too small for
a statistically meaningful accuracy figure; results are a proof-of-concept of the
classification + inference pipeline. The hardware benchmark (Task 4) is the
primary quantitative deliverable.

Repo placement: failure_analysis/fa_classify.py
Run:  python fa_classify.py --data fa_prepared --model resnet18 --epochs 30
Dependencies: torch, torchvision.
"""
import argparse
import csv
import os

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models

MEAN = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 0.225]


def build_model(name, n_classes, device):
    if name == "resnet18":
        m = models.resnet18(weights="DEFAULT")
        for p in m.parameters():
            p.requires_grad = False          # freeze backbone
        m.fc = nn.Linear(m.fc.in_features, n_classes)   # new trainable head
        head = m.fc.parameters()
    elif name == "mobilenet_v2":
        m = models.mobilenet_v2(weights="DEFAULT")
        for p in m.parameters():
            p.requires_grad = False
        m.classifier[1] = nn.Linear(m.classifier[1].in_features, n_classes)
        head = m.classifier[1].parameters()
    else:
        raise ValueError(name)
    return m.to(device), head


def loaders(data_dir):
    train_tf = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15),
        transforms.ColorJitter(0.2, 0.2),
        transforms.ToTensor(),
        transforms.Normalize(MEAN, STD),
    ])
    test_tf = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(MEAN, STD),
    ])
    tr = datasets.ImageFolder(os.path.join(data_dir, "train"), train_tf)
    te = datasets.ImageFolder(os.path.join(data_dir, "test"), test_tf)
    return tr, te


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="fa_prepared")
    ap.add_argument("--model", default="resnet18", choices=["resnet18", "mobilenet_v2"])
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--weights", default="fa_model.pt")
    ap.add_argument("--out", default="fa_predictions.csv")
    args = ap.parse_args()

    torch.manual_seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    tr, te = loaders(args.data)
    classes = tr.classes
    print(f"device={device} | classes={classes}")
    print(f"train={len(tr)} images | test={len(te)} images\n")

    model, head = build_model(args.model, len(classes), device)
    opt = torch.optim.Adam(head, lr=args.lr)

    train_loader = DataLoader(tr, batch_size=8, shuffle=True)
    model.train()
    for ep in range(1, args.epochs + 1):
        run = 0.0
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            opt.zero_grad()
            loss = F.cross_entropy(model(xb), yb)
            loss.backward(); opt.step()
            run += loss.item()
        if ep % 5 == 0 or ep == 1:
            print(f"  epoch {ep:3d}  train_loss={run/len(train_loader):.4f}")

    # ---- evaluate on the held-out test images ----
    model.eval()
    test_loader = DataLoader(te, batch_size=1, shuffle=False)
    rows, correct = [], 0
    print("\nPer-image predictions (test set):")
    print(f"{'file':<34}{'true':<22}{'pred':<22}{'conf':>6}  ok")
    for i, (xb, yb) in enumerate(test_loader):
        with torch.no_grad():
            probs = F.softmax(model(xb.to(device)), dim=1)[0]
        conf, pred = probs.max(0)
        true_c = classes[yb.item()]; pred_c = classes[pred.item()]
        ok = pred.item() == yb.item(); correct += ok
        fname = os.path.basename(te.samples[i][0])
        print(f"{fname:<34}{true_c:<22}{pred_c:<22}{conf.item()*100:5.1f}%  {'Y' if ok else 'N'}")
        rows.append({"file": fname, "true_class": true_c, "pred_class": pred_c,
                     "confidence": round(conf.item(), 4), "correct": int(ok)})

    acc = correct / len(te)
    print(f"\nOverall test accuracy: {acc*100:.1f}%  ({correct}/{len(te)})")
    print("(N=8 test images — illustrative only; see note in header.)")

    with open(args.out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["file", "true_class", "pred_class", "confidence", "correct"])
        w.writeheader(); w.writerows(rows)
    torch.save({"state_dict": model.state_dict(), "classes": classes,
                "model": args.model, "test_acc": acc}, args.weights)
    print(f"saved predictions: {args.out}\nsaved weights: {args.weights}")


if __name__ == "__main__":
    main()
