"""
TEST DE VALIDATION ISOLE — CNN2D sur CIFAR-10, entrainement LOCAL classique.
============================================================================
Objectif : verifier que l'architecture du modele (jobs/progressive/models_pytorch.py)
et les donnees (get_cifar10) sont capables d'apprendre correctement, INDEPENDAMMENT
du systeme distribue (pas de serveur, pas de volontaire, pas de compression de
gradients, pas d'asynchronie/staleness).

Si ce script atteint une bonne precision (70%+) en quelques dizaines d'epoques
REELLES (passage complet sur les 50 000 images a chaque epoque), cela prouve que
le modele et les donnees sont sains, et que l'ecart de performance observe dans
le systeme distribue vient bien des contraintes du systeme (compression,
asynchronie, couverture partielle par "epoque") -- pas d'un bug ou d'un mauvais
choix d'architecture.

Usage : python sanity_check_cnn2d_local.py
"""
import time
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from jobs.progressive.models_pytorch import CNN2D
from jobs.progressive.data_providers import get_cifar10

N_EPOCHS = 30          # epoques REELLES (passage complet sur les 50000 images a chaque fois)
BATCH_SIZE = 128
LR = 1e-3

def evaluate(model, loader, device):
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            pred = model(x).argmax(dim=1)
            correct += (pred == y).sum().item()
            total += y.size(0)
    return correct / total

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device : {device}")

    print("Chargement de CIFAR-10 (avec augmentation train)...")
    train_set = get_cifar10(train=True)
    test_set = get_cifar10(train=False)
    train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
    test_loader = DataLoader(test_set, batch_size=256, shuffle=False, num_workers=0)
    print(f"Train: {len(train_set)} images | Test: {len(test_set)} images")

    model = CNN2D(n_classes=10).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Modele CNN2D : {n_params:,} parametres")

    optimizer = optim.Adam(model.parameters(), lr=LR)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=N_EPOCHS)
    criterion = nn.CrossEntropyLoss()

    print(f"\nEntrainement LOCAL classique : {N_EPOCHS} epoques REELLES "
          f"(chacune = passage complet sur les {len(train_set)} images)\n")

    for epoch in range(1, N_EPOCHS + 1):
        model.train()
        t0 = time.time()
        running_loss = 0.0
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            out = model(x)
            loss = criterion(out, y)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * x.size(0)
        scheduler.step()

        train_loss = running_loss / len(train_set)
        test_acc = evaluate(model, test_loader, device)
        dt = time.time() - t0
        print(f"Epoque {epoch:3d}/{N_EPOCHS} | loss={train_loss:.4f} | "
              f"precision_test={test_acc*100:5.2f}% | {dt:.1f}s")

    print("\nTermine. Compare cette precision finale a celle obtenue par le "
          "systeme distribue (meme modele, memes donnees) pour isoler l'effet "
          "reel de la distribution/compression/asynchronie.")

if __name__ == "__main__":
    main()
