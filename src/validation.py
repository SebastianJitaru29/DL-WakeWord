from train_utils import load_data, train_fcn, train_pca_ridge
from matplotlib import pyplot as plt
import numpy as np

# Load Data
X, y = load_data()

## Stratified 5-Fold CV for FCN
fcn_history = train_fcn(X, y, n_epochs=2)

# Plot results
plt.figure(figsize=(15, 6))

# F1-score plot
plt.subplot(1, 2, 1)
plt.plot(fcn_history["train_f1"], label="Train F1-score", linestyle="-", color="green")
plt.plot(fcn_history["val_f1"], label="Val F1-score", linestyle="--", color="orange")
plt.xlabel("Epochs")
plt.ylabel("F1-score")
plt.title("FCN: Train vs. Validation F1-score")
plt.legend()

# Loss plot
plt.subplot(1, 2, 2)
plt.plot(fcn_history["train_loss"], label="Train Loss", linestyle="-", color="red")
plt.plot(fcn_history["val_loss"], label="Val Loss", linestyle="--", color="purple")
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.title("FCN: Train vs. Validation Loss")
plt.legend()

# Show plots
plt.tight_layout()
plt.show()

print("\nFinal FCN Results:")
print(f"Mean Accuracy: {fcn_history['val_acc'].mean():.4f}")
print(f"Mean F1-score: {fcn_history['val_f1'].mean():.4f}")



## Stratified 5-Fold CV for PCA + Ridge
mean_acc, std_acc, mean_f1, std_f1 = train_pca_ridge(X, y, num_components=50)

print("\nFinal PCA + Ridge Results:")
print(f"Mean Accuracy: {mean_acc:.4f} - Std Dev: {std_acc:.4f}")
print(f"Mean F1-score: {mean_f1:.4f} - Std Dev: {std_f1:.4f}")
