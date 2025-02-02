import numpy as np
import torch
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.decomposition import PCA
from sklearn.linear_model import RidgeClassifier
from sklearn.metrics import f1_score, accuracy_score
from sklearn.model_selection import StratifiedKFold, KFold
from fcn_model import FullyConnectedNN

def load_data():
    """Load positive and negative features and return combined dataset."""
    negative_features = np.load("data/negative_features.npy")
    positive_features = np.load("data/positive_features.npy")

    # Combine data and labels
    X = np.vstack((negative_features, positive_features))
    y = np.array([0] * len(negative_features) + [1] * len(positive_features)).astype(np.float32)[..., None]

    return X, y

def train_fcn(X, y, batch_size=512, n_epochs=100, learning_rate=0.001, device=None):
    """Perform 5-fold cross-validation on the Fully Connected Neural Network."""
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    X_tensor, y_tensor = torch.from_numpy(X).float(), torch.from_numpy(y).float()
    kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=10)

    history = {key: np.zeros(n_epochs) for key in ["train_acc", "val_acc", "train_f1", "val_f1", "train_loss", "val_loss"]}

    for fold, (train_idx, test_idx) in enumerate(kf.split(X, y.flatten())):
        print(f"\nFold {fold+1}/5")
        X_train, X_test = X_tensor[train_idx], X_tensor[test_idx]
        y_train, y_test = y_tensor[train_idx], y_tensor[test_idx]

        train_loader = DataLoader(TensorDataset(X_train, y_train), batch_size=batch_size, shuffle=True)
        test_loader = DataLoader(TensorDataset(X_test, y_test), batch_size=batch_size, shuffle=False)

        model = FullyConnectedNN(input_dim=X.shape[1] * X.shape[2]).to(device)
        loss_fn, optimizer = torch.nn.BCELoss(), optim.Adam(model.parameters(), lr=learning_rate)

        for epoch in range(n_epochs):
            model.train()
            train_loss, train_correct, train_total = 0, 0, 0
            all_train_preds, all_train_labels = [], []

            for x_batch, y_batch in train_loader:
                x_batch, y_batch = x_batch.to(device), y_batch.to(device)
                optimizer.zero_grad()
                predictions = model(x_batch)
                loss = loss_fn(predictions, y_batch)
                loss.backward()
                optimizer.step()
                train_loss += loss.item()

                pred_labels = (predictions >= 0.5).float()
                train_correct += (pred_labels == y_batch).sum().item()
                train_total += y_batch.size(0)

                all_train_preds.extend(pred_labels.cpu().numpy())
                all_train_labels.extend(y_batch.cpu().numpy())

            train_loss /= len(train_loader)
            train_acc = train_correct / train_total
            train_f1 = f1_score(all_train_labels, all_train_preds)

            # Validation
            model.eval()
            val_loss, val_correct, val_total = 0, 0, 0
            all_val_preds, all_val_labels = [], []

            with torch.no_grad():
                for x_test, y_test in test_loader:
                    x_test, y_test = x_test.to(device), y_test.to(device)
                    val_preds = model(x_test)
                    val_loss += loss_fn(val_preds, y_test).item()
                    pred_labels = (val_preds >= 0.5).float()

                    val_correct += (pred_labels == y_test).sum().item()
                    val_total += y_test.size(0)
                    all_val_preds.extend(pred_labels.cpu().numpy())
                    all_val_labels.extend(y_test.cpu().numpy())

            val_loss /= len(test_loader)
            val_acc = val_correct / val_total
            val_f1 = f1_score(all_val_labels, all_val_preds)

            history["train_acc"][epoch] += train_acc
            history["val_acc"][epoch] += val_acc
            history["train_f1"][epoch] += train_f1
            history["val_f1"][epoch] += val_f1
            history["train_loss"][epoch] += train_loss
            history["val_loss"][epoch] += val_loss
            

    return {key: val / 5 for key, val in history.items()}

def train_pca_ridge(X, y,  num_components=10):
    """Perform 5-fold cross-validation on PCA + RidgeClassifier."""
    X_flat = X.reshape(X.shape[0], -1)
    n_components = min(X_flat.shape[1], num_components)
    kf = KFold(n_splits=5, shuffle=True, random_state=10)

    fold_results = []
    for fold, (train_idx, test_idx) in enumerate(kf.split(X_flat, y.flatten())):
        print(f"\nFold {fold+1}/5")
        X_train, X_test = X_flat[train_idx], X_flat[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        pca = PCA(n_components=n_components)
        X_train_pca, X_test_pca = pca.fit_transform(X_train), pca.transform(X_test)

        ridge = RidgeClassifier(alpha=1.0)
        ridge.fit(X_train_pca, y_train.ravel())

        y_pred = ridge.predict(X_test_pca)
        accuracy, f1 = accuracy_score(y_test, y_pred), f1_score(y_test, y_pred)

        print(f"Accuracy: {accuracy:.4f} - F1-score: {f1:.4f}")
        fold_results.append((accuracy, f1))

    return np.mean([res[0] for res in fold_results]), np.std([res[0] for res in fold_results]), \
           np.mean([res[1] for res in fold_results]), np.std([res[1] for res in fold_results])
