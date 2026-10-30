import random
import numpy as np
import torch
from torchvision import datasets
from sklearn.model_selection import train_test_split 
import time 
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt

SEED = 42 


def set_seed(seed: int = SEED) -> None:
    random.seed(seed)                  
    np.random.seed(seed)               
    torch.manual_seed(seed)            
    torch.cuda.manual_seed_all(seed)   


def get_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")





CLASS_NAMES = ["T-shirt/top", "Trouser", "Pullover", "Dress", "Coat",
               "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot"]


def load_fashion_mnist(data_dir: str = "data"):
    train = datasets.FashionMNIST(root=data_dir, train=True, download=True)   
    test = datasets.FashionMNIST(root=data_dir, train=False, download=True)   

    X_train = train.data.numpy()     
    y_train = train.targets.numpy()   
    X_test = test.data.numpy()
    y_test = test.targets.numpy()
    return X_train, y_train, X_test, y_test






def get_splits(val_size: int = 10_000, seed: int = SEED, data_dir: str = "data"):
    X_full, y_full, X_test, y_test = load_fashion_mnist(data_dir)

    
    X_full = X_full.reshape(len(X_full), -1).astype(np.float32) / 255.0
    X_test = X_test.reshape(len(X_test), -1).astype(np.float32) / 255.0

    
    X_train, X_val, y_train, y_val = train_test_split(
        X_full, y_full,
        test_size=val_size,     
        stratify=y_full,       
        random_state=seed,     
    )
    return X_train, y_train, X_val, y_val, X_test, y_test







def train_and_evaluate_classifier(Z_train, y_train, Z_eval, y_eval, seed: int = SEED):
    clf = make_pipeline(
        StandardScaler(),                                        
        LogisticRegression(max_iter=1000, random_state=seed),     
    )
    start = time.perf_counter()
    clf.fit(Z_train, y_train)                                  
    train_time = time.perf_counter() - start

    y_pred = clf.predict(Z_eval)                                
    acc = accuracy_score(y_eval, y_pred)                       
    return clf, acc, train_time



DIMS = [2, 8, 16, 32, 64, 128]




def plot_retrieval_panel(query_img, query_label, results, X_gallery, y_gallery,
                         k=5, title=None, save_path=None):
    n_spaces = len(results)
    fig, axes = plt.subplots(n_spaces, k + 1, figsize=((k + 1) * 1.5, n_spaces * 1.9))
    axes = np.atleast_2d(axes)                                         

    for r, (name, (idx, _)) in enumerate(results.items()):
        top = idx[0, :k]
        hits = int(np.sum(y_gallery[top] == query_label))              
        axes[r, 0].imshow(query_img.reshape(28, 28), cmap="gray")
        axes[r, 0].set_title(f"QUERY\n{CLASS_NAMES[query_label]}", fontsize=8, fontweight="bold")
        axes[r, 0].set_ylabel(f"{name}\n({hits}/{k} same class)", fontsize=8,
                              rotation=0, ha="right", va="center")
        for j, i in enumerate(top):
            same = y_gallery[i] == query_label
            axes[r, j + 1].imshow(X_gallery[i].reshape(28, 28), cmap="gray")
            axes[r, j + 1].set_title(CLASS_NAMES[y_gallery[i]], fontsize=8,
                                     color="green" if same else "red")

    for ax in axes.ravel():
        ax.set_xticks([]); ax.set_yticks([])
    if title:
        fig.suptitle(title, fontsize=10)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.show()