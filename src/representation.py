import numpy as np
from sklearn.decomposition import PCA
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from src.utils import SEED, set_seed, get_device 
from skimage.feature import hog


def fit_pca(X_train: np.ndarray, seed: int = SEED) -> PCA:
    pca = PCA(svd_solver="full", random_state=seed)
    pca.fit(X_train)                                
    return pca


def pca_project(pca: PCA, X: np.ndarray, n_components: int) -> np.ndarray:
    X_centered = X - pca.mean_                         
    W = pca.components_[:n_components]                  
    return X_centered @ W.T                        


def pca_reconstruct(pca: PCA, Z: np.ndarray) -> np.ndarray:
    k = Z.shape[1]                                     
    W = pca.components_[:k]
    return Z @ W + pca.mean_           



class Autoencoder(nn.Module):

    def __init__(self, latent_dim: int, input_dim: int = 784):
        super().__init__()
        self.latent_dim = latent_dim
        self.input_dim = input_dim                                    # 784 (Fashion-MNIST) ou 2352 (cor)
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 512), nn.ReLU(),
            nn.Linear(512, 256), nn.ReLU(),
            nn.Linear(256, latent_dim),
        )
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 256), nn.ReLU(),
            nn.Linear(256, 512), nn.ReLU(),
            nn.Linear(512, input_dim), nn.Sigmoid(),
        )

    def forward(self, x):
        return self.decoder(self.encoder(x))    


def train_autoencoder(X_train, X_val, latent_dim, epochs=20, batch_size=256, lr=1e-3, seed=SEED):
    set_seed(seed)                                           
    device = get_device()
    model = Autoencoder(latent_dim, input_dim=X_train.shape[1]).to(device)   
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)  
    loss_fn = nn.MSELoss()                                    

    train_loader = DataLoader(
        TensorDataset(torch.from_numpy(X_train)),
        batch_size=batch_size,
        shuffle=True,
        generator=torch.Generator().manual_seed(seed),        
    )
    X_val_t = torch.from_numpy(X_val).to(device)              

    history = {"train_loss": [], "val_loss": []}
    best_val, best_state = float("inf"), None

    for epoch in range(epochs):
        model.train()
        total = 0.0
        for (xb,) in train_loader:
            xb = xb.to(device)
            optimizer.zero_grad()                
            loss = loss_fn(model(xb), xb)         
            loss.backward()                      
            optimizer.step()                     
            total += loss.item() * len(xb)
        train_loss = total / len(X_train)

        # ---- validação (sem atualizar pesos)
        model.eval()
        with torch.no_grad():
            val_loss = loss_fn(model(X_val_t), X_val_t).item()

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)

        if val_loss < best_val:
            best_val = val_loss
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

    model.load_state_dict(best_state)           
    model.eval()
    return model, history


@torch.no_grad()
def ae_encode(model, X, batch_size=4096):
    device = next(model.parameters()).device
    out = []
    for i in range(0, len(X), batch_size):       
        xb = torch.from_numpy(np.asarray(X[i:i + batch_size], dtype=np.float32)).to(device)
        out.append(model.encoder(xb).cpu().numpy())
    return np.concatenate(out)


@torch.no_grad()
def ae_decode(model, Z, batch_size=4096):
    device = next(model.parameters()).device
    out = []
    for i in range(0, len(Z), batch_size):
        zb = torch.from_numpy(np.asarray(Z[i:i + batch_size], dtype=np.float32)).to(device)
        out.append(model.decoder(zb).cpu().numpy())
    return np.concatenate(out)


def save_autoencoder(model, path):
    torch.save(model.state_dict(), path)


def load_autoencoder(latent_dim, path, input_dim=784):
    device = get_device()
    model = Autoencoder(latent_dim, input_dim)
    model.load_state_dict(torch.load(path, map_location=device))
    model.to(device).eval()
    return model





HOG_PARAMS = dict(
    orientations=9,            
    pixels_per_cell=(7, 7),    
    cells_per_block=(2, 2),    
    block_norm="L2-Hys",      
)


def compute_hog(X: np.ndarray) -> np.ndarray:
    return np.array([hog(x.reshape(28, 28), **HOG_PARAMS) for x in X], dtype=np.float32)