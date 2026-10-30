# Projeto Prático 2 – Redução de Dimensionalidade (Fashion-MNIST)

**Machine Learning – Universidade da Beira Interior, 2026/27**
André Fazendeiro (n.º 48191)

Representações compactas de imagens de roupa (PCA, autoencoder, HOG e histogramas de cor) usadas em duas tarefas: **classificação** e **pesquisa de peças parecidas** (*look-alike retrieval*). O objetivo é mostrar que a melhor representação para classificar não é necessariamente a melhor para pesquisar.

---

## Estrutura do repositório

```
├── src/
│   ├── representation.py   # PCA (projeção/reconstrução), autoencoder (encode/decode), HOG
│   ├── retrieval.py        # pesquisa top-k em vários espaços + precision@k
│   ├── color_data.py       # extensão a cores: subconjunto, histograma de cor, cinzento
│   └── utils.py            # seeds, carregar dados, split, classificador, painéis
├── notebooks/              # experiências, por ordem (00 a 16)
├── models/                 # modelos já treinados (autoencoders .pt, classificador em píxeis)
├── results/
│   ├── figures/            # todas as figuras do relatório (+ retrieval_panels/)
│   └── tables/             # todas as tabelas (.csv)
├── report/                 # relatório (LaTeX/PDF)
├── data/                   # datasets (NÃO estão no repositório, são descarregados automaticamente)
└── requirements.txt
```

## Como correr

### Google Colab
```python
!git clone https://github.com/<UTILIZADOR>/<REPOSITORIO>.git
%cd <REPOSITORIO>
!pip install -q umap-learn scikit-image datasets scikit-learn==1.9.1
```
Depois, cada notebook pode ser executado assim (exemplo):
```python
!jupyter nbconvert --to notebook --execute --inplace \
  --ExecutePreprocessor.kernel_name=python3 --ExecutePreprocessor.timeout=-1 \
  notebooks/09_classification.ipynb
```
A versão do scikit-learn é fixada porque o classificador guardado em `models/clf_raw.joblib` foi treinado com a 1.9.1. A GPU é opcional: só acelera o treino dos autoencoders.

### Localmente
```bash
python -m venv .venv
.venv\Scripts\activate            # Windows (Linux/Mac: source .venv/bin/activate)
python -m pip install -r requirements.txt
```
Os notebooks podem ser abertos a partir da pasta `notebooks/`. A primeira célula de cada um muda automaticamente para a pasta principal do projeto.

## Ordem dos notebooks

| Notebook | O que faz | Tempo aprox. |
|---|---|---|
| `00_setup_check` | Verifica bibliotecas e GPU | segundos |
| `01_data_exploration` | Carrega o Fashion-MNIST, contagens e exemplos | < 1 min |
| `02_splits` | Normalização e split 50k/10k/10k reprodutível | segundos |
| `03_baseline_raw_pixels` | Regressão logística nos 784 píxeis (**opcional**: o modelo já está em `models/`) | ~14 min (CPU) |
| `04_pca` | Variância explicada e erro de reconstrução | < 1 min |
| `05_visualization_2d` | PCA vs t-SNE vs UMAP em 2D | ~2 min |
| `06_autoencoder` | Treina 6 autoencoders, d = 2…128 (**opcional**: já estão em `models/`) | ~6 min (GPU) |
| `07_representation_check` | Testa o `representation.py` e compara os espaços latentes 2D | < 1 min |
| `08_reconstruction_panels` | Painéis de reconstrução e MSE no teste | < 1 min |
| `09_classification` | Tabela de classificação: píxeis vs PCA vs AE | ~2 min |
| `10_texture_hog` | Descritor HOG e classificação com HOG | ~3 min |
| `11_retrieval_module` | Demonstração do `retrieval.py` | < 1 min |
| `12_retrieval_evaluation` | precision@5/10 em todos os espaços e dimensões | ~1 min |
| `13_retrieval_panels` | Painéis de retrieval e precision por classe | ~1 min |
| `14a_color_dataset` | Cria o subconjunto a cores | ~1 min (+ download) |
| `14b_color_retrieval` | Cor vs textura vs embedding no dataset a cores | ~1 min |
| `15_failure_cases` | Casos de falha (sistemáticos) | < 1 min |
| `16_main_conclusion` | Classificação vs retrieval: figura-resumo, saturação, Spearman | segundos |

## Datasets

### Fashion-MNIST (obrigatório)
Descarregado automaticamente pelo `torchvision` a partir da fonte oficial (<https://github.com/zalandoresearch/fashion-mnist>) para `data/`.
- Píxeis divididos por 255 (valores em [0, 1]) e imagens achatadas para vetores de 784.
- Split: 50 000 treino / 10 000 validação (estratificado, tirado dos 60 000 de treino) / 10 000 teste (oficial).
- A validação foi usada para decisões (épocas do autoencoder). O teste só foi usado para os resultados finais.

### Extensão a cores – **foi usada**
- **Fonte:** *Fashion Product Images (Small)*, através da cópia no Hugging Face `ashraq/fashion-product-images-small` (original no Kaggle: `paramaggarwal/fashion-product-images-small`). É descarregada automaticamente pela biblioteca `datasets` no notebook `14a`.
- **Subconjunto:** 5 tipos de peça (Tshirts, Shirts, Tops, Casual Shoes, Handbags) × 6 cores (Black, White, Blue, Red, Green, Yellow), com no máximo 100 imagens por combinação (amostragem com seed 42) e combinações com menos de 10 imagens ignoradas. Ficaram **2704 imagens**.
- **Preparação:** cada foto é colocada num quadrado branco (para não ser deformada) e reduzida para **28×28 RGB**, com valores em [0, 1]. O resultado é guardado em `data/color_subset.npz`.
- **Split:** 80% galeria (2163) / 20% queries (541), estratificado por tipo + cor.
- **Labels** (tipo e cor) usadas só para avaliar, nunca no ranking.
- **Limitações:** a roupa de tronco é fotografada vestida por modelos (aparecem cara, pele e calças), as peças brancas confundem-se com o fundo branco, o amarelo é raro (19 a 67 imagens em 4 dos 5 tipos) e há algum ruído nas labels de cor.

## Reprodutibilidade
- Seed fixa (`SEED = 42`) para `random`, `numpy`, `torch` e todos os `random_state` do scikit-learn.
- O split é gerado por uma função determinística (`get_splits`), por isso é sempre igual.
- Os autoencoders foram treinados na GPU. Ao treinar de novo podem surgir diferenças mínimas (4.ª–5.ª casa decimal), que não alteram as conclusões.

## Principais resultados (conjunto de teste)

| Espaço | Dimensão | Accuracy | precision@10 |
|---|---|---|---|
| Píxeis brutos | 784 | 0,833 | 0,800 |
| HOG | 324 | 0,844 | 0,782 |
| PCA | 32 / 128 | 0,815 / 0,841 | 0,797 / 0,806 |
| Autoencoder | 32 / 128 | 0,823 / **0,858** | 0,818 / **0,824** |

- O autoencoder é a melhor representação em quase tudo. Com 8 dimensões já pesquisa tão bem como os 784 píxeis.
- Classificar ≠ pesquisar: o HOG é o 2.º melhor a classificar e o 10.º a pesquisar, e a pesquisa satura com 32 dimensões, enquanto a classificação precisa de 64–128.
- Cores: o histograma é o melhor para "mesma cor" (0,68), o HOG para "mesmo tipo" (0,75), e o autoencoder é o melhor compromisso (0,51 / 0,75).

## Requisitos do enunciado → onde estão

| Requisito | Onde |
|---|---|
| `representation.py` com PCA e autoencoder | `src/representation.py` (testado em `07`) |
| `retrieval.py` com top-k em ≥ 3 espaços | `src/retrieval.py` (forma/PCA, textura/HOG, embedding/AE; + cor em `14b`) |
| Fashion-MNIST + subconjunto a cores | `01`, `02`, `14a` |
| Dimensões 2, 8, 16, 32, 64, 128 | `04`, `06`, `09`, `12` |
| Labels escondidas no ranking | `retrieval.py` não recebe labels; só `precision_at_k` as usa |
| Painéis de reconstrução | `08` |
| Tabela de classificação | `09` (+ HOG em `10`) |
| Painéis de retrieval | `13`, `14b` → `results/figures/retrieval_panels/` |
| Análise de dimensionalidade | `09`, `12`, `16` |
| ≥ 5 casos de falha | `15` (9 casos) |
