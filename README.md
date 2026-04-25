# 🛍️ Segmentação de Clientes com K-Means

Trabalho acadêmico da disciplina de **Inteligência Artificial** (CMP1110) — PUC Goiás, 2025/1.

Implementação do algoritmo **K-Means** para agrupar clientes de um shopping com base em características como renda anual, pontuação de gastos e idade.

---

## 📁 Estrutura do Projeto

```
kmeans-mall-customers/
├── Mall_Customers.csv       # Dataset de segmentação de clientes
├── kmeans.py                # Script principal com toda a análise
├── requirements.txt         # Dependências do projeto
└── README.md
```

---

## 📊 Dataset

**Mall Customer Segmentation Data** — 200 clientes com as seguintes variáveis:

| Variável | Descrição |
|---|---|
| `CustomerID` | Identificador único |
| `Gender` | Gênero (Male/Female) |
| `Age` | Idade |
| `Annual Income (k$)` | Renda anual em milhares de dólares |
| `Spending Score (1-100)` | Pontuação de gastos atribuída pelo shopping |

---

## ⚙️ Etapas da Análise

### 1. Pré-processamento
- Codificação da variável categórica `Gender` (Male=1, Female=0)
- Seleção das features numéricas: `Age`, `Annual Income`, `Spending Score`
- Normalização com `StandardScaler`

### 2. Seleção do número de clusters
- **Método do Cotovelo (Elbow Method)** para identificar o K ideal
- **Silhouette Score** calculado para K = 2 a K = 10
- K ótimo escolhido: **5**

### 3. Modelagem
- Treinamento do K-Means com K=5
- Adição dos rótulos de cluster ao DataFrame
- Extração e análise dos centroides

### 4. Visualizações geradas
- Gráfico do Método do Cotovelo → `elbow_method.png`
- Silhouette Score por K → `silhouette_scores.png`
- Clusters 2D (Renda vs Gasto) → `clusters_2d_income_spending.png`
- Clusters 3D (Idade × Renda × Gasto) → `clusters_3d.png`

### 5. Interpretação dos Clusters

| Cluster | Perfil |
|---|---|
| 0 | Renda alta, gasto alto — clientes alvo premium |
| 1 | Renda baixa, gasto baixo — clientes conservadores |
| 2 | Renda alta, gasto baixo — clientes cautelosos |
| 3 | Renda baixa, gasto alto — clientes impulsivos |
| 4 | Renda média, gasto médio — clientes padrão |

---

## 🚀 Como executar

### Pré-requisitos
```bash
pip install -r requirements.txt
```

### Execução
```bash
python kmeans.py
```

Os gráficos serão salvos como arquivos `.png` no diretório atual.

---

## 🛠️ Tecnologias

- Python 3.x
- pandas
- numpy
- scikit-learn
- matplotlib
- seaborn

---

## 👤 Informações

- **Disciplina:** Inteligência Artificial — CMP1110
- **Instituição:** PUC Goiás — Campus Escola Politécnica
- **Professor(a):** Clarimar J. Coelho
- **Semestre:** 2025/1
