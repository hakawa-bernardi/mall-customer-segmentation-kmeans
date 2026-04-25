# =============================================================================
# Segmentação de Clientes com K-Means
# Disciplina: Inteligência Artificial — PUC Goiás
# =============================================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from mpl_toolkits.mplot3d import Axes3D
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

# =============================================================================
# 1. CARREGAMENTO E PRÉ-PROCESSAMENTO DOS DADOS
# =============================================================================

df = pd.read_csv('Mall_Customers.csv')

print("=== Visão geral do dataset ===")
print(df.head())
print(f"\nShape: {df.shape}")
print(f"\nValores nulos:\n{df.isnull().sum()}")

# Codificação da variável categórica Gender (Male=1, Female=0)
df['Gender'] = LabelEncoder().fit_transform(df['Gender'])

# Seleção das features numéricas relevantes
features = df[['Age', 'Annual Income (k$)', 'Spending Score (1-100)']]

# Normalização com StandardScaler
scaler = StandardScaler()
scaled_features = scaler.fit_transform(features)

# =============================================================================
# 2. SELEÇÃO DO NÚMERO DE CLUSTERS
# =============================================================================

wcss = []
silhouette_scores = []
K_range = range(2, 11)

for k in K_range:
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    kmeans.fit(scaled_features)
    wcss.append(kmeans.inertia_)
    silhouette_scores.append(silhouette_score(scaled_features, kmeans.labels_))

# --- Gráfico: Método do Cotovelo ---
plt.figure(figsize=(8, 4))
plt.plot(K_range, wcss, 'bx-', linewidth=2, markersize=8)
plt.xlabel('Número de clusters (k)')
plt.ylabel('WCSS (Inércia)')
plt.title('Método do Cotovelo (Elbow Method)')
plt.xticks(K_range)
plt.grid(True, linestyle='--', alpha=0.5)
plt.tight_layout()
plt.savefig('elbow_method.png', dpi=150)
plt.close()

# --- Gráfico: Silhouette Score ---
plt.figure(figsize=(8, 4))
plt.plot(K_range, silhouette_scores, 'go--', linewidth=2, markersize=8)
plt.xlabel('Número de clusters (k)')
plt.ylabel('Silhouette Score')
plt.title('Silhouette Score por número de clusters')
plt.xticks(K_range)
plt.grid(True, linestyle='--', alpha=0.5)
plt.tight_layout()
plt.savefig('silhouette_scores.png', dpi=150)
plt.close()

# Justificativa: o cotovelo é visível em k=5 e o Silhouette Score
# apresenta valor satisfatório, indicando bom equilíbrio entre
# coesão intra-cluster e separação entre clusters.
k_optimal = 5

# =============================================================================
# 3. MODELAGEM COM K-MEANS
# =============================================================================

kmeans = KMeans(n_clusters=k_optimal, random_state=42, n_init=10)
df['Cluster'] = kmeans.fit_predict(scaled_features)

# Centroides na escala original
centroids_original = scaler.inverse_transform(kmeans.cluster_centers_)
centroids_df = pd.DataFrame(centroids_original, columns=features.columns)
centroids_df.index.name = 'Cluster'

print("\n=== Centroides (escala original) ===")
print(centroids_df.round(2))

# =============================================================================
# 4. VISUALIZAÇÕES
# =============================================================================

sns.set(style='whitegrid')

# --- Gráfico 2D: Renda Anual vs Pontuação de Gasto ---
plt.figure(figsize=(9, 6))
sns.scatterplot(
    data=df,
    x='Annual Income (k$)',
    y='Spending Score (1-100)',
    hue='Cluster',
    palette='Set2',
    s=80,
    edgecolor='white',
    linewidth=0.5
)
plt.scatter(
    centroids_df['Annual Income (k$)'],
    centroids_df['Spending Score (1-100)'],
    c='black', marker='X', s=200, zorder=5, label='Centroides'
)
plt.title('Segmentação de Clientes — K-Means (k=5)', fontsize=13)
plt.xlabel('Renda Anual (k$)')
plt.ylabel('Pontuação de Gasto (1-100)')
plt.legend(title='Cluster', bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout()
plt.savefig('clusters_2d_income_spending.png', dpi=150)
plt.close()

# --- Gráfico 3D: Idade x Renda x Gasto ---
fig = plt.figure(figsize=(11, 8))
ax = fig.add_subplot(111, projection='3d')
colors = plt.cm.Set2(np.linspace(0, 1, k_optimal))

for cluster_id in range(k_optimal):
    mask = df['Cluster'] == cluster_id
    ax.scatter(
        df.loc[mask, 'Age'],
        df.loc[mask, 'Annual Income (k$)'],
        df.loc[mask, 'Spending Score (1-100)'],
        c=[colors[cluster_id]], s=60, label=f'Cluster {cluster_id}'
    )

ax.set_xlabel('Idade')
ax.set_ylabel('Renda (k$)')
ax.set_zlabel('Pontuação de Gasto')
ax.set_title('Clusters em 3D: Renda vs Gasto vs Idade', fontsize=12)
ax.legend(loc='upper left')
plt.tight_layout()
plt.savefig('clusters_3d.png', dpi=150)
plt.close()

# =============================================================================
# 5. ESTATÍSTICAS E INTERPRETAÇÃO POR CLUSTER
# =============================================================================

cluster_stats = df.groupby('Cluster')[['Age', 'Annual Income (k$)', 'Spending Score (1-100)']].mean()
cluster_stats.columns = ['Idade Média', 'Renda Média (k$)', 'Gasto Médio']
cluster_size = df['Cluster'].value_counts().sort_index()

print("\n=== Estatísticas por Cluster ===")
print(cluster_stats.round(1).to_string())

print("\n=== Interpretação dos Clusters ===")
for i, row in cluster_stats.iterrows():
    print(f"\nCluster {i}:")
    print(f"  - Idade média    : {row['Idade Média']:.1f} anos")
    print(f"  - Renda média    : {row['Renda Média (k$)']:.1f} k$")
    print(f"  - Gasto médio    : {row['Gasto Médio']:.1f} / 100")
    print(f"  - Nº de clientes : {cluster_size[i]}")
