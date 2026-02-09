# %% [markdown]
# # Segmentacao de Clientes - Analise Completa
# ## UCI Online Retail Dataset - RFM + Clustering
#
# Pipeline end-to-end seguindo o CRISP-DM:
# 1. Definicao do Problema
# 2. Carregamento e EDA
# 3. Preparacao dos Dados e Feature Engineering (RFM)
# 4. Modelagem (KMeans, Hierarchical, DBSCAN)
# 5. Otimizacao de Hiperparametros
# 6. Avaliacao Final
# 7. Interpretacao (SHAP)
# 8. Salvamento e Relatorio

# %%
# ============================================================
# SETUP E CONFIGURACAO
# ============================================================
import random, os, sys
import numpy as np
import pandas as pd
import yaml
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import warnings
import logging
warnings.filterwarnings('ignore')

# Configurar logging uma vez para todo o projeto
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

def set_seed(seed=42):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    return seed

SEED = set_seed()

# Configuracao de plots
sns.set_style('whitegrid')
sns.set_palette('Set2')
plt.rcParams['figure.dpi'] = 150
plt.rcParams['savefig.dpi'] = 150
plt.rcParams['figure.figsize'] = [10, 6]

# Caminhos do projeto
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, PROJECT_ROOT)

DATA_PATH = os.path.join(PROJECT_ROOT, 'data', 'raw', 'online_retail_raw.csv')
FIGURES_PATH = os.path.join(PROJECT_ROOT, 'outputs', 'figures')
MODELS_PATH = os.path.join(PROJECT_ROOT, 'outputs', 'models')
REPORTS_PATH = os.path.join(PROJECT_ROOT, 'outputs', 'reports')
PROCESSED_PATH = os.path.join(PROJECT_ROOT, 'data', 'processed')
FEATURES_PATH = os.path.join(PROJECT_ROOT, 'data', 'features')

for p in [FIGURES_PATH, MODELS_PATH, REPORTS_PATH, PROCESSED_PATH, FEATURES_PATH]:
    os.makedirs(p, exist_ok=True)

# Imports dos modulos src/
from src.data.data_loader import load_from_csv
from src.data.data_cleaner import clean_data
from src.features.rfm_calculator import calculate_rfm_features

print(f"Projeto raiz: {PROJECT_ROOT}")
print(f"Arquivo de dados: {DATA_PATH}")
print(f"Dados existem: {os.path.exists(DATA_PATH)}")

# Carregar config.yaml
CONFIG_PATH = os.path.join(PROJECT_ROOT, 'config', 'config.yaml')
with open(CONFIG_PATH, 'r', encoding='utf-8') as f:
    yaml_config = yaml.safe_load(f)

# %%
# ============================================================
# FASE 1: DEFINICAO DO PROBLEMA
# ============================================================
config = {
    "problem_type": "clustering",
    "target_col": None,
    "success_metric": "silhouette_score",
    "threshold": yaml_config['evaluation'].get('min_silhouette_score', 0.4),
    "seed": SEED,
    "business_goals": [
        "Identificar clientes de alto valor para retencao",
        "Detectar clientes em risco de churn",
        "Encontrar oportunidades de crescimento em novos clientes"
    ]
}

print("=" * 60)
print("FASE 1: DEFINICAO DO PROBLEMA")
print("=" * 60)
print(f"Tipo: {config['problem_type']}")
print(f"Metrica principal: {config['success_metric']}")
print(f"Limiar de sucesso: {config['threshold']}")
print(f"Seed: {config['seed']}")
print("\nObjetivos de negocio:")
for goal in config['business_goals']:
    print(f"  - {goal}")

# %%
# ============================================================
# FASE 2: CARREGAMENTO DOS DADOS E EDA
# ============================================================
print("\n" + "=" * 60)
print("FASE 2: CARREGAMENTO DOS DADOS E EDA")
print("=" * 60)

df = load_from_csv(DATA_PATH)
print(f"\n1. SHAPE: {df.shape}")
print(f"   Linhas: {df.shape[0]:,}")
print(f"   Colunas: {df.shape[1]}")

# %%
print(f"\n2. TIPOS DE DADOS:")
print(df.dtypes)

# %%
print(f"\n3. VALORES AUSENTES:")
missing = df.isnull().sum()
missing_pct = (missing / len(df) * 100).round(2)
missing_df = pd.DataFrame({'Contagem': missing, 'Percentual (%)': missing_pct})
missing_df = missing_df[missing_df['Contagem'] > 0].sort_values('Contagem', ascending=False)
print(missing_df)

# %%
print(f"\n4. DUPLICATAS: {df.duplicated().sum():,} ({df.duplicated().mean()*100:.2f}%)")

# %%
print(f"\n5. ESTATISTICAS DESCRITIVAS (Numericas):")
print(df.describe())
print(f"\n   ESTATISTICAS DESCRITIVAS (Categoricas):")
print(df.describe(include='object'))

# %%
print(f"\n6. PRIMEIRAS LINHAS:")
print(df.head(10))

# %%
print(f"\n7. INFORMACOES ADICIONAIS:")
print(f"   Clientes unicos: {df['CustomerID'].nunique():,} (com NaN ignorado)")
print(f"   Invoices unicos: {df['InvoiceNo'].nunique():,}")
print(f"   Produtos unicos: {df['StockCode'].nunique():,}")
print(f"   Paises: {df['Country'].nunique()}")

df['InvoiceDate'] = pd.to_datetime(df['InvoiceDate'])
print(f"   Periodo: {df['InvoiceDate'].min()} a {df['InvoiceDate'].max()}")

# %%
# ============================================================
# EDA - VISUALIZACOES
# ============================================================
print("\n--- Gerando visualizacoes EDA ---")

# Plot 1: Valores ausentes
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
missing_data = df.isnull().sum()
missing_data = missing_data[missing_data > 0].sort_values(ascending=True)
if len(missing_data) > 0:
    missing_data.plot(kind='barh', ax=ax1, color='coral')
    ax1.set_title('Valores Ausentes (Contagem)', fontweight='bold')
    ax1.set_xlabel('Contagem')

    missing_pct_plot = (missing_data / len(df) * 100)
    missing_pct_plot.plot(kind='barh', ax=ax2, color='steelblue')
    ax2.set_title('Valores Ausentes (%)', fontweight='bold')
    ax2.set_xlabel('Percentual (%)')
plt.tight_layout()
plt.savefig(os.path.join(FIGURES_PATH, 'eda_missing_values.png'), dpi=150, bbox_inches='tight')
plt.close()
print("  Salvo: eda_missing_values.png")

# %%
# Plot 2: Distribuicao de Quantity e UnitPrice
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

df['Quantity'].hist(bins=100, ax=axes[0, 0], edgecolor='black', alpha=0.7)
axes[0, 0].set_title('Distribuicao de Quantity (Bruto)', fontweight='bold')
axes[0, 0].set_xlabel('Quantity')
axes[0, 0].set_ylabel('Frequencia')
axes[0, 0].axvline(df['Quantity'].median(), color='red', linestyle='--', label=f'Mediana: {df["Quantity"].median():.0f}')
axes[0, 0].legend()

df['UnitPrice'].hist(bins=100, ax=axes[0, 1], edgecolor='black', alpha=0.7, color='orange')
axes[0, 1].set_title('Distribuicao de UnitPrice (Bruto)', fontweight='bold')
axes[0, 1].set_xlabel('UnitPrice')
axes[0, 1].set_ylabel('Frequencia')
axes[0, 1].axvline(df['UnitPrice'].median(), color='red', linestyle='--', label=f'Mediana: {df["UnitPrice"].median():.2f}')
axes[0, 1].legend()

df.boxplot(column='Quantity', ax=axes[1, 0])
axes[1, 0].set_title('Boxplot - Quantity', fontweight='bold')

df.boxplot(column='UnitPrice', ax=axes[1, 1])
axes[1, 1].set_title('Boxplot - UnitPrice', fontweight='bold')

plt.tight_layout()
plt.savefig(os.path.join(FIGURES_PATH, 'eda_distributions_raw.png'), dpi=150, bbox_inches='tight')
plt.close()
print("  Salvo: eda_distributions_raw.png")

# %%
# Plot 3: Top 10 paises
fig, ax = plt.subplots(figsize=(10, 6))
country_counts = df['Country'].value_counts().head(10)
country_counts.plot(kind='barh', ax=ax, color=sns.color_palette('Set2', 10))
ax.set_title('Top 10 Paises por Numero de Transacoes', fontweight='bold')
ax.set_xlabel('Numero de Transacoes')
ax.invert_yaxis()
plt.tight_layout()
plt.savefig(os.path.join(FIGURES_PATH, 'eda_top_countries.png'), dpi=150, bbox_inches='tight')
plt.close()
print("  Salvo: eda_top_countries.png")

# %%
# Plot 4: Transacoes ao longo do tempo
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 8))

daily_transactions = df.set_index('InvoiceDate').resample('D').size()
daily_transactions.plot(ax=ax1, linewidth=1, color='steelblue')
ax1.set_title('Transacoes Diarias', fontweight='bold')
ax1.set_ylabel('Numero de Transacoes')
ax1.grid(True, alpha=0.3)

monthly_revenue = df.assign(TotalPrice=df['Quantity'] * df['UnitPrice']).set_index('InvoiceDate')['TotalPrice'].resample('ME').sum()
monthly_revenue.plot(ax=ax2, linewidth=2, color='green', marker='o')
ax2.set_title('Receita Mensal', fontweight='bold')
ax2.set_ylabel('Receita Total ($)')
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(os.path.join(FIGURES_PATH, 'eda_time_series.png'), dpi=150, bbox_inches='tight')
plt.close()
print("  Salvo: eda_time_series.png")

# %%
# Plot 5: Distribuicao por dia da semana e hora
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

df['DayOfWeek'] = df['InvoiceDate'].dt.dayofweek
day_names = ['Seg', 'Ter', 'Qua', 'Qui', 'Sex', 'Sab', 'Dom']
day_counts = df['DayOfWeek'].value_counts().sort_index()
ax1.bar(range(len(day_counts)), day_counts.values, color=sns.color_palette('Set2', 7))
ax1.set_xticks(range(len(day_counts)))
ax1.set_xticklabels([day_names[i] for i in day_counts.index])
ax1.set_title('Transacoes por Dia da Semana', fontweight='bold')
ax1.set_ylabel('Contagem')

df['Hour'] = df['InvoiceDate'].dt.hour
hour_counts = df['Hour'].value_counts().sort_index()
ax2.bar(hour_counts.index, hour_counts.values, color='steelblue', alpha=0.7)
ax2.set_title('Transacoes por Hora', fontweight='bold')
ax2.set_xlabel('Hora')
ax2.set_ylabel('Contagem')

plt.tight_layout()
plt.savefig(os.path.join(FIGURES_PATH, 'eda_temporal_patterns.png'), dpi=150, bbox_inches='tight')
plt.close()
print("  Salvo: eda_temporal_patterns.png")

print("\n--- EDA completa! ---")

# %%
# ============================================================
# FASE 3: PREPARACAO DOS DADOS E FEATURE ENGINEERING
# ============================================================
print("\n" + "=" * 60)
print("FASE 3: PREPARACAO DOS DADOS E FEATURE ENGINEERING")
print("=" * 60)

# --- Limpeza dos dados ---
print("\n--- Limpeza dos dados ---")
print(f"Linhas iniciais: {len(df):,}")

df_clean = clean_data(df)

# Converter CustomerID para int (clean_data mantem como float)
df_clean['CustomerID'] = df_clean['CustomerID'].astype(int)

print(f"Linhas finais: {len(df_clean):,}")
print(f"Retidas: {len(df_clean) / len(df) * 100:.1f}%")

# Salvar dados limpos
df_clean.to_csv(os.path.join(PROCESSED_PATH, 'cleaned_data.csv'), index=False)
print(f"  Dados limpos salvos em: data/processed/cleaned_data.csv")

# %%
# --- Feature Engineering RFM ---
print("\n--- Feature Engineering RFM ---")

analysis_date = df_clean['InvoiceDate'].max() + pd.Timedelta(days=1)
print(f"Data de analise (referencia): {analysis_date}")

rfm_df = calculate_rfm_features(df_clean, analysis_date=analysis_date)

print(f"\nClientes totais: {len(rfm_df):,}")
print(f"Features: {list(rfm_df.columns)}")
print(f"\nEstatisticas RFM:")
print(rfm_df[['Recency', 'Frequency', 'Monetary']].describe())

# %%
print("\n--- RFM Scoring ---")
print(f"Distribuicao R_Score:\n{rfm_df['R_Score'].value_counts().sort_index()}")
print(f"\nDistribuicao F_Score:\n{rfm_df['F_Score'].value_counts().sort_index()}")
print(f"\nDistribuicao M_Score:\n{rfm_df['M_Score'].value_counts().sort_index()}")

# %%
# Salvar features RFM
rfm_df.to_csv(os.path.join(FEATURES_PATH, 'customer_rfm_features.csv'), index=False)
print(f"\nFeatures RFM salvas em: data/features/customer_rfm_features.csv")

# %%
# Visualizacoes RFM
print("\n--- Visualizacoes RFM ---")

# Plot: Distribuicoes RFM
fig, axes = plt.subplots(2, 3, figsize=(18, 10))

for idx, (col, title) in enumerate([
    ('Recency', 'Recency (dias)'),
    ('Frequency', 'Frequencia (transacoes)'),
    ('Monetary', 'Valor Monetario ($)'),
    ('CustomerTenure', 'Tenure do Cliente (dias)'),
    ('AvgOrderValue', 'Valor Medio do Pedido ($)'),
    ('ProductDiversity', 'Diversidade de Produtos')
]):
    row, c = divmod(idx, 3)
    rfm_df[col].hist(bins=50, ax=axes[row, c], edgecolor='black', alpha=0.7)
    axes[row, c].set_title(f'Distribuicao - {title}', fontweight='bold')
    axes[row, c].set_xlabel(title)
    axes[row, c].set_ylabel('Frequencia')
    axes[row, c].axvline(rfm_df[col].median(), color='red', linestyle='--',
                          label=f'Mediana: {rfm_df[col].median():.1f}')
    axes[row, c].legend(fontsize=8)

plt.tight_layout()
plt.savefig(os.path.join(FIGURES_PATH, 'rfm_distributions.png'), dpi=150, bbox_inches='tight')
plt.close()
print("  Salvo: rfm_distributions.png")

# %%
# Plot: Correlacao RFM
fig, ax = plt.subplots(figsize=(12, 10))
rfm_numeric = rfm_df[['Recency', 'Frequency', 'Monetary', 'AvgOrderValue',
                       'OrderValueStd', 'TotalUnits', 'ProductDiversity',
                       'CustomerTenure', 'AvgPurchaseInterval', 'AvgItemsPerTransaction']]
corr_matrix = rfm_numeric.corr()
sns.heatmap(corr_matrix, annot=True, fmt='.2f', cmap='coolwarm', center=0,
            square=True, linewidths=1, cbar_kws={'shrink': 0.8}, ax=ax)
ax.set_title('Matriz de Correlacao - Features RFM', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig(os.path.join(FIGURES_PATH, 'rfm_correlation_matrix.png'), dpi=150, bbox_inches='tight')
plt.close()
print("  Salvo: rfm_correlation_matrix.png")

# %%
# Plot: Boxplots RFM
fig, axes = plt.subplots(1, 3, figsize=(15, 5))
for idx, col in enumerate(['Recency', 'Frequency', 'Monetary']):
    rfm_df.boxplot(column=col, ax=axes[idx])
    axes[idx].set_title(f'Boxplot - {col}', fontweight='bold')
plt.tight_layout()
plt.savefig(os.path.join(FIGURES_PATH, 'rfm_boxplots.png'), dpi=150, bbox_inches='tight')
plt.close()
print("  Salvo: rfm_boxplots.png")

# %%
# Plot: Scatter RFM pairwise
fig, axes = plt.subplots(1, 3, figsize=(18, 5))

axes[0].scatter(rfm_df['Recency'], rfm_df['Frequency'], alpha=0.3, s=10, c='steelblue')
axes[0].set_xlabel('Recency (dias)')
axes[0].set_ylabel('Frequency')
axes[0].set_title('Recency vs Frequency', fontweight='bold')

axes[1].scatter(rfm_df['Recency'], rfm_df['Monetary'], alpha=0.3, s=10, c='green')
axes[1].set_xlabel('Recency (dias)')
axes[1].set_ylabel('Monetary ($)')
axes[1].set_title('Recency vs Monetary', fontweight='bold')

axes[2].scatter(rfm_df['Frequency'], rfm_df['Monetary'], alpha=0.3, s=10, c='coral')
axes[2].set_xlabel('Frequency')
axes[2].set_ylabel('Monetary ($)')
axes[2].set_title('Frequency vs Monetary', fontweight='bold')

plt.tight_layout()
plt.savefig(os.path.join(FIGURES_PATH, 'rfm_scatter_pairwise.png'), dpi=150, bbox_inches='tight')
plt.close()
print("  Salvo: rfm_scatter_pairwise.png")

# %%
# ============================================================
# FASE 4: MODELAGEM - CLUSTERING
# ============================================================
print("\n" + "=" * 60)
print("FASE 4: MODELAGEM - CLUSTERING")
print("=" * 60)

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans, AgglomerativeClustering, DBSCAN
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score
from sklearn.decomposition import PCA
from scipy.cluster.hierarchy import dendrogram, linkage
from scipy.spatial.distance import cdist

# --- Hopkins Statistic ---
print("\n--- Hopkins Statistic (tendencia a clustering) ---")

def hopkins_statistic(X, sample_size=None, random_state=42):
    n, d = X.shape
    if sample_size is None:
        sample_size = min(n // 10, 200)
    rng = np.random.RandomState(random_state)
    sample_idx = rng.choice(n, sample_size, replace=False)
    X_sample = X[sample_idx]
    X_uniform = rng.uniform(X.min(axis=0), X.max(axis=0), size=(sample_size, d))
    nn_sample = cdist(X_sample, X[np.setdiff1d(np.arange(n), sample_idx)]).min(axis=1)
    nn_uniform = cdist(X_uniform, X).min(axis=1)
    return nn_uniform.sum() / (nn_uniform.sum() + nn_sample.sum())

rfm_for_hopkins = rfm_df[['Recency', 'Frequency', 'Monetary']].values
hopkins_val = hopkins_statistic(rfm_for_hopkins)
print(f"  Hopkins statistic: {hopkins_val:.4f}")
print(f"  Interpretacao: {'dados tem tendencia a formar clusters' if hopkins_val > 0.5 else 'dados podem ser uniformes'}")
print(f"  (> 0.5 = tendencia, > 0.7 = forte tendencia, ~0.5 = aleatorio)")

# Features para clustering (do config.yaml)
clustering_features = yaml_config['modeling'].get('features_to_use', ['Recency', 'Frequency', 'Monetary'])
feature_transformation = yaml_config['modeling'].get('feature_transformation', 'log1p')
X_cluster = rfm_df[clustering_features].copy()

# Aplicar transformacao para reduzir skewness
print(f"\nAplicando transformacao '{feature_transformation}' para reduzir assimetria...")
X_log = X_cluster.copy()
for col in clustering_features:
    if feature_transformation == 'log1p':
        X_log[col] = np.log1p(X_cluster[col])
    elif feature_transformation == 'sqrt':
        X_log[col] = np.sqrt(X_cluster[col])
    # None/identity: sem transformacao
    print(f"  {col}: skew antes={X_cluster[col].skew():.2f}, depois={X_log[col].skew():.2f}")

# Visualizacao de skewness antes/depois
skew_before = X_cluster.skew()
skew_after = X_log.skew()
skew_comparison = pd.DataFrame({
    'Antes': skew_before,
    'Depois': skew_after,
    'Reducao (%)': ((skew_before.abs() - skew_after.abs()) / skew_before.abs() * 100).round(1)
})
skew_comparison.to_csv(os.path.join(REPORTS_PATH, 'skewness_comparison.csv'))
print(f"\n  Tabela de skewness salva: reports/skewness_comparison.csv")

fig, ax = plt.subplots(figsize=(10, 5))
x_pos = np.arange(len(clustering_features))
width = 0.35
ax.bar(x_pos - width/2, skew_before.values, width, label='Antes (original)', color='coral')
ax.bar(x_pos + width/2, skew_after.values, width, label=f'Depois ({feature_transformation})', color='steelblue')
ax.set_xticks(x_pos)
ax.set_xticklabels(clustering_features)
ax.set_ylabel('Skewness')
ax.set_title('Skewness antes e depois da transformacao', fontweight='bold')
ax.legend()
ax.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
ax.grid(True, alpha=0.3, axis='y')
plt.tight_layout()
plt.savefig(os.path.join(FIGURES_PATH, 'skewness_comparison.png'), dpi=150, bbox_inches='tight')
plt.close()
print("  Salvo: skewness_comparison.png")

# Padronizar (StandardScaler)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_log)
print(f"\nDados escalados: shape = {X_scaled.shape}")

# %%
# --- Comparacao de feature sets ---
print("\n--- Comparacao de feature sets ---")
extended_features = ['Recency', 'Frequency', 'Monetary', 'ProductDiversity', 'AvgOrderValue']
extended_available = all(f in rfm_df.columns for f in extended_features)

if extended_available:
    X_ext = rfm_df[extended_features].copy().apply(np.log1p)
    scaler_ext = StandardScaler()
    X_ext_scaled = scaler_ext.fit_transform(X_ext)

    # Testar com K=4 para ambos
    test_k = 4
    km_rfm = KMeans(n_clusters=test_k, n_init=10, random_state=SEED)
    km_ext = KMeans(n_clusters=test_k, n_init=10, random_state=SEED)

    labels_rfm = km_rfm.fit_predict(X_scaled)
    labels_ext = km_ext.fit_predict(X_ext_scaled)

    sil_rfm = silhouette_score(X_scaled, labels_rfm)
    sil_ext = silhouette_score(X_ext_scaled, labels_ext)

    print(f"  RFM-only ({clustering_features}): Silhouette = {sil_rfm:.4f}")
    print(f"  RFM-extended ({extended_features}): Silhouette = {sil_ext:.4f}")

    if sil_ext > sil_rfm + 0.01:
        print(f"  -> RFM-extended e melhor. Usando features estendidas.")
        clustering_features = extended_features
        X_cluster = rfm_df[clustering_features].copy()
        X_log = X_cluster.apply(np.log1p)
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X_log)
    else:
        print(f"  -> RFM-only e igual ou melhor. Mantendo features originais.")
else:
    print("  Features estendidas nao disponiveis. Mantendo RFM-only.")

# %%
# --- Encontrar K otimo ---
print("\n--- Encontrando numero otimo de clusters (K) ---")

k_range = range(2, 11)
inertias = []
silhouette_scores = []
db_scores = []
ch_scores = []

for k in k_range:
    kmeans = KMeans(n_clusters=k, n_init=10, max_iter=300, random_state=SEED)
    labels = kmeans.fit_predict(X_scaled)
    inertias.append(kmeans.inertia_)

    sil = silhouette_score(X_scaled, labels)
    db = davies_bouldin_score(X_scaled, labels)
    ch = calinski_harabasz_score(X_scaled, labels)

    silhouette_scores.append(sil)
    db_scores.append(db)
    ch_scores.append(ch)

    print(f"  K={k}: Silhouette={sil:.4f}, Davies-Bouldin={db:.4f}, Calinski-Harabasz={ch:.1f}")

# %%
# Plot: Elbow + Silhouette + Davies-Bouldin
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

axes[0, 0].plot(list(k_range), inertias, 'bo-', linewidth=2, markersize=8)
axes[0, 0].set_title('Metodo do Cotovelo (Elbow)', fontweight='bold')
axes[0, 0].set_xlabel('Numero de Clusters (K)')
axes[0, 0].set_ylabel('Inercia (WSS)')
axes[0, 0].grid(True, alpha=0.3)

best_k_sil = list(k_range)[np.argmax(silhouette_scores)]
axes[0, 1].plot(list(k_range), silhouette_scores, 'go-', linewidth=2, markersize=8)
axes[0, 1].scatter([best_k_sil], [max(silhouette_scores)], c='red', s=200, marker='*', zorder=5,
                    label=f'Melhor K={best_k_sil}')
axes[0, 1].set_title('Silhouette Score', fontweight='bold')
axes[0, 1].set_xlabel('Numero de Clusters (K)')
axes[0, 1].set_ylabel('Silhouette Score')
axes[0, 1].legend()
axes[0, 1].grid(True, alpha=0.3)

best_k_db = list(k_range)[np.argmin(db_scores)]
axes[1, 0].plot(list(k_range), db_scores, 'ro-', linewidth=2, markersize=8)
axes[1, 0].scatter([best_k_db], [min(db_scores)], c='blue', s=200, marker='*', zorder=5,
                    label=f'Melhor K={best_k_db}')
axes[1, 0].set_title('Davies-Bouldin Index (menor = melhor)', fontweight='bold')
axes[1, 0].set_xlabel('Numero de Clusters (K)')
axes[1, 0].set_ylabel('Davies-Bouldin Index')
axes[1, 0].legend()
axes[1, 0].grid(True, alpha=0.3)

best_k_ch = list(k_range)[np.argmax(ch_scores)]
axes[1, 1].plot(list(k_range), ch_scores, 'mo-', linewidth=2, markersize=8)
axes[1, 1].scatter([best_k_ch], [max(ch_scores)], c='red', s=200, marker='*', zorder=5,
                    label=f'Melhor K={best_k_ch}')
axes[1, 1].set_title('Calinski-Harabasz Index (maior = melhor)', fontweight='bold')
axes[1, 1].set_xlabel('Numero de Clusters (K)')
axes[1, 1].set_ylabel('Calinski-Harabasz Index')
axes[1, 1].legend()
axes[1, 1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(os.path.join(FIGURES_PATH, 'optimal_k_analysis.png'), dpi=150, bbox_inches='tight')
plt.close()
print(f"\nMelhor K por Silhouette: {best_k_sil}")
print(f"Melhor K por Davies-Bouldin: {best_k_db}")
print(f"Melhor K por Calinski-Harabasz: {best_k_ch}")
print("  Salvo: optimal_k_analysis.png")

# Decidir K otimo
if 4 <= best_k_sil <= 6:
    optimal_k = best_k_sil
else:
    k_candidates = [k for k in range(4, 7)]
    k_sil_candidates = [(k, silhouette_scores[k-2]) for k in k_candidates]
    optimal_k = max(k_sil_candidates, key=lambda x: x[1])[0]

print(f"\nK otimo selecionado: {optimal_k}")

# %%
# --- Comparacao de Algoritmos ---
print("\n--- Comparacao de Algoritmos ---")

# 1. KMeans
kmeans = KMeans(n_clusters=optimal_k, n_init=10, max_iter=300, random_state=SEED)
kmeans_labels = kmeans.fit_predict(X_scaled)

# 2. Hierarchical (Ward)
hierarchical = AgglomerativeClustering(n_clusters=optimal_k, linkage='ward')
hier_labels = hierarchical.fit_predict(X_scaled)

# 3. DBSCAN (eps automatico)
from sklearn.neighbors import NearestNeighbors
nn = NearestNeighbors(n_neighbors=10)
nn.fit(X_scaled)
distances, _ = nn.kneighbors(X_scaled)
sorted_distances = np.sort(distances[:, -1])
eps_auto = np.percentile(sorted_distances, 90)
dbscan = DBSCAN(eps=eps_auto, min_samples=10)
dbscan_labels = dbscan.fit_predict(X_scaled)

results_comparison = []
for name, labels in [('KMeans', kmeans_labels), ('Hierarchical', hier_labels), ('DBSCAN', dbscan_labels)]:
    n_clusters_found = len(set(labels)) - (1 if -1 in labels else 0)
    n_noise = (labels == -1).sum()

    if n_clusters_found >= 2:
        if n_noise > 0:
            mask = labels != -1
            sil = silhouette_score(X_scaled[mask], labels[mask])
            db = davies_bouldin_score(X_scaled[mask], labels[mask])
            ch = calinski_harabasz_score(X_scaled[mask], labels[mask])
        else:
            sil = silhouette_score(X_scaled, labels)
            db = davies_bouldin_score(X_scaled, labels)
            ch = calinski_harabasz_score(X_scaled, labels)
    else:
        sil = db = ch = np.nan

    results_comparison.append({
        'Algoritmo': name,
        'N_Clusters': n_clusters_found,
        'N_Noise': n_noise,
        'Silhouette': sil,
        'Davies_Bouldin': db,
        'Calinski_Harabasz': ch
    })
    print(f"  {name}: K={n_clusters_found}, Noise={n_noise}, Sil={sil:.4f}, DB={db:.4f}, CH={ch:.1f}")

comparison_df = pd.DataFrame(results_comparison)
print(f"\n{comparison_df.to_string(index=False)}")

# %%
# --- Dendrograma ---
print("\n--- Dendrograma ---")
Z = linkage(X_scaled[:2000], method='ward')

fig, ax = plt.subplots(figsize=(14, 7))
dendrogram(Z, truncate_mode='level', p=5, ax=ax, color_threshold=0,
           above_threshold_color='steelblue')
ax.set_title('Dendrograma - Clustering Hierarquico (Ward)', fontweight='bold')
ax.set_xlabel('Amostras')
ax.set_ylabel('Distancia')
ax.axhline(y=Z[-optimal_k+1, 2], color='red', linestyle='--', label=f'Corte em K={optimal_k}')
ax.legend()
plt.tight_layout()
plt.savefig(os.path.join(FIGURES_PATH, 'dendrogram.png'), dpi=150, bbox_inches='tight')
plt.close()
print("  Salvo: dendrogram.png")

# %%
# ============================================================
# FASE 5: OTIMIZACAO DE HIPERPARAMETROS (Optuna)
# ============================================================
print("\n" + "=" * 60)
print("FASE 5: OTIMIZACAO COM OPTUNA")
print("=" * 60)

import optuna
optuna.logging.set_verbosity(optuna.logging.WARNING)

def objective(trial):
    n_clusters = trial.suggest_int('n_clusters', 3, 8)
    n_init = trial.suggest_int('n_init', 5, 20)
    max_iter = trial.suggest_int('max_iter', 200, 500)
    algorithm = trial.suggest_categorical('algorithm', ['lloyd', 'elkan'])

    km = KMeans(
        n_clusters=n_clusters,
        n_init=n_init,
        max_iter=max_iter,
        algorithm=algorithm,
        random_state=SEED
    )
    labels = km.fit_predict(X_scaled)

    sil = silhouette_score(X_scaled, labels)

    # Penalizar clusters muito pequenos (< 5%)
    min_cluster_pct = min(np.bincount(labels)) / len(labels)
    if min_cluster_pct < 0.05:
        sil *= 0.8

    return sil

study = optuna.create_study(direction='maximize',
                             sampler=optuna.samplers.TPESampler(seed=SEED))
study.optimize(objective, n_trials=50, show_progress_bar=False)

print(f"\nMelhor trial:")
print(f"  Silhouette Score: {study.best_value:.4f}")
print(f"  Parametros: {study.best_params}")

# %%
# --- Modelo final ---
best_params = study.best_params
best_n_clusters = best_params['n_clusters']

if not (4 <= best_n_clusters <= 6):
    print(f"\nOptuna sugeriu K={best_n_clusters}, mas por requisito de negocio mantendo K={optimal_k}")
    best_n_clusters = optimal_k
else:
    optimal_k = best_n_clusters

final_kmeans = KMeans(
    n_clusters=best_n_clusters,
    n_init=best_params['n_init'],
    max_iter=best_params['max_iter'],
    algorithm=best_params['algorithm'],
    random_state=SEED
)
final_labels = final_kmeans.fit_predict(X_scaled)

print(f"\nModelo final treinado com K={best_n_clusters}")
print(f"Distribuicao dos clusters:")
for i in range(best_n_clusters):
    count = (final_labels == i).sum()
    pct = count / len(final_labels) * 100
    print(f"  Cluster {i}: {count:,} clientes ({pct:.1f}%)")

# %%
# ============================================================
# FASE 6: AVALIACAO FINAL
# ============================================================
print("\n" + "=" * 60)
print("FASE 6: AVALIACAO FINAL")
print("=" * 60)

from sklearn.metrics import silhouette_samples

final_sil = silhouette_score(X_scaled, final_labels)
final_db = davies_bouldin_score(X_scaled, final_labels)
final_ch = calinski_harabasz_score(X_scaled, final_labels)

test_metrics = {
    'silhouette_score': final_sil,
    'davies_bouldin_index': final_db,
    'calinski_harabasz_index': final_ch
}

print(f"\nMetricas do Modelo Final:")
print(f"  Silhouette Score: {final_sil:.4f} (limiar: >= {config['threshold']})")
print(f"  Davies-Bouldin Index: {final_db:.4f} (limiar: < 1.5)")
print(f"  Calinski-Harabasz Index: {final_ch:.1f}")

sil_pass = final_sil >= config['threshold']
db_pass = final_db < 1.5
print(f"\n  Silhouette: {'APROVADO' if sil_pass else 'REPROVADO'}")
print(f"  Davies-Bouldin: {'APROVADO' if db_pass else 'REPROVADO'}")

# %%
# Silhouette por cluster
print("\n--- Silhouette por Cluster ---")
sil_samples = silhouette_samples(X_scaled, final_labels)

for cluster_id in range(best_n_clusters):
    cluster_sil = sil_samples[final_labels == cluster_id]
    print(f"  Cluster {cluster_id}: media={cluster_sil.mean():.4f}, std={cluster_sil.std():.4f}, "
          f"min={cluster_sil.min():.4f}, max={cluster_sil.max():.4f}")

# %%
# Plot: Silhouette Analysis
fig, ax = plt.subplots(figsize=(10, 7))

y_lower = 10
colors = plt.cm.Set2(np.linspace(0, 1, best_n_clusters))

for i in range(best_n_clusters):
    cluster_sil = sil_samples[final_labels == i]
    cluster_sil.sort()
    size = len(cluster_sil)
    y_upper = y_lower + size

    ax.fill_betweenx(np.arange(y_lower, y_upper), 0, cluster_sil,
                      facecolor=colors[i], edgecolor=colors[i], alpha=0.7)
    ax.text(-0.05, y_lower + 0.5 * size, f'C{i}', fontweight='bold')
    y_lower = y_upper + 10

ax.axvline(x=final_sil, color='red', linestyle='--', linewidth=2,
            label=f'Media: {final_sil:.3f}')
ax.set_title('Analise de Silhouette por Cluster', fontsize=14, fontweight='bold')
ax.set_xlabel('Coeficiente de Silhouette')
ax.set_ylabel('Cluster')
ax.set_xlim([-0.1, 1])
ax.legend(loc='best')
ax.grid(True, alpha=0.3, axis='x')
plt.tight_layout()
plt.savefig(os.path.join(FIGURES_PATH, 'silhouette_analysis.png'), dpi=150, bbox_inches='tight')
plt.close()
print("  Salvo: silhouette_analysis.png")

# %%
# --- Kruskal-Wallis entre segmentos ---
print("\n--- Kruskal-Wallis (diferenca entre segmentos) ---")
from scipy.stats import kruskal

rfm_df['Cluster'] = final_labels
kw_results = {}
rfm_test_features = ['Recency', 'Frequency', 'Monetary']
for feat in rfm_test_features:
    groups = [rfm_df.loc[rfm_df['Cluster'] == c, feat].values for c in range(best_n_clusters)]
    stat, pval = kruskal(*groups)
    kw_results[feat] = {'H_statistic': stat, 'p_value': pval}
    sig = 'SIM' if pval < 0.05 else 'NAO'
    print(f"  {feat}: H={stat:.2f}, p={pval:.2e} -> Diferenca significativa: {sig}")

kw_df = pd.DataFrame(kw_results).T
kw_df.to_csv(os.path.join(REPORTS_PATH, 'kruskal_wallis_results.csv'))
print("  Salvo: reports/kruskal_wallis_results.csv")

# %%
# --- Customer-level train/test split ---
print("\n--- Validacao train/test (80/20 por cliente) ---")
from sklearn.model_selection import train_test_split

customer_ids = rfm_df['CustomerID'].values
train_idx, test_idx = train_test_split(
    np.arange(len(rfm_df)), test_size=0.2, random_state=SEED
)

X_train = X_scaled[train_idx]
X_test = X_scaled[test_idx]

km_train = KMeans(
    n_clusters=best_n_clusters, n_init=best_params['n_init'],
    max_iter=best_params['max_iter'], algorithm=best_params['algorithm'],
    random_state=SEED
)
train_labels = km_train.fit_predict(X_train)
test_labels = km_train.predict(X_test)

sil_train = silhouette_score(X_train, train_labels)
sil_test = silhouette_score(X_test, test_labels)

print(f"  Train Silhouette: {sil_train:.4f} (n={len(train_idx)})")
print(f"  Test Silhouette:  {sil_test:.4f} (n={len(test_idx)})")
generalization_gap = abs(sil_train - sil_test)
print(f"  Gap: {generalization_gap:.4f}")
print(f"  Generalizacao: {'BOA (gap < 0.05)' if generalization_gap < 0.05 else 'VERIFICAR (gap >= 0.05)'}")

# %%
# --- Perfil dos Clusters ---
print("\n--- Perfil dos Clusters ---")
rfm_df['Cluster'] = final_labels

cluster_summary = rfm_df.groupby('Cluster').agg({
    'Recency': 'mean',
    'Frequency': 'mean',
    'Monetary': ['mean', 'sum'],
    'AvgOrderValue': 'mean',
    'ProductDiversity': 'mean',
    'CustomerTenure': 'mean',
    'CustomerID': 'count'
}).round(2)

cluster_summary.columns = ['Recency_Mean', 'Frequency_Mean', 'Monetary_Mean',
                            'Monetary_Total', 'AvgOrderValue_Mean',
                            'ProductDiversity_Mean', 'Tenure_Mean', 'N_Customers']
cluster_summary['Pct_Customers'] = (cluster_summary['N_Customers'] / cluster_summary['N_Customers'].sum() * 100).round(1)
cluster_summary['Pct_Revenue'] = (cluster_summary['Monetary_Total'] / cluster_summary['Monetary_Total'].sum() * 100).round(1)

print(cluster_summary.to_string())

# %%
# --- Nomes de Segmentos ---
print("\n--- Atribuindo Nomes de Segmentos ---")

def assign_segment_name(row, summary_df):
    r = row['Recency_Mean']
    f = row['Frequency_Mean']
    m = row['Monetary_Mean']

    r_median = summary_df['Recency_Mean'].median()
    f_median = summary_df['Frequency_Mean'].median()
    m_median = summary_df['Monetary_Mean'].median()

    r_low = r <= r_median
    f_high = f >= f_median
    m_high = m >= m_median

    if r_low and f_high and m_high:
        return 'Champions'
    elif r_low and f_high:
        return 'Clientes Leais'
    elif m_high and not f_high:
        return 'Grandes Gastadores'
    elif not r_low and f_high:
        return 'Em Risco'
    elif r_low and not f_high:
        return 'Novos Clientes'
    elif not r_low and not f_high and not m_high:
        return 'Hibernando'
    else:
        return 'Outros'

segment_names = {}
for idx, row in cluster_summary.iterrows():
    name = assign_segment_name(row, cluster_summary)
    segment_names[idx] = name
    print(f"  Cluster {idx}: {name}")
    print(f"    Recency={row['Recency_Mean']:.0f}d, Freq={row['Frequency_Mean']:.1f}, "
          f"Monetary=${row['Monetary_Mean']:.0f}, N={row['N_Customers']:,} ({row['Pct_Customers']}%)")

rfm_df['Segment_Name'] = rfm_df['Cluster'].map(segment_names)

# %%
# ============================================================
# FASE 7: INTERPRETACAO (SHAP)
# ============================================================
print("\n" + "=" * 60)
print("FASE 7: INTERPRETACAO COM SHAP")
print("=" * 60)

from sklearn.ensemble import RandomForestClassifier
import shap

# Random Forest proxy
rf_proxy = RandomForestClassifier(n_estimators=200, random_state=SEED, n_jobs=-1)
rf_proxy.fit(X_scaled, final_labels)
rf_accuracy = rf_proxy.score(X_scaled, final_labels)
print(f"Random Forest proxy accuracy: {rf_accuracy:.4f}")

feature_importance = pd.DataFrame({
    'Feature': clustering_features,
    'Importance': rf_proxy.feature_importances_
}).sort_values('Importance', ascending=False)
print(f"\nFeature Importance (Random Forest):")
print(feature_importance.to_string(index=False))

# %%
# SHAP values
print("\n--- Calculando SHAP values ---")
explainer = shap.TreeExplainer(rf_proxy)

if len(X_scaled) > 2000:
    np.random.seed(SEED)
    sample_idx = np.random.choice(len(X_scaled), 2000, replace=False)
    X_shap = X_scaled[sample_idx]
else:
    X_shap = X_scaled

shap_values = explainer.shap_values(X_shap)

# %%
# Plot: SHAP bar (importancia media)
if isinstance(shap_values, list):
    mean_abs_shap = np.zeros(len(clustering_features))
    for sv_class in shap_values:
        mean_abs_shap += np.abs(sv_class).mean(axis=0)
    mean_abs_shap /= len(shap_values)
elif shap_values.ndim == 3:
    mean_abs_shap = np.abs(shap_values).mean(axis=(0, 2))
else:
    mean_abs_shap = np.abs(shap_values).mean(axis=0)

fig, ax = plt.subplots(figsize=(10, 6))
sorted_idx = np.argsort(mean_abs_shap)
ax.barh(range(len(clustering_features)), mean_abs_shap[sorted_idx], color='steelblue')
ax.set_yticks(range(len(clustering_features)))
ax.set_yticklabels([clustering_features[i] for i in sorted_idx])
ax.set_xlabel('Mean |SHAP value|')
ax.set_title('SHAP - Importancia Media das Features', fontweight='bold')
ax.grid(True, alpha=0.3, axis='x')
plt.tight_layout()
plt.savefig(os.path.join(FIGURES_PATH, 'shap_bar_importance.png'), dpi=150, bbox_inches='tight')
plt.close()
print("  Salvo: shap_bar_importance.png")

# %%
# Plot: SHAP summary por classe
if isinstance(shap_values, list):
    n_classes = len(shap_values)
elif shap_values.ndim == 3:
    n_classes = shap_values.shape[2]
else:
    n_classes = 1

if n_classes > 1:
    fig, axes = plt.subplots(1, min(n_classes, 4), figsize=(5 * min(n_classes, 4), 5))
    if n_classes == 1:
        axes = [axes]
    for c_idx in range(min(n_classes, 4)):
        ax = axes[c_idx]
        if isinstance(shap_values, list):
            sv_c = shap_values[c_idx]
        else:
            sv_c = shap_values[:, :, c_idx]
        mean_abs_c = np.abs(sv_c).mean(axis=0)
        s_idx = np.argsort(mean_abs_c)
        ax.barh(range(len(clustering_features)), mean_abs_c[s_idx],
                color=plt.cm.Set2(c_idx / max(n_classes - 1, 1)))
        ax.set_yticks(range(len(clustering_features)))
        ax.set_yticklabels([clustering_features[i] for i in s_idx])
        ax.set_title(f'Cluster {c_idx}', fontweight='bold')
        ax.set_xlabel('Mean |SHAP|')
        ax.grid(True, alpha=0.3, axis='x')
    plt.suptitle('SHAP - Importancia por Cluster', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_PATH, 'shap_summary.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print("  Salvo: shap_summary.png")

# %%
# Plot: SHAP dependence
if isinstance(shap_values, list):
    sv_ref = shap_values[0]
elif shap_values.ndim == 3:
    sv_ref = shap_values[:, :, 0]
else:
    sv_ref = shap_values

for i, feat in enumerate(clustering_features):
    fig, ax = plt.subplots(figsize=(8, 5))
    shap.dependence_plot(i, sv_ref, X_shap,
                          feature_names=clustering_features,
                          show=False, ax=ax)
    plt.title(f'SHAP Dependence - {feat}', fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_PATH, f'shap_dependence_{feat.lower()}.png'),
                dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Salvo: shap_dependence_{feat.lower()}.png")

# %%
# ============================================================
# FASE 8: SALVAMENTO E RELATORIO FINAL
# ============================================================
print("\n" + "=" * 60)
print("FASE 8: SALVAMENTO E RELATORIO FINAL")
print("=" * 60)

import joblib
import json

print("\n--- Salvando artefatos ---")

joblib.dump(final_kmeans, os.path.join(MODELS_PATH, 'kmeans_model.joblib'), compress=3)
print(f"  Modelo salvo: kmeans_model.joblib")

joblib.dump(scaler, os.path.join(MODELS_PATH, 'scaler.joblib'), compress=3)
print(f"  Scaler salvo: scaler.joblib")

metadata = {
    'problem_type': config['problem_type'],
    'best_model': 'KMeans',
    'best_params': {
        'n_clusters': best_n_clusters,
        'n_init': best_params['n_init'],
        'max_iter': best_params['max_iter'],
        'algorithm': best_params['algorithm']
    },
    'test_metrics': test_metrics,
    'features_used': clustering_features,
    'log_transform': True,
    'scaler': 'StandardScaler',
    'training_samples': len(rfm_df),
    'segment_names': segment_names,
    'timestamp': datetime.now().isoformat()
}

with open(os.path.join(MODELS_PATH, 'model_metadata.json'), 'w') as f:
    json.dump(metadata, f, indent=2, default=str)
print(f"  Metadata salva: model_metadata.json")

rfm_df.to_csv(os.path.join(REPORTS_PATH, 'customer_segments.csv'), index=False)
print(f"  Segmentos salvos: customer_segments.csv")

# %%
# --- Visualizacoes Finais ---
print("\n--- Gerando visualizacoes finais ---")

# Plot: PCA 2D
pca_2d = PCA(n_components=2, random_state=SEED)
X_pca_2d = pca_2d.fit_transform(X_scaled)

fig, ax = plt.subplots(figsize=(12, 9))
colors = plt.cm.Set2(np.linspace(0, 1, best_n_clusters))

for i in range(best_n_clusters):
    mask = final_labels == i
    ax.scatter(X_pca_2d[mask, 0], X_pca_2d[mask, 1],
               c=[colors[i]], label=f'{segment_names[i]} (n={mask.sum():,})',
               s=30, alpha=0.6, edgecolors='black', linewidth=0.3)

ax.set_xlabel(f'PC1 ({pca_2d.explained_variance_ratio_[0]:.1%} variancia)', fontsize=12)
ax.set_ylabel(f'PC2 ({pca_2d.explained_variance_ratio_[1]:.1%} variancia)', fontsize=12)
ax.set_title('Segmentos de Clientes (Projecao PCA 2D)', fontsize=14, fontweight='bold')
ax.legend(loc='best', fontsize=9)
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(FIGURES_PATH, 'clusters_pca_2d.png'), dpi=150, bbox_inches='tight')
plt.close()
print("  Salvo: clusters_pca_2d.png")

# %%
# Plot: Perfil dos clusters
fig, axes = plt.subplots(1, 2, figsize=(16, 6))

plot_data = cluster_summary[['Recency_Mean', 'Frequency_Mean', 'Monetary_Mean']].copy()
plot_data.index = [segment_names[i] for i in plot_data.index]
plot_data.plot(kind='bar', ax=axes[0], color=['steelblue', 'green', 'coral'])
axes[0].set_title('Perfil dos Segmentos (Valores Absolutos)', fontweight='bold')
axes[0].set_xlabel('Segmento')
axes[0].set_ylabel('Valor Medio')
axes[0].legend(['Recency (dias)', 'Frequency', 'Monetary ($)'])
axes[0].tick_params(axis='x', rotation=45)
axes[0].grid(True, alpha=0.3, axis='y')

plot_norm = (plot_data - plot_data.min()) / (plot_data.max() - plot_data.min())
plot_norm.plot(kind='bar', ax=axes[1], color=['steelblue', 'green', 'coral'])
axes[1].set_title('Perfil dos Segmentos (Normalizado 0-1)', fontweight='bold')
axes[1].set_xlabel('Segmento')
axes[1].set_ylabel('Valor Normalizado')
axes[1].legend(['Recency', 'Frequency', 'Monetary'])
axes[1].tick_params(axis='x', rotation=45)
axes[1].grid(True, alpha=0.3, axis='y')

plt.tight_layout()
plt.savefig(os.path.join(FIGURES_PATH, 'segment_profiles.png'), dpi=150, bbox_inches='tight')
plt.close()
print("  Salvo: segment_profiles.png")

# %%
# Plot: Tamanho dos clusters
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

cluster_sizes = rfm_df['Segment_Name'].value_counts()
cluster_sizes.plot(kind='bar', ax=ax1, color=sns.color_palette('Set2', len(cluster_sizes)))
ax1.set_title('Tamanho dos Segmentos (Contagem)', fontweight='bold')
ax1.set_xlabel('Segmento')
ax1.set_ylabel('Numero de Clientes')
ax1.tick_params(axis='x', rotation=45)
ax1.grid(True, alpha=0.3, axis='y')
for i, (idx, val) in enumerate(cluster_sizes.items()):
    ax1.text(i, val + 10, f'{val:,}', ha='center', fontsize=9)

ax2.pie(cluster_sizes.values, labels=cluster_sizes.index, autopct='%1.1f%%',
        startangle=90, colors=sns.color_palette('Set2', len(cluster_sizes)))
ax2.set_title('Tamanho dos Segmentos (%)', fontweight='bold')

plt.tight_layout()
plt.savefig(os.path.join(FIGURES_PATH, 'segment_sizes.png'), dpi=150, bbox_inches='tight')
plt.close()
print("  Salvo: segment_sizes.png")

# %%
# Plot: Heatmap
fig, ax = plt.subplots(figsize=(12, 6))

heatmap_data = cluster_summary[['Recency_Mean', 'Frequency_Mean', 'Monetary_Mean',
                                  'AvgOrderValue_Mean', 'ProductDiversity_Mean', 'Tenure_Mean']].copy()
heatmap_data.index = [segment_names[i] for i in heatmap_data.index]
heatmap_data.columns = ['Recency', 'Frequencia', 'Valor Monetario',
                         'Ticket Medio', 'Diversidade Produtos', 'Tenure']

heatmap_norm = (heatmap_data - heatmap_data.min()) / (heatmap_data.max() - heatmap_data.min())

sns.heatmap(heatmap_norm, annot=heatmap_data.round(1).values, fmt='',
            cmap='YlOrRd', linewidths=1, ax=ax,
            cbar_kws={'label': 'Valor Normalizado'})
ax.set_title('Perfil Detalhado dos Segmentos', fontsize=14, fontweight='bold')
ax.set_ylabel('Segmento')
plt.tight_layout()
plt.savefig(os.path.join(FIGURES_PATH, 'segment_heatmap.png'), dpi=150, bbox_inches='tight')
plt.close()
print("  Salvo: segment_heatmap.png")

# %%
# Plot: Receita por segmento
fig, ax = plt.subplots(figsize=(10, 6))
revenue_data = cluster_summary[['Monetary_Total']].copy()
revenue_data.index = [segment_names[i] for i in revenue_data.index]
revenue_data = revenue_data.sort_values('Monetary_Total', ascending=True)

revenue_data.plot(kind='barh', ax=ax, color='steelblue', legend=False)
ax.set_title('Contribuicao de Receita por Segmento', fontweight='bold')
ax.set_xlabel('Receita Total ($)')
ax.set_ylabel('Segmento')
for i, (idx, row) in enumerate(revenue_data.iterrows()):
    pct = row['Monetary_Total'] / revenue_data['Monetary_Total'].sum() * 100
    ax.text(row['Monetary_Total'] + 1000, i, f'${row["Monetary_Total"]:,.0f} ({pct:.1f}%)', va='center', fontsize=9)
ax.grid(True, alpha=0.3, axis='x')
plt.tight_layout()
plt.savefig(os.path.join(FIGURES_PATH, 'revenue_by_segment.png'), dpi=150, bbox_inches='tight')
plt.close()
print("  Salvo: revenue_by_segment.png")

# %%
# Plot 3D interativo (Plotly)
import plotly.express as px

df_3d = rfm_df[['Recency', 'Frequency', 'Monetary', 'Segment_Name', 'CustomerID']].copy()
fig_3d = px.scatter_3d(
    df_3d,
    x='Recency', y='Frequency', z='Monetary',
    color='Segment_Name',
    title='Segmentos de Clientes (Espaco RFM 3D)',
    labels={'Recency': 'Recency (dias)', 'Frequency': 'Frequencia', 'Monetary': 'Valor Monetario ($)'},
    opacity=0.6,
    hover_data=['CustomerID']
)
fig_3d.update_traces(marker=dict(size=3, line=dict(width=0.3, color='DarkSlateGrey')))
fig_3d.write_html(os.path.join(FIGURES_PATH, 'rfm_scatter_3d.html'))
print("  Salvo: rfm_scatter_3d.html")

# %%
# --- Estabilidade ---
print("\n--- Teste de Estabilidade ---")
from sklearn.metrics import adjusted_rand_score

ari_scores = []
sil_scores_stability = []

for run in range(10):
    km_test = KMeans(n_clusters=best_n_clusters, n_init=best_params['n_init'],
                     random_state=SEED + run)
    test_labels = km_test.fit_predict(X_scaled)
    ari = adjusted_rand_score(final_labels, test_labels)
    sil_test = silhouette_score(X_scaled, test_labels)
    ari_scores.append(ari)
    sil_scores_stability.append(sil_test)

print(f"  ARI medio: {np.mean(ari_scores):.4f} (+/- {np.std(ari_scores):.4f})")
print(f"  Silhouette medio: {np.mean(sil_scores_stability):.4f} (+/- {np.std(sil_scores_stability):.4f})")

stability_pass = np.mean(ari_scores) >= 0.8
print(f"  Estabilidade: {'APROVADO (ARI >= 0.8)' if stability_pass else 'REPROVADO'}")

# %%
# --- Relatorio Final ---
print("\n--- Gerando relatorio final ---")

report_path = os.path.join(REPORTS_PATH, 'final_report.txt')

with open(report_path, 'w', encoding='utf-8') as f:
    f.write("=" * 70 + "\n")
    f.write("RELATORIO FINAL - SEGMENTACAO DE CLIENTES\n")
    f.write("UCI Online Retail Dataset\n")
    f.write(f"Data: {datetime.now().strftime('%Y-%m-%d %H:%M')}\n")
    f.write("=" * 70 + "\n\n")

    f.write("1. RESUMO EXECUTIVO\n")
    f.write("-" * 70 + "\n")
    f.write(f"Clientes analisados: {len(rfm_df):,}\n")
    f.write(f"Segmentos identificados: {best_n_clusters}\n")
    f.write(f"Algoritmo: KMeans (otimizado com Optuna)\n")
    f.write(f"Features: {', '.join(clustering_features)}\n")
    f.write(f"Transformacao: log1p + StandardScaler\n\n")

    f.write("2. METRICAS DE AVALIACAO\n")
    f.write("-" * 70 + "\n")
    f.write(f"Silhouette Score: {final_sil:.4f} {'APROVADO' if sil_pass else 'REPROVADO'}\n")
    f.write(f"Davies-Bouldin Index: {final_db:.4f} {'APROVADO' if db_pass else 'REPROVADO'}\n")
    f.write(f"Calinski-Harabasz Index: {final_ch:.1f}\n")
    f.write(f"Estabilidade (ARI): {np.mean(ari_scores):.4f} {'APROVADO' if stability_pass else 'REPROVADO'}\n")
    f.write(f"Hopkins Statistic: {hopkins_val:.4f} ({'tendencia a clusters' if hopkins_val > 0.5 else 'sem tendencia'})\n")
    f.write(f"Train/Test Silhouette: {sil_train:.4f} / {sil_test:.4f} (gap: {generalization_gap:.4f})\n\n")

    f.write("  Kruskal-Wallis (H0: segmentos identicos):\n")
    for feat, res in kw_results.items():
        sig = 'p < 0.05' if res['p_value'] < 0.05 else 'p >= 0.05'
        f.write(f"    {feat}: H={res['H_statistic']:.2f}, p={res['p_value']:.2e} ({sig})\n")
    f.write("\n")

    f.write("3. PERFIL DOS SEGMENTOS\n")
    f.write("-" * 70 + "\n")
    for idx, row in cluster_summary.iterrows():
        name = segment_names[idx]
        f.write(f"\n  [{name}] (Cluster {idx})\n")
        f.write(f"    Clientes: {row['N_Customers']:,.0f} ({row['Pct_Customers']}%)\n")
        f.write(f"    Receita: ${row['Monetary_Total']:,.0f} ({row['Pct_Revenue']}%)\n")
        f.write(f"    Recency: {row['Recency_Mean']:.0f} dias\n")
        f.write(f"    Frequencia: {row['Frequency_Mean']:.1f} transacoes\n")
        f.write(f"    Valor medio: ${row['Monetary_Mean']:,.0f}\n")
        f.write(f"    Ticket medio: ${row['AvgOrderValue_Mean']:,.0f}\n")
        f.write(f"    Diversidade: {row['ProductDiversity_Mean']:.0f} produtos\n")

    f.write("\n\n4. INSIGHTS DE NEGOCIO\n")
    f.write("-" * 70 + "\n")

    if 'Champions' in segment_names.values():
        champ_idx = [k for k, v in segment_names.items() if v == 'Champions'][0]
        champ = cluster_summary.loc[champ_idx]
        f.write(f"\n  RETENCAO - Champions:\n")
        f.write(f"    {champ['N_Customers']:,.0f} clientes responsaveis por {champ['Pct_Revenue']}% da receita\n")
        f.write(f"    Acao: Programa VIP, beneficios exclusivos, early access\n")

    if 'Em Risco' in segment_names.values():
        risk_idx = [k for k, v in segment_names.items() if v == 'Em Risco'][0]
        risk = cluster_summary.loc[risk_idx]
        f.write(f"\n  PREVENCAO DE CHURN - Em Risco:\n")
        f.write(f"    {risk['N_Customers']:,.0f} clientes com alta frequencia historica mas inativos\n")
        f.write(f"    Acao: Campanha de reengajamento, descontos personalizados, pesquisa de satisfacao\n")

    if 'Novos Clientes' in segment_names.values():
        new_idx = [k for k, v in segment_names.items() if v == 'Novos Clientes'][0]
        new = cluster_summary.loc[new_idx]
        f.write(f"\n  CRESCIMENTO - Novos Clientes:\n")
        f.write(f"    {new['N_Customers']:,.0f} clientes com compras recentes mas baixa frequencia\n")
        f.write(f"    Acao: Programa de onboarding, cross-sell, incentivos para segunda compra\n")

    f.write("\n\n5. PROXIMOS PASSOS\n")
    f.write("-" * 70 + "\n")
    f.write("  1. Validar segmentos com equipe de marketing\n")
    f.write("  2. Implementar campanhas segmento-especificas\n")
    f.write("  3. Monitorar drift nos segmentos mensalmente\n")
    f.write("  4. A/B testar estrategias por segmento\n")
    f.write("  5. Calcular Customer Lifetime Value (CLV) por segmento\n")

    f.write("\n\n6. ARTEFATOS GERADOS\n")
    f.write("-" * 70 + "\n")
    f.write("  Modelos:\n")
    f.write("    - outputs/models/kmeans_model.joblib\n")
    f.write("    - outputs/models/scaler.joblib\n")
    f.write("    - outputs/models/model_metadata.json\n")
    f.write("  Dados:\n")
    f.write("    - data/processed/cleaned_data.csv\n")
    f.write("    - data/features/customer_rfm_features.csv\n")
    f.write("    - outputs/reports/customer_segments.csv\n")
    f.write("  Visualizacoes:\n")
    f.write("    - outputs/figures/ (todos os graficos PNG e HTML)\n")

    f.write("\n\n" + "=" * 70 + "\n")

print(f"  Relatorio salvo: {report_path}")

# %%
segment_summary_path = os.path.join(REPORTS_PATH, 'segment_summary.csv')
export_summary = cluster_summary.copy()
export_summary['Segment_Name'] = export_summary.index.map(segment_names)
export_summary.to_csv(segment_summary_path)
print(f"  Resumo salvos: {segment_summary_path}")

# %%
# --- Resumo Final ---
print("\n" + "=" * 60)
print("ANALISE COMPLETA!")
print("=" * 60)
print(f"\nResultados:")
print(f"  Clientes segmentados: {len(rfm_df):,}")
print(f"  Segmentos: {best_n_clusters}")
print(f"  Silhouette Score: {final_sil:.4f}")
print(f"  Estabilidade (ARI): {np.mean(ari_scores):.4f}")
print(f"\nSegmentos identificados:")
for idx, name in segment_names.items():
    n = (final_labels == idx).sum()
    pct = n / len(final_labels) * 100
    print(f"  {idx}. {name}: {n:,} ({pct:.1f}%)")

print(f"\nArquivos gerados:")
print(f"  Figuras: {FIGURES_PATH}")
print(f"  Modelos: {MODELS_PATH}")
print(f"  Relatorios: {REPORTS_PATH}")
print(f"\nPipeline concluido com sucesso!")
