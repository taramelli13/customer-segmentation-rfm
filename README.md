# Segmentacao de clientes - UCI Online Retail Dataset

Segmentacao de clientes com RFM (Recency, Frequency, Monetary) e clustering no dataset UCI Online Retail.

## Objetivos de negocio

1. Identificar clientes de alto valor para retencao
2. Detectar clientes em risco de churn
3. Encontrar novos clientes com potencial de conversao

## Estrutura do projeto

```
Segmentacao de clientes/
├── config/
│   └── config.yaml              # Configuracoes do projeto
│
├── data/
│   ├── raw/
│   ├── processed/
│   │   └── cleaned_data.csv     # Dados limpos (392K transacoes)
│   └── features/
│       └── customer_rfm_features.csv  # Features RFM por cliente
│
├── notebooks/
│   ├── customer_segmentation_analysis.ipynb  # Notebook principal
│   ├── customer_segmentation_analysis.py     # Script equivalente
│   └── data/
│       └── raw/
│           └── online_retail_raw.csv         # Dataset original (541K transacoes)
│
├── outputs/
│   ├── figures/                 # 22 graficos PNG + 1 HTML interativo
│   ├── models/
│   │   ├── kmeans_model.joblib  # Modelo KMeans treinado
│   │   ├── scaler.joblib        # StandardScaler ajustado
│   │   └── model_metadata.json  # Metadados e hiperparametros
│   └── reports/
│       ├── customer_segments.csv    # Segmento atribuido por cliente
│       ├── segment_summary.csv      # Estatisticas por segmento
│       └── final_report.txt         # Relatorio executivo
│
├── src/                         # Modulos auxiliares
│   ├── data/
│   │   ├── data_loader.py       # Carga de dados
│   │   └── data_cleaner.py      # Pre-processamento
│   ├── features/
│   │   └── rfm_calculator.py    # Features RFM
│   ├── models/
│   │   └── clustering.py        # Algoritmos de clustering
│   ├── evaluation/
│   │   └── metrics.py           # Metricas de avaliacao
│   └── visualization/
│       ├── eda_plots.py         # Visualizacoes EDA
│       └── cluster_plots.py     # Visualizacoes de clusters
│
├── tests/                       # Testes unitarios (62 testes)
├── requirements.txt
└── README.md
```

## Dataset

Fonte: [UCI Online Retail Dataset](https://archive.ics.uci.edu/ml/datasets/Online+Retail) (Kaggle: `carrie1/ecommerce-data`)

~541.000 transacoes | 4.338 clientes apos limpeza | Dezembro 2010 a Dezembro 2011 | Varejo online do Reino Unido

Colunas: InvoiceNo, StockCode, Description, Quantity, InvoiceDate, UnitPrice, CustomerID, Country.

## Setup

### Pre-requisitos

- Python 3.8+
- pip

### Instalacao

1. Navegar ate o projeto:
   ```bash
   cd "path/to/Segmentacao de clientes"
   ```

2. Criar ambiente virtual:
   ```bash
   python -m venv .venv

   # Windows
   .venv\Scripts\activate

   # Mac/Linux
   source .venv/bin/activate
   ```

3. Instalar dependencias:
   ```bash
   pip install -r requirements.txt
   ```

4. Verificar instalacao:
   ```bash
   python -c "import pandas; import sklearn; print('OK')"
   ```

## Como usar

### Notebook

O notebook `customer_segmentation_analysis.ipynb` tem a analise completa:

```bash
jupyter notebook notebooks/customer_segmentation_analysis.ipynb
```

Execute as celulas na ordem. Segue o CRISP-DM:
1. Definicao do problema
2. Analise exploratoria (EDA)
3. Preparacao dos dados
4. Modelagem (KMeans, Hierarchical, DBSCAN)
5. Otimizacao com Optuna
6. Avaliacao e metricas
7. Interpretacao (SHAP)
8. Relatorio final

### Script

```bash
python notebooks/customer_segmentation_analysis.py
```

## Metodologia

### 1. Pre-processamento

- Remocao de registros sem CustomerID (~25% dos dados)
- Exclusao de transacoes canceladas (InvoiceNo com prefixo 'C')
- Capping de outliers por percentil (1%-99%)
- Criacao de TotalPrice = Quantity x UnitPrice
- Resultado: 541.909 -> 392.673 transacoes (72.5% retidos)

### 2. Features RFM

| Feature | Descricao | Calculo |
|---------|-----------|---------|
| Recency | Dias desde a ultima compra | Data referencia - ultima compra |
| Frequency | Numero de transacoes | Contagem de faturas unicas |
| Monetary | Receita total | Soma de TotalPrice |

Tambem calculadas: Tenure, ProductDiversity, AvgOrderValue.

### 3. Transformacao

- `np.log1p` para reduzir assimetria das distribuicoes RFM
- StandardScaler para normalizar

### 4. Selecao de K

- K=2 a K=10 com Elbow, Silhouette, Davies-Bouldin e Calinski-Harabasz
- K estatisticamente otimo: K=2
- K escolhido: 4 (restricao de negocio: 4-6 segmentos para acionabilidade)

### 5. Comparacao de algoritmos

| Algoritmo | Silhouette | Davies-Bouldin | Calinski-Harabasz |
|-----------|------------|----------------|-------------------|
| KMeans | 0.337 | 1.01 | 3389.5 |
| Hierarchical (Ward) | 0.321 | 1.08 | 3175.2 |
| DBSCAN | - | - | - |

KMeans selecionado, otimizado com Optuna (50 trials).

### 6. Interpretacao com SHAP

- Random Forest como classificador proxy (accuracy ~95%)
- SHAP TreeExplainer para importancia das features
- Ranking: Recency > Monetary > Frequency

## Resultados

### Segmentos (K=4)

| Segmento | Clientes | % Receita | Recency | Frequencia | Valor medio |
|----------|----------|-----------|---------|------------|-------------|
| Champions | 728 (16.8%) | 63.6% | 12 dias | 13.6x | $6.679 |
| Novos Clientes | 848 (19.5%) | 5.6% | 19 dias | 2.1x | $506 |
| Em Risco | 1.193 (27.5%) | 24.2% | 70 dias | 4.0x | $1.547 |
| Hibernando | 1.569 (36.2%) | 6.6% | 186 dias | 1.3x | $321 |

16.8% dos clientes (Champions) geram 63.6% da receita total.

### Metricas

| Metrica | Valor | Status |
|---------|-------|--------|
| Silhouette Score | 0.337 | Moderado (tipico para e-commerce) |
| Davies-Bouldin Index | 1.01 | Proximo do alvo (<1.0) |
| Calinski-Harabasz | 3389.5 | Alto |
| Estabilidade (ARI) | 0.995 | Excelente |

### Recomendacoes

- Champions: Programa VIP, beneficios exclusivos, early access
- Em Risco: Campanha de reengajamento, descontos personalizados, pesquisa de satisfacao
- Novos Clientes: Onboarding, cross-sell, incentivos para segunda compra
- Hibernando: Campanha de reativacao com ofertas agressivas ou aceitar churn natural

## Visualizacoes

### EDA

- `eda_missing_values.png` - Dados ausentes
- `eda_distributions_raw.png` - Distribuicoes
- `eda_top_countries.png` - Top paises
- `eda_time_series.png` - Serie temporal
- `eda_temporal_patterns.png` - Padroes temporais

### Features RFM

- `rfm_distributions.png` - Distribuicoes R, F, M
- `rfm_correlation_matrix.png` - Correlacoes
- `rfm_boxplots.png` - Boxplots
- `rfm_scatter_pairwise.png` - Scatter pareados
- `rfm_scatter_3d.html` - Scatter 3D (Plotly)

### Modelagem

- `optimal_k_analysis.png` - Analise do K otimo
- `dendrogram.png` - Dendrograma
- `silhouette_analysis.png` - Silhueta por cluster
- `clusters_pca_2d.png` - PCA 2D

### Resultados

- `segment_profiles.png` - Perfis dos segmentos
- `segment_sizes.png` - Tamanho dos segmentos
- `segment_heatmap.png` - Heatmap
- `revenue_by_segment.png` - Receita por segmento
- `shap_bar_importance.png` - Importancia SHAP
- `shap_summary.png` - SHAP summary
- `shap_dependence_*.png` - Dependence plots

## Tecnologias

pandas, numpy, scikit-learn, scipy, optuna, shap, matplotlib, seaborn, plotly, joblib, Jupyter

## Proximos passos

1. Validar segmentos com equipe de marketing
2. Implementar campanhas por segmento
3. Monitorar drift nos segmentos mensalmente
4. A/B testar estrategias por segmento
5. Calcular CLV por segmento

## Referencias

- [UCI Online Retail Dataset](https://archive.ics.uci.edu/ml/datasets/Online+Retail)
- [Kaggle: carrie1/ecommerce-data](https://www.kaggle.com/datasets/carrie1/ecommerce-data)
- [RFM Analysis (Wikipedia)](https://en.wikipedia.org/wiki/RFM_(market_research))

## Licenca

Projeto para fins educacionais e de pesquisa.
