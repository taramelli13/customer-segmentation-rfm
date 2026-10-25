# Gera o modelo estrela (CSVs) usado no dashboard Power BI.
#
# Uso:
#   python powerbi/build_powerbi_data.py "caminho/para/Online Retail.xlsx"
#
# Saida em powerbi/data/: fato_vendas, dim_cliente, dim_produto,
# dim_calendario, dim_segmento.

import os
import sys

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, PROJECT_ROOT)

from src.features.rfm_calculator import calculate_rfm_features

SEED = 42
N_CLUSTERS = 4  # restricao de negocio do projeto (4-6 segmentos), ver README
OUT_PATH = os.path.join(PROJECT_ROOT, 'powerbi', 'data')

SEGMENT_INFO = {
    'Champions': (1, 'Compram com frequencia, recentemente e gastam muito',
                  'Programa de fidelidade, acesso antecipado, upsell'),
    'Clientes Leais': (2, 'Compram com frequencia e recentemente',
                       'Recompensar recorrencia, cross-sell'),
    'Grandes Gastadores': (3, 'Gastam muito, mas compram pouco',
                           'Aumentar frequencia com ofertas personalizadas'),
    'Novos Clientes': (4, 'Compraram recentemente, poucas vezes',
                       'Onboarding e incentivo a segunda compra'),
    'Em Risco': (5, 'Ja compraram bastante, mas estao sumindo',
                 'Campanha de reativacao antes de perder o cliente'),
    'Hibernando': (6, 'Pouca compra, ha muito tempo',
                   'Reativacao de baixo custo ou aceitar a perda'),
    'Outros': (7, 'Perfil misto', 'Analisar caso a caso'),
}


def clean_data(df):
    # Mesmas regras do config/config.yaml
    df = df.dropna(subset=['CustomerID']).copy()
    df['InvoiceNo'] = df['InvoiceNo'].astype(str)
    df = df[~df['InvoiceNo'].str.startswith('C')]
    df = df[df['UnitPrice'] >= 0.01]
    df = df[df['Quantity'] > 0]
    df = df.drop_duplicates()

    for col in ['Quantity', 'UnitPrice']:
        low, high = df[col].quantile([0.01, 0.99])
        df[col] = df[col].clip(low, high)

    df['TotalPrice'] = df['Quantity'] * df['UnitPrice']
    df['CustomerID'] = df['CustomerID'].astype(int)
    df['StockCode'] = df['StockCode'].astype(str).str.upper().str.strip()
    df['Description'] = df['Description'].astype(str).str.strip()
    return df.reset_index(drop=True)


def assign_segment_name(row, summary_df):
    # Mesma regra do notebook (FASE 6)
    r_low = row['Recency_Mean'] <= summary_df['Recency_Mean'].median()
    f_high = row['Frequency_Mean'] >= summary_df['Frequency_Mean'].median()
    m_high = row['Monetary_Mean'] >= summary_df['Monetary_Mean'].median()

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
    return 'Outros'


def segment_customers(rfm_df):
    X = np.log1p(rfm_df[['Recency', 'Frequency', 'Monetary']])
    X_scaled = StandardScaler().fit_transform(X)
    rfm_df['Cluster'] = KMeans(n_clusters=N_CLUSTERS, n_init=10,
                               random_state=SEED).fit_predict(X_scaled)

    summary = rfm_df.groupby('Cluster').agg(
        Recency_Mean=('Recency', 'mean'),
        Frequency_Mean=('Frequency', 'mean'),
        Monetary_Mean=('Monetary', 'mean'),
    )
    names = {idx: assign_segment_name(row, summary) for idx, row in summary.iterrows()}
    rfm_df['Segmento'] = rfm_df['Cluster'].map(names)
    return rfm_df


def build_calendar(start, end):
    dates = pd.date_range(start, end, freq='D')
    meses = ['Jan', 'Fev', 'Mar', 'Abr', 'Mai', 'Jun',
             'Jul', 'Ago', 'Set', 'Out', 'Nov', 'Dez']
    dias = ['Seg', 'Ter', 'Qua', 'Qui', 'Sex', 'Sab', 'Dom']
    return pd.DataFrame({
        'Data': dates.date,
        'Ano': dates.year,
        'Trimestre': 'T' + dates.quarter.astype(str),
        'MesNum': dates.month,
        'Mes': [meses[m - 1] for m in dates.month],
        'AnoMes': dates.strftime('%Y-%m'),
        'DiaSemanaNum': dates.day_of_week + 1,
        'DiaSemana': [dias[d] for d in dates.day_of_week],
    })


def main(xlsx_path):
    os.makedirs(OUT_PATH, exist_ok=True)

    print(f'Lendo {xlsx_path} ...')
    raw = pd.read_excel(xlsx_path, dtype={'InvoiceNo': str, 'StockCode': str})
    df = clean_data(raw)
    print(f'Transacoes: {len(raw):,} -> {len(df):,} | Clientes: {df["CustomerID"].nunique():,}')

    analysis_date = df['InvoiceDate'].max() + pd.Timedelta(days=1)
    rfm = segment_customers(calculate_rfm_features(df, analysis_date=analysis_date))

    fato = df[['InvoiceNo', 'InvoiceDate', 'StockCode', 'CustomerID',
               'Country', 'Quantity', 'UnitPrice', 'TotalPrice']].copy()
    fato.insert(2, 'Data', fato['InvoiceDate'].dt.date)
    fato.columns = ['NotaFiscal', 'DataHora', 'Data', 'CodProduto', 'ClienteID',
                    'Pais', 'Quantidade', 'PrecoUnitario', 'Receita']

    main_country = df.groupby('CustomerID')['Country'].agg(lambda s: s.mode().iat[0])
    cliente = rfm[['CustomerID', 'Segmento', 'Cluster', 'Recency', 'Frequency',
                   'Monetary', 'AvgOrderValue', 'R_Score', 'F_Score', 'M_Score',
                   'RFM_Score', 'FirstPurchaseDate', 'LastPurchaseDate']].copy()
    cliente.insert(1, 'Pais', cliente['CustomerID'].map(main_country))
    cliente['FirstPurchaseDate'] = cliente['FirstPurchaseDate'].dt.date
    cliente['LastPurchaseDate'] = cliente['LastPurchaseDate'].dt.date
    cliente.columns = ['ClienteID', 'Pais', 'Segmento', 'Cluster', 'RecenciaDias',
                       'Frequencia', 'ValorTotal', 'TicketMedioItem', 'R_Score',
                       'F_Score', 'M_Score', 'RFM_Score', 'PrimeiraCompra',
                       'UltimaCompra']

    produto = (df.groupby('StockCode')['Description']
                 .agg(lambda s: s.mode().iat[0]).reset_index())
    produto.columns = ['CodProduto', 'Descricao']

    segmento = pd.DataFrame(
        [(name, *info) for name, info in SEGMENT_INFO.items()
         if name in set(cliente['Segmento'])],
        columns=['Segmento', 'Ordem', 'Descricao', 'AcaoRecomendada'])

    calendario = build_calendar(df['InvoiceDate'].min().normalize(),
                                df['InvoiceDate'].max().normalize())

    tables = {'fato_vendas': fato, 'dim_cliente': cliente, 'dim_produto': produto,
              'dim_calendario': calendario, 'dim_segmento': segmento}
    for name, table in tables.items():
        table.to_csv(os.path.join(OUT_PATH, f'{name}.csv'), index=False, encoding='utf-8',
                     float_format='%.2f')
        print(f'  {name}.csv: {len(table):,} linhas')

    resumo = cliente.groupby('Segmento').agg(
        Clientes=('ClienteID', 'count'), Receita=('ValorTotal', 'sum'),
        Recencia=('RecenciaDias', 'mean'), Frequencia=('Frequencia', 'mean'),
        ValorMedio=('ValorTotal', 'mean'))
    resumo['PctClientes'] = (resumo['Clientes'] / resumo['Clientes'].sum() * 100).round(1)
    resumo['PctReceita'] = (resumo['Receita'] / resumo['Receita'].sum() * 100).round(1)
    print(resumo.round(1).to_string())


if __name__ == '__main__':
    if len(sys.argv) != 2:
        sys.exit('Uso: python powerbi/build_powerbi_data.py "Online Retail.xlsx"')
    main(sys.argv[1])
