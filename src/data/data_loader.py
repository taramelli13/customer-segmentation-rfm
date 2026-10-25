import pandas as pd
import logging

logger = logging.getLogger(__name__)

DTYPES = {'InvoiceNo': str, 'StockCode': str}


def load_from_csv(file_path, encodings=('utf-8', 'ISO-8859-1')):
    # O CSV do Kaggle (carrie1/ecommerce-data) vem em ISO-8859-1
    for encoding in encodings:
        try:
            df = pd.read_csv(file_path, dtype=DTYPES, encoding=encoding)
            break
        except UnicodeDecodeError:
            logger.info(f"Falha ao ler com encoding {encoding}, tentando o proximo...")
    else:
        raise ValueError(f"Nao foi possivel ler {file_path} com os encodings {encodings}")

    df['InvoiceDate'] = pd.to_datetime(df['InvoiceDate'])

    logger.info(f"Dados carregados de {file_path}: {df.shape}")

    return df


def load_from_excel(file_path):
    df = pd.read_excel(file_path, dtype=DTYPES)
    df['InvoiceDate'] = pd.to_datetime(df['InvoiceDate'])

    logger.info(f"Dados carregados de {file_path}: {df.shape}")

    return df
