# Dashboard Power BI — Segmentação de clientes RFM

Dashboard construído sobre os segmentos do modelo KMeans deste repositório. Responde três perguntas de negócio: quanto a loja vende e como isso evolui, quem são os clientes que sustentam a receita, e quanto dinheiro está em risco de ir embora.

## Dados

Os CSVs em `powerbi/data/` são gerados por `build_powerbi_data.py` a partir do [UCI Online Retail](https://archive.ics.uci.edu/dataset/352/online+retail):

```bash
python powerbi/build_powerbi_data.py "Online Retail.xlsx"
```

| Tabela | Linhas | Grão | Chave |
|---|---|---|---|
| fato_vendas | 392.688 | um item de uma nota fiscal | — |
| dim_cliente | 4.338 | um cliente, com métricas RFM e segmento | ClienteID |
| dim_produto | 3.664 | um produto | CodProduto |
| dim_calendario | 374 | um dia (01/12/2010 a 09/12/2011) | Data |
| dim_segmento | 4 | um segmento, com descrição e ação recomendada | Segmento |

## Modelo (estrela)

```
dim_calendario ─┐
dim_produto ────┼──< fato_vendas >── dim_cliente >── dim_segmento
```

| De (lado muitos) | Para (lado um) | Cardinalidade | Direção do filtro |
|---|---|---|---|
| fato_vendas[Data] | dim_calendario[Data] | * : 1 | Única |
| fato_vendas[CodProduto] | dim_produto[CodProduto] | * : 1 | Única |
| fato_vendas[ClienteID] | dim_cliente[ClienteID] | * : 1 | Única |
| dim_cliente[Segmento] | dim_segmento[Segmento] | * : 1 | Única |

`dim_calendario` é marcada como tabela de datas. Os eixos e filtros de segmento sempre usam `dim_segmento[Segmento]`, por isso `dim_cliente[Segmento]` fica oculta.

## Medidas DAX

Ficam todas numa tabela vazia chamada `_Medidas`.

```dax
Receita Total = SUM ( fato_vendas[Receita] )

Pedidos = DISTINCTCOUNT ( fato_vendas[NotaFiscal] )

Clientes Ativos = DISTINCTCOUNT ( fato_vendas[ClienteID] )

Ticket Medio = DIVIDE ( [Receita Total], [Pedidos] )

Receita Mes Anterior =
CALCULATE ( [Receita Total], DATEADD ( dim_calendario[Data], -1, MONTH ) )

Variacao MoM % =
DIVIDE ( [Receita Total] - [Receita Mes Anterior], [Receita Mes Anterior] )

% da Receita =
DIVIDE ( [Receita Total], CALCULATE ( [Receita Total], REMOVEFILTERS ( dim_segmento ) ) )

Clientes na Base = COUNTROWS ( dim_cliente )

% dos Clientes =
DIVIDE ( [Clientes na Base], CALCULATE ( [Clientes na Base], REMOVEFILTERS ( dim_segmento ) ) )

Recencia Media (dias) = AVERAGE ( dim_cliente[RecenciaDias] )

Receita em Risco =
CALCULATE ( SUM ( dim_cliente[ValorTotal] ), dim_segmento[Segmento] = "Em Risco" )

Clientes em Risco =
CALCULATE ( [Clientes na Base], dim_segmento[Segmento] = "Em Risco" )
```

## Páginas

1. **Visão geral:** cartões de Receita Total, Pedidos, Clientes Ativos e Ticket Medio; linha de receita por mês com Variacao MoM %; barras com os 10 países que mais compram; filtros de segmento e de país.
2. **Segmentos:** barras agrupadas comparando % dos Clientes com % da Receita por segmento; dispersão de Recência × Frequência por cliente, colorida por segmento; tabela com descrição e ação recomendada.
3. **Clientes em risco:** cartões de Receita em Risco e Clientes em Risco; tabela dos clientes do segmento "Em Risco" ordenada por valor gasto, com dias desde a última compra.

## Principais achados

- 16,6% dos clientes (Champions) geram 63,5% da receita.
- "Em Risco" reúne 1.195 clientes, que não compram há 70 dias em média e respondem por 24,2% da receita histórica. É o alvo prioritário de uma campanha de reativação.
- 36,8% da base está hibernando, com pouco valor (6,7% da receita).

---

## Passo a passo (para quem nunca usou o Power BI)

### 1. Instalar
Instale o **Power BI Desktop** pela Microsoft Store do Windows (é grátis). Na primeira vez que abrir, feche a janela de login.

### 2. Importar os CSVs (atenção ao separador decimal)
Os CSVs usam ponto como separador decimal (`15.30`). Num Windows em português, o Power BI pode ler `15.30` como **1530**. Para evitar isso:

1. **Página Inicial → Obter dados → Texto/CSV** → escolha `fato_vendas.csv` → **Transformar Dados** (não clique em Carregar).
2. No Power Query, clique com o botão direito no cabeçalho `Receita` → **Alterar Tipo → Usando Localidade…** → Tipo: *Número Decimal*, Localidade: *Inglês (Estados Unidos)*. Faça o mesmo com `PrecoUnitario` e, na `dim_cliente`, com `ValorTotal` e `TicketMedioItem`.
3. Confira se as colunas `Data`, `PrimeiraCompra` e `UltimaCompra` estão como **Data**. Se aparecerem como texto, use a mesma opção "Usando Localidade" com o tipo *Data*.
4. Repita para os outros quatro CSVs (**Nova Fonte → Texto/CSV**) e clique em **Fechar e Aplicar**.

### 3. Relacionamentos
Vá para a **Exibição de modelo** (terceiro ícone à esquerda). O Power BI costuma criar algumas ligações sozinho. Apague as que estiverem erradas e deixe exatamente as quatro da tabela "Modelo" acima, arrastando a coluna de uma tabela até a coluna correspondente da outra.

Depois clique em `dim_calendario` → **Ferramentas de tabela → Marcar como tabela de datas** → coluna `Data`.

### 4. Ordenações
- `dim_calendario[Mes]` → **Classificar por coluna** → `MesNum`
- `dim_calendario[DiaSemana]` → **Classificar por coluna** → `DiaSemanaNum`
- `dim_segmento[Segmento]` → **Classificar por coluna** → `Ordem`

### 5. Medidas
**Página Inicial → Inserir dados** → crie uma tabela vazia chamada `_Medidas`. Com ela selecionada, use **Nova medida** e cole uma fórmula da seção DAX de cada vez.

### 6. Visuais
Monte as três páginas descritas acima. Formate as medidas de valor como moeda (£, já que a loja é britânica) e as de % como porcentagem.

### 7. Publicar
**Página Inicial → Publicar**. Precisa de uma conta Microsoft de trabalho ou estudante. Se não tiver, exporte em **Arquivo → Exportar → PDF** e tire prints de cada página para o README.
