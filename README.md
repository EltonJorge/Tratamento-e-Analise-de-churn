# 📊 Telecom X - Análise de Evasão de Clientes

[![Status](https://img.shields.io/badge/Status-Concluído-brightgreen)]() 
[![Python](https://img.shields.io/badge/Python-3.9+-blue?logo=python&logoColor=white)](https://www.python.org/) 
[![LinkedIn](https://img.shields.io/badge/LinkedIn-Conecte--se-blue?logo=linkedin)](https://www.linkedin.com/in/eltonjorgee)

---

## 🧠 Sobre o Projeto

A Telecom X enfrenta um alto índice de evasão de clientes (Churn). Este projeto foi desenvolvido a pedido da equipe de Ciência de Dados, com o objetivo de praticar o ETL (Extract, Transform, Load - Extração, Transformação e Carga) e realizar as primeiras análises para descobrir insights valiosos sobre o perfil de clientes da empresa, buscando entender o motivo da alta taxa de churn.

> 🗂️ Este trabalho é essencial para apoiar a próxima etapa do time de Data Science, que é a criação de modelos preditivos e soluções estratégicas. Uma boa preparação de dados é o primeiro passo para se realizar modelos preditivos de excelência; dados sujos ou mal trabalhados podem levar o modelo a trazer resultados equivocados.
---

## 🗂️ Sumário

1.  📖 [Configuração Inicial e Bibliotecas](#configuração-inicial-e-bibliotecas)
2.  📌 [Extração de Dados](#extracao-de-dados)
    * ⬇️ Baixando dados da API JSON
    * 🧹 Normalizando dados JSON
3.  🧹 [Tratamento de Dados](#tratamento-de-dados)
    * 🔄 Removendo Dados Duplicados
    * 🗑️ Removendo dados faltantes (com acordo das orientações da equipe)
    * ♻️ Tratando Dados Inconsistentes 
    * 🧱 Criação de Novas Colunas 
    * 🔍 Análise de Variáveis Categóricas
    * 📉 Análise de Variáveis Numéricas
4.  🛠️ [Padronização e Transformação de Dados](#padronização-e-transformação-de-dados) 
5.  📈 [Análise Exploratória de Dados (EDA) e Insights](#análise-exploratória-de-dados-eda-e-insights)
    * 🎯 Distribuição do Churn (Evasão)
    * 🕵️ Análise de Churn por Variáveis Categóricas
    * 📊 Visualização de Dados (Data Visualization)
    * 🔥 Heatmap Churn - Contrato x Pagamento
6.  ⚙️ [Preparação dos Dados para Modelagem Futura](#preparação-dos-dados-para-modelagem-futura)
    * 🔢 Transformando colunas categóricas em numéricas (Codificação)
    * ✨ Criando novas colunas 
7.  📑 [Principais Insights](#principais-insights)
    * 💡 [Recomendações](#recomendações)
    * 🚀 Sugestões
---

## Configuração Inicial e Bibliotecas

```python
#importando a biblioteca
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from matplotlib.ticker import MultipleLocator

```
---
## Extração de Dados

```python
url = "https://raw.githubusercontent.com/ingridcristh/challenge2-data-science/refs/heads/main/TelecomX_Data.json"

telecomx = pd.read_json(url)
print(telecomx)

telecomx.info()

#normalizando e desaninhando os dados da colunas
telecomx = pd.json_normalize(telecomx.to_dict(orient = "records"))

telecomx.head(10)
```
---

## Tratamento de Dados

```python
#obtendo informaçoes do DataFrame
telecomx.info()

#verifique se há valores únicos em cada coluna
for col in telecomx.columns:
  print(f"valores únicos na coluna '{col}': {telecomx[col].nunique()}")
  if telecomx[col].nunique() < 50: # imprimindo apenas se houver menos de 50 valores unicos
    print(telecomx[col].unique())
    print("-" * 50)

#Verificando se há valores nulos
telecomx.isnull().sum()

#Verificando se a dados duplicados
telecomx.duplicated().sum()

#Verificando as colunas com espaço em brancos
espacos_em_branco = telecomx.apply(lambda x: x.astype(str).str.strip() == '').sum()
print(espacos_em_branco[espacos_em_branco > 0])

#Transformando a coluna account.Charges.Total em coluna em tipo float
telecomx["account.Charges.Total"] = pd.to_numeric(telecomx["account.Charges.Total"], errors = "coerce")

#Tratando a coluna churn, substituindo os espaços em brancos por nan
telecomx["Churn"] = telecomx["Churn"].replace(r'^\s*$',np.nan, regex= True)

#criando coluna e calculando os gastos diario
telecomx["Gasto_diario"] = ((telecomx["Total_pago"] / telecomx["Meses_contrato"]) /30).round(2)

#inserindo a coluna na posição especifica
telecomx.insert(19, "Gasto_diario", telecomx.pop("Gasto_diario"))

#tratando dos valor NAN, removendo da base de dados.
telecomx = telecomx.dropna()

telecomx.isnull().sum()
```
---

## Padronização e Transformação de Dados

```python
telecomx = telecomx.rename(columns ={"customerID": "ID_cliente", "customer.gender" : "Genero",
                          "customer.SeniorCitizen": "Idoso", "customer.Partner": "Possui_parceiro",
                          "customer.Dependents": "Possui_dependentes", "customer.tenure": "Meses_contrato",
                          "phone.PhoneService": "Servico_telefonico", "phone.MultipleLines": "Multiplas_linhas",
                          "internet.InternetService": "Internet", "internet.OnlineSecurity": "Seguranca_online",
                          "internet.OnlineBackup": "backup_online", "internet.DeviceProtection": "Protecao_dispositivo",
                          "internet.TechSupport": "suporte_tecnico_adcional", "internet.StreamingTV": "Streaming_TV",
                          "internet.StreamingMovies": "Streaming_filmes", "account.Contract": "Contrato",
                          "account.PaperlessBilling": "fatura_digital", "account.PaymentMethod": "Forma_pagamento",
                          "account.Charges.Monthly": "Valor_mensal", "account.Charges.Total": "Total_pago"})


```

---

## 📈#análise-exploratória-de-dados-eda-e-insights

### 🎯 Distribuição de Churn
![Distribuição de Churn](img/Distribuição_de_Churn.png)

### 📊 Churn por Tempo de Contrato
![Distribuição do Churn por tempo de contrato](img/Churn_por_tempo_de_contrato.png)

### 📄 Tipos de Contrato
![Distribuição dos Tipos de Contrato](img/Distribuição_dos_Tipos_de_Contrato.png)

### 👥 Perfil dos Clientes
![Distribuição de Clientes](img/distribuicao_clientes.png)

### 📌 Perfil de Serviço
![Perfil de Serviço](img/radar_perfis_clientes.png)


### 💸 Valor Mensal vs Tempo de Contrato com Churn
![Relação Tempo vs Mensalidade](img/Relação_do_tempo_de_contrato_e_valores_pago_mensalmente_para_evasão.png)

### 🔥 Heatmap Churn - Contrato x Pagamento
![Heatmap Churn](img/heatmap_churn_contrato_pagamento.png)

---

## Preparação dos Dados para Modelagem Futura

```python
#Salvando o arquivo pre tratado em formato csv
telecomx.to_csv("telecomx.csv", index = False)

#copiando o dados para outro DataFrame, para manter os dados segurados
dados_telecomx = telecomx.copy()
dados_telecomx.head(5)

colunas_binarias = [
    "Churn",
    "Possui_parceiro",
    "Possui_dependentes",
    "Servico_telefonico",
    "fatura_digital"
]

mapeamento_binarios = {"Yes": 1, "No": 0}

for col in colunas_binarias:
  print(f"{col}:{dados_telecomx[col].unique()}")

for col in colunas_binarias:
  dados_telecomx[col] = dados_telecomx[col].map(mapeamento_binarios)

dados_telecomx.head()

dados_telecomx["Assina_internet"] = dados_telecomx["Internet"].map(lambda x: 0 if x == "No" else 1)

dados_telecomx.insert(10, "Assina_internet", dados_telecomx.pop("Assina_internet"))

# Colunas de serviços
servicos = [
    "Servico_telefonico", "Streaming_TV", "Streaming_filmes", "Assina_internet",
    "backup_online", "Seguranca_online", "Protecao_dispositivo", "suporte_tecnico_adcional"
]

# Mapeamento para transformar Yes/No/etc em 1/0
mapeamento_binario = {"Yes": 1, "No": 0, "No internet service": 0, "No phone service": 0}

# Só aplica o map se a coluna ainda for do tipo "object" (texto)
for col in servicos:
    if dados_telecomx[col].dtype == "object":
        dados_telecomx[col] = dados_telecomx[col].map(mapeamento_binario)

# Agora pode somar com segurança
dados_telecomx["Qtd_servicos"] = dados_telecomx[servicos].sum(axis=1).astype(int)

dados_telecomx["Todos_servicos"] = (dados_telecomx["Qtd_servicos"] == len(servicos)).astype(int)

```
---

🚀 **O que foi praticado**

✅ Importação e manipulação de dados de uma API (formato JSON)

✅ Normalização dos dados que estavam aninhados

✅ Aplicação de processos ETL (Extração, Transformação e Carga)

✅ Tratamento de dados faltantes, inconsistentes e categóricos

✅ Padronização de dados

✅ Renomeando colunas

✅ Dataviz com Seaborn, Matplotlib e Plotly (para gráficos interativos)

✅ Criação de colunas derivadas, como Contas Diárias, para enriquecer a análise

✅ Realização de Análise Exploratória de Dados (EDA) para entender padrões de evasão

✅ Estruturação dos dados para futura modelagem preditiva

---

## 📈 Principais Insights

* 💡 Não há variações significativas de taxa de evasão por **gênero** ou por **dependentes**.
* 📌 **Contratos curtos (Month-to-month)** possuem os maiores índices de churn.
* 💳 **Métodos de pagamento automáticos** (cartão ou transferência) estão associados a menor evasão, enquanto **Electronic Check** apresenta alto risco de cancelamento.
* ⚠️ Clientes que optaram pelo método de pagamento **Electronic Check** e contratos mensais têm uma taxa de 53,73% de churn.
* 🌐 Clientes que utilizam **Fiber Optic** têm maior tendência ao cancelamento em comparação com usuários de **DSL**, indicando possíveis problemas na entrega desse serviço.
* 📞 O **serviço de telefonia** é o mais popular, sendo utilizado por 90,33% dos nossos clientes.
* 🔐 Clientes com contratos de **1 ou 2 anos** apresentam **baixa taxa de churn**, mostrando que a fidelização é altamente eficiente.
* ✉️ Clientes que optaram pela **fatura digital** têm uma taxa de churn de 33,58%, quase o dobro da **fatura fisica**.
* ⬇️ Clientes que utilizam mais serviços (a partir de 2 serviços) têm uma taxa de churn decrescente.
* 💰 Os perfis mais lucrativos combinam contratos de longa duração e pacotes completos (Fibra + Streaming + Telefonia)
---

## 💡 Recomendações

- 💼 **Oferecer benefícios para contratos de longo prazo**, como descontos, bônus ou serviços adicionais.  
- 💳 **Incentivar o uso de pagamento automático**, oferecendo vantagens para clientes que migrarem para esse método.  
- 🛠️ **Melhorar a qualidade do serviço de fibra óptica**, revisando falhas técnicas e problemas de atendimento.  
- 🎯 **Focar em campanhas personalizadas para clientes com maior risco de churn**, especialmente aqueles com contrato mensal, pagamento manual e sem parceiro.  
- 📊 **Implementar monitoramento contínuo do churn**, integrando essa análise a modelos preditivos futuros.  
- 👨‍👩‍👧‍👦 **Desenvolver planos familiares**, considerando que cerca de 70% dos clientes não possuem filhos, e que a taxa de churn é menor entre os que têm parceiro.  
- 📦 **Criar pacotes de serviços integrados**, já que clientes com mais serviços contratados apresentam menor taxa de churn.

---

## ⚙️ Stack de Tecnologias e Ferramentas  

- [![Python](https://img.shields.io/badge/Python-3.9+-blue?logo=python&logoColor=white)](https://www.python.org/)  
- [![Pandas](https://img.shields.io/badge/Pandas-Data%20Analysis-%23150458?logo=pandas&logoColor=white)](https://pandas.pydata.org/)  
- [![NumPy](https://img.shields.io/badge/NumPy-Numerical-%23013243?logo=numpy&logoColor=white)](https://numpy.org/)  
- [![Matplotlib](https://img.shields.io/badge/Matplotlib-Graphs-orange?logo=matplotlib&logoColor=white)](https://matplotlib.org/)  
- [![Seaborn](https://img.shields.io/badge/Seaborn-Statistical-%2300CED1?logo=seaborn&logoColor=white)](https://seaborn.pydata.org/)  
- [![Plotly](https://img.shields.io/badge/Plotly-Interactive-%2300498B?logo=plotly&logoColor=white)](https://plotly.com/python/)  
- [![Jupyter](https://img.shields.io/badge/Jupyter-Notebook-F37626?logo=jupyter&logoColor=white)](https://jupyter.org/)  
- [![Colab](https://img.shields.io/badge/Google%20Colab-Notebook-%23F9AB00?logo=googlecolab&logoColor=white)](https://colab.research.google.com/)  
- [![Quarto](https://img.shields.io/badge/Quarto-Reporting-%2300599C?logo=quarto&logoColor=white)](https://quarto.org/)  
- [![VS Code](https://img.shields.io/badge/VS%20Code-Editor-%23007ACC?logo=visualstudiocode&logoColor=white)](https://code.visualstudio.com/)  
- [![GitHub](https://img.shields.io/badge/GitHub-Code-%23121011?logo=github&logoColor=white)](https://github.com/)  


---

## 🤝 Conecte-se comigo

[![LinkedIn](https://img.shields.io/badge/LinkedIn-Conecte--se-blue?logo=linkedin&logoColor=white)](https://www.linkedin.com/in/eltonjorgee)  
[![GitHub](https://img.shields.io/badge/GitHub-Visite--me-black?logo=github&logoColor=white)](https://github.com/EltonJorge)  
[![Email](https://img.shields.io/badge/Email-eltonoliveirajorge@hotmail.com-informational?logo=gmail&logoColor=white)](mailto:eltonoliveirajorge@hotmail.com)  
[![Telefone](https://img.shields.io/badge/Telefone-+55--21--96452--3066-lightgrey?logo=whatsapp&logoColor=white)](tel:+5521964523066)

---