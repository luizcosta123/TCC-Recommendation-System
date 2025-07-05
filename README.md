# TCC-Recommendation-System

Este projeto é um sistema de recomendação desenvolvido como parte de um Trabalho de Conclusão de Curso (TCC). O objetivo é agrupar estudantes com perfis acadêmicos semelhantes utilizando o algoritmo de clusterização K-Modes e, a partir desses grupos, fornecer insights e recomendações.

## Funcionalidades

- **Treinamento de Modelo:** Treina um modelo de K-Modes com base em respostas de formulários de estudantes.
- **Análise de Clusters:** Gera relatórios e visualizações para analisar as características de cada cluster.
- **Consulta de Alunos:** Permite buscar a qual cluster um determinado aluno pertence.

## Estrutura do Projeto

```
TCC-Recommendation-System/
├── app/
│   └── kmodes/
│       ├── data/                 # Datasets
│       ├── results/              # Resultados
│       ├── scripts/              # Scripts para análise dos clusters
│       └── training/             # Script para treinamento do modelo
│           └── models/           # Modelos treinados e metadados
├── .gitignore
├── README.md
└── requirements.txt
```

## Instalação e Configuração

Siga os passos abaixo para configurar o ambiente de desenvolvimento.

**1. Clone o Repositório**

Primeiro, clone este repositório para a sua máquina local.

**2. Crie e Ative um Ambiente Virtual**

É uma boa prática usar um ambiente virtual para isolar as dependências do projeto.

*   **No Windows:**
    ```bash
    python -m venv .venv
    .venv\Scripts\activate
    ```
*   **No macOS/Linux:**
    ```bash
    python3 -m venv .venv
    source .venv/bin/activate
    ```

**3. Instale as Dependências**
```bash
pip install -r requirements.txt
```

## Como Usar

Os comandos a seguir devem ser executados a partir do diretório raiz do projeto.

### 1. Treinando o Modelo

Para treinar o modelo K-Modes, execute o script de treinamento. O modelo, os clusters e os metadados serão salvos na pasta `app/kmodes/training/models/`.

```bash
python app/kmodes/training/kmodes_training.py
```

### 2. Executando os Scripts de Análise

Os scripts na pasta `app/kmodes/scripts/` podem ser executados para gerar análises sobre os clusters. Por exemplo:

```bash
python app/kmodes/scripts/all_analysis.py
```
