# Busca Semântica com LangChain & pgVector
### Semantic Search with LangChain & pgVector

---

> 🇧🇷 [Português](#português) · 🇺🇸 [English](#english)

---

## Português

Pipeline de RAG (Retrieval-Augmented Generation) pronto para produção que ingere PDFs, armazena embeddings no PostgreSQL via pgVector e expõe uma interface de chat no terminal com OpenAI.

### Arquitetura

```
PDF
 └─▶ PyPDFLoader  ──▶  RecursiveCharacterTextSplitter  ──▶  OpenAI Embeddings
                                                                    │
                                                                    ▼
                                                             PGVector (Postgres)
                                                                    │
                                              ┌─────────────────────┘
                                              ▼
                                   Busca por Similaridade (top-10)
                                              │
                                              ▼
                                   ConversationalRetrievalChain
                                              │
                                              ▼
                                        ChatOpenAI (GPT-4o-mini)
                                              │
                                              ▼
                                       Resposta no Terminal
```

| Arquivo | Responsabilidade |
|---------|-----------------|
| `src/config.py` | Configurações carregadas do `.env` |
| `src/ingest.py` | PDF → chunks → embeddings → Postgres |
| `src/search.py` | Busca semântica avulsa via terminal |
| `src/chat.py` | Loop interativo de chat RAG |

### Pré-requisitos

- Docker & Docker Compose
- Python 3.11+
- Uma chave de API da OpenAI

### Início Rápido

#### 1. Configurar o ambiente

```bash
cp .env.example .env
# Edite o .env e defina o OPENAI_API_KEY
```

#### 2. Subir o Postgres com pgVector

```bash
docker compose up -d
# Aguarde ficar saudável:
docker compose ps
```

#### 3. Instalar as dependências Python

```bash
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

#### 4. Ingerir um PDF

```bash
python src/ingest.py caminho/para/documento.pdf
```

O script irá:
- Carregar todas as páginas do PDF
- Dividir o texto em chunks de 1.000 caracteres com sobreposição de 150
- Gerar embeddings com OpenAI (`text-embedding-3-small`)
- Persistir os vetores na coleção `documents` no Postgres

#### 5. Busca semântica (avulsa)

```bash
python src/search.py "Qual é o tema principal do documento?"
# Alterar a quantidade de resultados:
python src/search.py "modelo de precificação" --top-k 5
```

#### 6. Chat interativo

```bash
python src/chat.py
```

O assistente responde exclusivamente com base nos trechos recuperados do documento. Ele não inventa respostas — se o contexto não contiver a informação, ele declara isso explicitamente.

### Referência de Configuração

| Variável | Padrão | Descrição |
|----------|--------|-----------|
| `OPENAI_API_KEY` | — | **Obrigatório.** Chave secreta da OpenAI |
| `POSTGRES_USER` | `postgres` | Usuário do banco de dados |
| `POSTGRES_PASSWORD` | `postgres` | Senha do banco de dados |
| `POSTGRES_HOST` | `localhost` | Host do banco de dados |
| `POSTGRES_PORT` | `5432` | Porta do banco de dados |
| `POSTGRES_DB` | `semantic_search` | Nome do banco de dados |
| `DATABASE_URL` | derivado | Substitui a string de conexão completa |

### Parâmetros de Chunking

Configurados em `src/config.py`:

| Parâmetro | Valor |
|-----------|-------|
| `chunk_size` | 1.000 chars |
| `chunk_overlap` | 150 chars |
| `search_top_k` | 10 resultados |
| `embedding_model` | `text-embedding-3-small` |
| `chat_model` | `gpt-4o-mini` |

### Regras Anti-Alucinação do Prompt

A interface de chat impõe quatro regras obrigatórias ao LLM:

1. Responder somente com base no CONTEXTO fornecido.
2. Se o CONTEXTO não contiver a informação necessária, retornar exatamente: _"Não tenho informações necessárias para responder sua pergunta."_
3. Nunca inventar informações nem utilizar conhecimento externo ao CONTEXTO.
4. Nunca produzir opiniões ou interpretações além do que está escrito no CONTEXTO.

---

## English

Production-ready RAG (Retrieval-Augmented Generation) pipeline that ingests PDFs, stores embeddings in PostgreSQL via pgVector, and exposes a CLI chat interface powered by OpenAI.

### Architecture

```
PDF
 └─▶ PyPDFLoader  ──▶  RecursiveCharacterTextSplitter  ──▶  OpenAI Embeddings
                                                                    │
                                                                    ▼
                                                             PGVector (Postgres)
                                                                    │
                                              ┌─────────────────────┘
                                              ▼
                                   Similarity Search (top-10)
                                              │
                                              ▼
                                   ConversationalRetrievalChain
                                              │
                                              ▼
                                        ChatOpenAI (GPT-4o-mini)
                                              │
                                              ▼
                                         CLI Answer
```

| File | Responsibility |
|------|----------------|
| `src/config.py` | Settings loaded from `.env` |
| `src/ingest.py` | PDF → chunks → embeddings → Postgres |
| `src/search.py` | One-shot semantic search CLI |
| `src/chat.py` | Interactive RAG chat loop |

### Prerequisites

- Docker & Docker Compose
- Python 3.11+
- An OpenAI API key

### Quick Start

#### 1. Configure environment

```bash
cp .env.example .env
# Edit .env and set OPENAI_API_KEY
```

#### 2. Start Postgres with pgVector

```bash
docker compose up -d
# Wait until healthy:
docker compose ps
```

#### 3. Install Python dependencies

```bash
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

#### 4. Ingest a PDF

```bash
python src/ingest.py path/to/document.pdf
```

The script will:
- Load every page of the PDF
- Split text into 1 000-character chunks with 150-character overlap
- Generate OpenAI embeddings (`text-embedding-3-small`)
- Persist vectors in the `documents` collection inside Postgres

#### 5. Semantic search (one-shot)

```bash
python src/search.py "What is the main topic of the document?"
# Override result count:
python src/search.py "pricing model" --top-k 5
```

#### 6. Interactive chat

```bash
python src/chat.py
```

The assistant is strictly grounded in the retrieved document excerpts. It will
not invent answers — if the context lacks the information it says so explicitly.

### Configuration Reference

| Variable | Default | Description |
|----------|---------|-------------|
| `OPENAI_API_KEY` | — | **Required.** OpenAI secret key |
| `POSTGRES_USER` | `postgres` | Database user |
| `POSTGRES_PASSWORD` | `postgres` | Database password |
| `POSTGRES_HOST` | `localhost` | Database host |
| `POSTGRES_PORT` | `5432` | Database port |
| `POSTGRES_DB` | `semantic_search` | Database name |
| `DATABASE_URL` | derived | Override full connection string |

### Chunk Parameters

Configured in `src/config.py`:

| Parameter | Value |
|-----------|-------|
| `chunk_size` | 1 000 chars |
| `chunk_overlap` | 150 chars |
| `search_top_k` | 10 results |
| `embedding_model` | `text-embedding-3-small` |
| `chat_model` | `gpt-4o-mini` |

### Anti-Hallucination Prompt Rules

The chat interface enforces four mandatory rules on the LLM:

1. Answer only based on the provided CONTEXT.
2. If the CONTEXT does not contain the necessary information, return exactly: _"Não tenho informações necessárias para responder sua pergunta."_
3. Never invent information or use any knowledge external to the CONTEXT.
4. Never produce opinions or interpretations beyond what is explicitly written in the CONTEXT.
