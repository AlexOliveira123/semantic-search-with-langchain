#!/usr/bin/env python3
"""Interface de chat RAG interativa com busca semântica via pgVector."""

from langchain_core.prompts import PromptTemplate
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_postgres import PGVector

from langchain.chains import (
    create_history_aware_retriever,
    create_retrieval_chain,
)
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.chat_history import InMemoryChatMessageHistory

from config import (
    CHAT_MODEL,
    COLLECTION_NAME,
    DATABASE_URL,
    EMBEDDING_MODEL,
    OPENAI_API_KEY,
    SEARCH_TOP_K,
)

# ---------------------------------------------------------------------------
# Prompts
# ---------------------------------------------------------------------------
CONDENSE_QUESTION_PROMPT = PromptTemplate.from_template(
    """Dado o histórico da conversa e uma pergunta de acompanhamento, reformule a
pergunta de acompanhamento para que seja uma pergunta independente no idioma original.

Histórico da conversa:
{chat_history}

Pergunta de acompanhamento: {input}

Pergunta independente:"""
)

QA_PROMPT = PromptTemplate.from_template(
    """Você é um assistente que responde perguntas com base exclusivamente no CONTEXTO fornecido abaixo.

Regras obrigatórias:
- Responda somente com base no CONTEXTO.
- Se a informação não estiver explicitamente no CONTEXTO, responda:
  "Não tenho informações necessárias para responder sua pergunta."
- Nunca invente ou use conhecimento externo.
- Nunca produza opiniões ou interpretações além do que está escrito.

CONTEXTO:
{context}

Pergunta: {input}

Resposta:"""
)

# ---------------------------------------------------------------------------
# Vector Store
# ---------------------------------------------------------------------------
def get_vector_store() -> PGVector:
    embeddings = OpenAIEmbeddings(
        model=EMBEDDING_MODEL,
        api_key=OPENAI_API_KEY,
    )
    return PGVector(
        embeddings=embeddings,
        collection_name=COLLECTION_NAME,
        connection=DATABASE_URL,
    )

# ---------------------------------------------------------------------------
# Memory (novo padrão)
# ---------------------------------------------------------------------------
store = {}

def get_session_history(session_id: str):
    if session_id not in store:
        store[session_id] = InMemoryChatMessageHistory()
    return store[session_id]

# ---------------------------------------------------------------------------
# Chain
# ---------------------------------------------------------------------------
def build_chain():
    llm = ChatOpenAI(
        model=CHAT_MODEL,
        temperature=1,
        api_key=OPENAI_API_KEY,
    )

    vector_store = get_vector_store()

    retriever = vector_store.as_retriever(
        search_type="similarity",
        search_kwargs={"k": SEARCH_TOP_K},
    )

    # 🔹 Retriever com histórico
    history_aware_retriever = create_history_aware_retriever(
        llm,
        retriever,
        CONDENSE_QUESTION_PROMPT,
    )

    # 🔹 Chain de resposta
    question_answer_chain = create_stuff_documents_chain(
        llm,
        QA_PROMPT,
    )

    # 🔹 Retrieval chain final
    rag_chain = create_retrieval_chain(
        history_aware_retriever,
        question_answer_chain,
    )

    # 🔹 Memória nova
    chain_with_memory = RunnableWithMessageHistory(
        rag_chain,
        get_session_history,
        input_messages_key="input",
        history_messages_key="chat_history",
        output_messages_key="answer",
    )

    return chain_with_memory

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def format_sources(source_docs: list) -> str:
    seen: set[str] = set()
    lines: list[str] = []
    for doc in source_docs:
        source = doc.metadata.get("source", "unknown")
        page = doc.metadata.get("page", "?")
        key = f"{source}:{page}"
        if key not in seen:
            seen.add(key)
            lines.append(f"  • {source}  (página {page})")
    return "\n".join(lines) if lines else "  • (sem metadados de origem)"

# ---------------------------------------------------------------------------
# Chat loop
# ---------------------------------------------------------------------------
def run_chat() -> None:
    print("\n" + "=" * 60)
    print("  Chat de Busca Semântica")
    print("  Digite 'sair' ou 'exit' para encerrar.")
    print("=" * 60 + "\n")

    chain = build_chain()

    session_id = "cli-session"  # pode evoluir pra multi-user

    while True:
        try:
            question = input("Você: ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\n[chat] Até logo.")
            break

        if not question:
            continue
        if question.lower() in {"sair", "exit", "quit"}:
            print("[chat] Até logo.")
            break

        result = chain.invoke(
            {"input": question},
            config={"configurable": {"session_id": session_id}},
        )

        answer = result.get("answer", "")
        source_docs = result.get("context", [])  # 🔥 substitui source_documents

        print(f"\nAssistente: {answer}")
        if source_docs:
            print("\nFontes:")
            print(format_sources(source_docs))
        print()


if __name__ == "__main__":
    run_chat()