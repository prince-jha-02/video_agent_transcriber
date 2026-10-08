import os
import hashlib
from typing import Dict, Any

from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough, RunnableLambda

load_dotenv()

DEFAULT_MODEL = os.getenv("GEMINI_MODEL", "gemini-3.5-flash-lite")
FALLBACK_MODELS = [
    DEFAULT_MODEL,
    "gemini-3.5-flash-lite",
    "gemini-flash-latest",
    "gemini-3.6-flash",
]
CANDIDATE_MODELS = list(dict.fromkeys(FALLBACK_MODELS))

# In-memory Q&A cache: (transcript_hash, question) -> answer
_QA_CACHE: Dict[str, str] = {}


def get_llm(model_name: str = None) -> ChatGoogleGenerativeAI:
    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        raise RuntimeError("GOOGLE_API_KEY is not set in environment or .env")
    model = model_name or DEFAULT_MODEL
    return ChatGoogleGenerativeAI(
        model=model,
        google_api_key=api_key,
        timeout=30,
    )


def format_docs(docs):
    return "\n\n".join([doc.page_content for doc in docs])


def build_rag_chain(transcript: str):
    """
    Build a Q&A chain for the transcript.
    For transcripts <= 30,000 chars (all <=10 min videos), provides direct
    context answering for instant responses without embedding latency.
    For longer transcripts, uses Chroma vector store retrieval.
    """
    # If transcript fits comfortably in LLM context, use direct context chain (instant!)
    if len(transcript) <= 30000:
        print("[DIRECT] Using direct full-transcript context for Q&A (0s embedding overhead)")

        prompt = ChatPromptTemplate.from_messages([
            (
                "system",
                """You are an expert meeting and video assistant. Answer the user's question 
based ONLY on the meeting transcript context provided below.

If the answer is not found in the context, say: 
"I could not find this information in the meeting transcript."

Always be concise and precise. If quoting someone, mention it clearly.

Context from meeting transcript:
{context}""",
            ),
            ("human", "{question}"),
        ])

        class DirectContextChain:
            def __init__(self, full_transcript: str):
                self.transcript = full_transcript
                self.prompt = prompt

            def invoke(self, question: str) -> str:
                cache_key = hashlib.sha256(f"{self.transcript[:100]}::{question.strip().lower()}".encode("utf-8")).hexdigest()
                if cache_key in _QA_CACHE:
                    print("[CACHE] Returning Q&A from cache (0 Gemini API calls)")
                    return _QA_CACHE[cache_key]

                api_key = os.getenv("GOOGLE_API_KEY")
                last_err = None

                for m in CANDIDATE_MODELS:
                    try:
                        llm = ChatGoogleGenerativeAI(model=m, google_api_key=api_key, timeout=30)
                        chain = self.prompt | llm | StrOutputParser()
                        ans = chain.invoke({"context": self.transcript, "question": question})
                        if ans:
                            _QA_CACHE[cache_key] = ans.strip()
                            return ans.strip()
                    except Exception as e:
                        last_err = e
                        print(f"[WARNING] Q&A attempt with {m} failed: {e}")
                        continue

                return f"Unable to answer question due to model unavailability: {last_err}"


        return DirectContextChain(transcript)

    # For very large transcripts, use vector store
    from core.vectore_store import build_vector_store, get_retriever
    vector_store = build_vector_store(transcript)
    retriever = get_retriever(vector_store, k=4)
    llm = get_llm()

    prompt = ChatPromptTemplate.from_messages([
        (
            "system",
            """You are an expert meeting assistant. Answer the user's question 
based ONLY on the meeting transcript context provided below.

If the answer is not found in the context, say: 
"I could not find this information in the meeting transcript."

Always be concise and precise. If quoting someone, mention it clearly.

Context from meeting transcript:
{context}""",
        ),
        ("human", "{question}"),
    ])

    rag_chain = (
        {
            "context": retriever | RunnableLambda(format_docs),
            "question": RunnablePassthrough(),
        }
        | prompt
        | llm
        | StrOutputParser()
    )

    return rag_chain


def load_rag_chain():
    from core.vectore_store import load_vector_store, get_retriever
    vector_store = load_vector_store()
    retriever = get_retriever(vector_store)
    llm = get_llm()

    prompt = ChatPromptTemplate.from_messages([
        (
            "system",
            """You are an expert meeting assistant. Answer the user's question 
based ONLY on the meeting transcript context provided below.

If the answer is not found in the context, say: 
"I could not find this information in the meeting transcript."

Always be concise and precise. If quoting someone, mention it clearly.

Context from meeting transcript:
{context}""",
        ),
        ("human", "{question}"),
    ])

    rag_chain = (
        {
            "context": retriever | RunnableLambda(format_docs),
            "question": RunnablePassthrough(),
        }
        | prompt
        | llm
        | StrOutputParser()
    )

    return rag_chain


def ask_question(rag_chain, question: str) -> str:
    print(f"Question: {question}")
    answer = rag_chain.invoke(question)
    print(f"Answer: {answer}")
    return answer



