"""
Configuración del proyecto de Recuperación de Información (Sesión 6)
Embeddings + Similitud Coseno + RAG + Base de Datos Vectorial ChromaDB
"""
import os
from dotenv import load_dotenv # Manejar API keys/configuración

# Cargar variables de entorno
load_dotenv()


class Config:
    """Configuración centralizada del proyecto"""

    # ============================================================
    # Base de Datos Vectorial: ChromaDB (persistente, sin servidor)
    # ============================================================
    CHROMA_PERSIST_DIR = os.getenv("CHROMA_PERSIST_DIR", "./chroma_db")
    COLLECTION_NAME = os.getenv("COLLECTION_NAME", "politicas_empresa")

    # ============================================================
    # Embeddings - Provider
    # Opciones: "sentence-transformers" (gratis, local) o "openai"
    # ============================================================
    EMBEDDING_PROVIDER = os.getenv("EMBEDDING_PROVIDER", "openai")

    # OpenAI Embeddings (opcional, requiere API key)
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
    OPENAI_MODEL = os.getenv("OPENAI_MODEL", "text-embedding-3-small") # embedding model 
    OPENAI_DIMENSION = int(os.getenv("OPENAI_DIMENSION", "1536")) 
    # el vector embedding de OpenAI tiene 1536 dimensiones para el modelo "text-embedding-3-small"
    
    # Modelo de chat para la demo RAG (opcional)
    # OPENAI_CHAT_MODEL = os.getenv("OPENAI_CHAT_MODEL", "gpt-4o-mini")
    
    QWEN_API_KEY = os.getenv("QWEN_API_KEY", "")
    QWEN_CHAT_MODEL = os.getenv("QWEN_CHAT_MODEL", "Qwen/Qwen3.8-27B")
    
    # usado para generate respuestas en la demo RAG, no para embeddings

    # Sentence Transformers (modelo local gratuito, recomendado)
    # Alternativa más ligera usada en la sesión: "all-MiniLM-L6-v2"
    
    SENTENCE_TRANSFORMER_MODEL = os.getenv(
        "SENTENCE_TRANSFORMER_MODEL",
        "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
    )
    SENTENCE_TRANSFORMER_DIMENSION = int(os.getenv("SENTENCE_TRANSFORMER_DIMENSION", "384"))

    # ============================================================
    # Chunking (Taller Práctico - Parte 1)
    # 200-500 tokens por chunk, overlap 10-20%
    # ============================================================
    CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "500"))
    CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "50"))

    # ============================================================
    # Búsqueda semántica (Taller Práctico - Parte 3)
    # ============================================================
    DEFAULT_TOP_K = 10
    DEFAULT_SCORE_THRESHOLD = 0.0

    # ============================================================
    # RAG (Retrieval Augmented Generation)
    # ============================================================
    RAG_CONTEXT_CHUNKS = 5  # top-k chunks used in the augmented prompt for RAG

    @classmethod
    def print_config(cls):
        """Imprime la configuración actual"""
        print("\n" + "=" * 60)
        print("SESIÓN 6 - RECUPERACIÓN DE INFORMACIÓN (RAG)")
        print("=" * 60)
        print(f"Base de datos vectorial: ChromaDB (persistente)")
        print(f"Directorio de persistencia: {cls.CHROMA_PERSIST_DIR}")
        print(f"Colección: {cls.COLLECTION_NAME}")
        print(f"\nProveedor de embeddings: {cls.EMBEDDING_PROVIDER}")

        if cls.EMBEDDING_PROVIDER == "openai":
            print(f"Embedding Model OpenAI: {cls.OPENAI_MODEL}")
            print(f"Dimensión: {cls.OPENAI_DIMENSION}")
            api_key_preview = cls.OPENAI_API_KEY[:10] + "..." if cls.OPENAI_API_KEY else "NO CONFIGURADA"
            print(f"API Key: {api_key_preview}")
        else:
            print(f"Modelo: {cls.SENTENCE_TRANSFORMER_MODEL}")
            print(f"Dimensión: {cls.SENTENCE_TRANSFORMER_DIMENSION}")

        print(f"\nChunking: size={cls.CHUNK_SIZE}, overlap={cls.CHUNK_OVERLAP}")
        print(f"Búsqueda: top_k={cls.DEFAULT_TOP_K}, score_threshold={cls.DEFAULT_SCORE_THRESHOLD}")
        print(f"RAG: {cls.RAG_CONTEXT_CHUNKS} chunks de contexto en el prompt")
        print("=" * 60 + "\n")


if __name__ == "__main__":
    Config.print_config()
