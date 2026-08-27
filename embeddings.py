"""
Sistema de Embeddings y Similitud (Sesión 6 - Taller Práctico Parte 2)

Implementación desde cero:
- Embeddings con Sentence Transformers (local, gratuito) u OpenAI (API)
- Similitud coseno con la fórmula del material:
    cos(θ) = (A · B) / (||A|| × ||B||)
- Chunking de documentos (Taller Práctico - Parte 1)
"""
from typing import List
import numpy as np
from config import Config


def cosine_similarity(vec1: List[float], vec2: List[float]) -> float:
    """
    Calcula la similitud coseno entre dos vectores usando la fórmula:

        cos(θ) = (A · B) / (||A|| × ||B||)

    Mide cuán similares son dos textos independientemente de su magnitud.
    Rango: -1 (opuestos) a 1 (idénticos). En NLP usualmente 0 a 1.

    Args:
        vec1: Primer vector (embedding)
        vec2: Segundo vector (embedding)

    Returns:
        Score de similitud coseno
    """
    a = np.array(vec1, dtype=np.float32)
    b = np.array(vec2, dtype=np.float32)

    dot_product = np.dot(a, b)          # A · B
    norm_a = np.linalg.norm(a)          # ||A||
    norm_b = np.linalg.norm(b)          # ||B||

    if norm_a == 0 or norm_b == 0:
        return 0.0

    return float(dot_product / (norm_a * norm_b))


def cosine_similarity_matrix(vectors: List[List[float]]) -> np.ndarray:
    """
    Matriz de similitud coseno entre todos los pares de vectores.
    Útil para visualizar qué textos son más parecidos entre sí.
    """
    matrix = np.array(vectors, dtype=np.float32)
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    normalized = matrix / np.maximum(norms, 1e-12)
    return normalized @ normalized.T


class SentenceTransformerEmbeddings:
    """Embeddings locales con Sentence Transformers (gratis, sin API key)"""

    def __init__(self, model_name: str = None):
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError:
            raise ImportError(
                "sentence-transformers no está instalado. "
                "Instálalo con: pip install sentence-transformers"
            )

        self.model_name = model_name or Config.SENTENCE_TRANSFORMER_MODEL
        self.dimension = Config.SENTENCE_TRANSFORMER_DIMENSION

        print(f"Cargando modelo de embeddings: {self.model_name}")
        self.model = SentenceTransformer(self.model_name)

        # Verificar dimensión real del modelo
        test_embedding = self.model.encode(["test"], show_progress_bar=False)
        actual_dim = len(test_embedding[0])

        if actual_dim != self.dimension:
            print(f"⚠️  Ajustando dimensión de {self.dimension} a {actual_dim}")
            self.dimension = actual_dim
            
            #adjustment of dimention 
            
            Config.SENTENCE_TRANSFORMER_DIMENSION = actual_dim

        print(f"✓ Modelo cargado. Dimensión: {self.dimension}")

    def embed_text(self, text: str) -> List[float]:
        """Genera el embedding de un texto único"""
        if not text or not text.strip():
            return [0.0] * self.dimension

         # return list of embedding vector
            
        embedding = self.model.encode(
            text,
            convert_to_numpy=True,
            show_progress_bar=False
        )
        return embedding.tolist()

    def embed_documents(self, texts: List[str], batch_size: int = 32,
                        show_progress: bool = True) -> List[List[float]]:
        """Genera embeddings para múltiples documentos en batch"""
        if not texts:
            return []

        embeddings = self.model.encode(
            texts,
            batch_size=batch_size,
            convert_to_numpy=True,
            show_progress_bar=show_progress
        )
        return embeddings.tolist()

    def embed_query(self, query: str) -> List[float]:
        """Genera el embedding de una consulta (fase online de RAG)"""
        return self.embed_text(query)

    def get_model_info(self) -> dict:
        """Obtiene información del modelo"""
        return {
            "provider": "Sentence Transformers",
            "model_name": self.model_name,
            "dimension": self.dimension,
            "max_seq_length": self.model.max_seq_length
        }


class OpenAIEmbeddings:
    """Embeddings usando la API de OpenAI (opcional, requiere API key)"""

    def __init__(self, api_key: str = None, model: str = None):
        try:
            from openai import OpenAI
        except ImportError:
            raise ImportError("openai no está instalado. Instálalo con: pip install openai")

        self.api_key = api_key or Config.OPENAI_API_KEY
        self.model_name = model or Config.OPENAI_MODEL
        self.dimension = Config.OPENAI_DIMENSION

        if not self.api_key or self.api_key == "tu-api-key-aqui":
            raise ValueError(
                "API Key de OpenAI no configurada. "
                "Configura OPENAI_API_KEY en el archivo .env"
            )

        self.client = OpenAI(api_key=self.api_key)
        print(f"✓ OpenAI Embeddings inicializado (modelo: {self.model_name})")

    def embed_text(self, text: str) -> List[float]:
        if not text or not text.strip():
            return [0.0] * self.dimension

        response = self.client.embeddings.create(input=text, model=self.model_name)
        return response.data[0].embedding

    def embed_documents(self, texts: List[str], batch_size: int = 100,
                        show_progress: bool = True) -> List[List[float]]:
        if not texts:
            return []

        embeddings = []
        
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            if show_progress:
                print(f"  Procesando batch {i // batch_size + 1}/{(len(texts) - 1) // batch_size + 1}...")
            response = self.client.embeddings.create(input=batch, model=self.model_name)
            embeddings.extend([item.embedding for item in response.data])

        return embeddings

    def embed_query(self, query: str) -> List[float]:
        return self.embed_text(query)

    def get_model_info(self) -> dict:
        return {
            "provider": "OpenAI",
            "model_name": self.model_name,
            "dimension": self.dimension
        }


def create_embeddings(provider: str = None):
    """
    Factory: crea el modelo de embeddings según la configuración

    Args:
        provider: "sentence-transformers" o "openai" (usa Config por defecto)

    Returns:
        Instancia del modelo de embeddings
    """
    provider = provider or Config.EMBEDDING_PROVIDER

    if provider.lower() == "openai":
        return OpenAIEmbeddings()
    elif provider.lower() == "sentence-transformers":
        return SentenceTransformerEmbeddings()
    else:
        raise ValueError(
            f"Proveedor '{provider}' no soportado. "
            "Usa 'sentence-transformers' o 'openai'"
        )


class EmbeddingModel:
    """
    Wrapper que usa automáticamente el proveedor configurado en .env
    """

    def __init__(self, provider: str = None):
        self._embedder = create_embeddings(provider)
        self.dimension = self._embedder.dimension
        self.model_name = self._embedder.model_name

    def embed_text(self, text: str) -> List[float]:
        return self._embedder.embed_text(text)

    def embed_documents(self, texts: List[str], batch_size: int = 32,
                        show_progress: bool = True) -> List[List[float]]:
        return self._embedder.embed_documents(texts, batch_size, show_progress)

    def embed_query(self, query: str) -> List[float]:
        return self._embedder.embed_query(query)

    def similarity(self, embedding1: List[float], embedding2: List[float]) -> float:
        """Similitud coseno entre dos embeddings"""
        return cosine_similarity(embedding1, embedding2)

    def get_model_info(self) -> dict:
        return self._embedder.get_model_info()


class TextChunker:
    """
    Divide textos en chunks (Taller Práctico - Parte 1)

    Tamaño recomendado: 200-500 tokens por chunk con overlap de 10-20%
    para mantener el contexto entre fragmentos.
    """

    def __init__(self, chunk_size: int = None, chunk_overlap: int = None):
        """
        Args:
            chunk_size: Tamaño máximo de cada chunk en caracteres
            chunk_overlap: Solapamiento entre chunks
        """
        self.chunk_size = chunk_size or Config.CHUNK_SIZE
        self.chunk_overlap = chunk_overlap or Config.CHUNK_OVERLAP

    def split_text(self, text: str) -> List[str]:
        """
        Divide un texto en chunks respetando los límites de oración
        
        (corta en el último punto o salto de línea antes del límite)
        """
        if not text or len(text) <= self.chunk_size:
            return [text] if text else []

        chunks = []
        start = 0

        while start < len(text):
            end = start + self.chunk_size

            # Buscar el último punto o salto de línea antes del límite
            # split by sentence boundary 
            
            if end < len(text):
                last_period = text.rfind('.', start, end)
                last_newline = text.rfind('\n', start, end)
                last_break = max(last_period, last_newline)

                if last_break > start:
                    end = last_break + 1

            chunk = text[start:end].strip()
            if chunk:
                chunks.append(chunk)

            # Mover el inicio dejando el solapamiento
            start = end - self.chunk_overlap if end < len(text) else end

        return chunks

    def split_documents(self, documents: List[str]) -> List[dict]:
        """
        Divide múltiples documentos en chunks con metadata

        Returns:
            Lista de diccionarios {"text": chunk, "metadata": {...}}
        """
        all_chunks = []

        for doc_id, doc in enumerate(documents):
            chunks = self.split_text(doc)

            for chunk_id, chunk in enumerate(chunks):
                all_chunks.append({
                    "text": chunk,
                    "metadata": {
                        "document_id": doc_id,
                        "chunk_id": chunk_id,
                        "total_chunks": len(chunks)
                    }
                })

        return all_chunks


def main():
    """Función de prueba rápida"""
    print("Probando embeddings y similitud coseno (Sesión 6)\n")

    # Crear modelo
    embedder = EmbeddingModel()

    # Textos de ejemplo
    texts = [
        "Los embeddings son representaciones vectoriales de texto",
        "Las representaciones vectoriales capturan el significado semántico",
        "Python es un lenguaje de programación popular",
    ]

    # Vectorizar
    embeddings = embedder.embed_documents(texts, show_progress=False)
    print(f"Generados {len(embeddings)} embeddings de dimensión {len(embeddings[0])}\n")

    # Similitud coseno: textos 0 y 1 son semánticamente cercanos,
    # el texto 2 (programación) debería estar más lejos
    print("Matriz de similitud coseno:")
    sim_matrix = cosine_similarity_matrix(embeddings)
    for row in sim_matrix:
        print("  " + "  ".join(f"{v:.3f}" for v in row))

    # Probar chunker
    chunker = TextChunker(chunk_size=100, chunk_overlap=20)
    long_text = "Los embeddings transforman texto en vectores numéricos. " * 10
    chunks = chunker.split_text(long_text)
    print(f"\nTexto dividido en {len(chunks)} chunks (size=100, overlap=20)")


if __name__ == "__main__":
    main()
