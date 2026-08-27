"""
Motor de Búsqueda Semántica con ChromaDB

Implementación de la arquitectura RAG vista en clase:

Fase de Indexación (offline):
  1. Recopilación de documentos
  2. Preprocesamiento y limpieza
  3. Chunking (TextChunker)
  4. Embedding (Sentence Transformers)
  5. Indexación en base de datos vectorial (ChromaDB)

Fase de Consulta (online):
  1. Usuario envía una pregunta
  2. Embedding de la pregunta
  3. Búsqueda Top-K chunks más similares (similitud coseno)
  4. Construcción del prompt aumentado (RAG)
"""

from typing import List, Dict, Optional
import uuid

from embeddings import EmbeddingModel, TextChunker
from config import Config
from tqdm import tqdm


class VectorSearchEngine:
    """
    Motor de búsqueda vectorial completo usando ChromaDB como base de
    datos vectorial persistente (sin Docker ni servidores externos).
    """

    def __init__(self, collection_name: str = None,
                 persist_dir: str = None,
                 embedding_model: EmbeddingModel = None):
        """
        Inicializa el motor de búsqueda

        Args:
            collection_name: Nombre de la colección (usa Config.COLLECTION_NAME)
            persist_dir: Carpeta donde ChromaDB guarda los vectores
            embedding_model: Modelo de embeddings (crea uno nuevo si no se provee)
        """
        self.collection_name = collection_name or Config.COLLECTION_NAME
        self.persist_dir = persist_dir or Config.CHROMA_PERSIST_DIR

        print("Inicializando motor de búsqueda semántica (ChromaDB)...")

        # Componente 1: modelo de embeddings
        self.embedder = embedding_model or EmbeddingModel()

        # Componente 2: base de datos vectorial persistente
        try:
            import chromadb
        except ImportError:
            raise ImportError(
                "chromadb no está instalado. Instálalo con: pip install chromadb"
            )

        self.client = chromadb.PersistentClient(path=self.persist_dir)
        self.collection = None

        print(f"ChromaDB persistente en: {self.persist_dir}")
        print("Motor de búsqueda inicializado\n")

    # ------------------------------------------------------------------
    # FASE DE INDEXACIÓN (offline)
    # ------------------------------------------------------------------

    def create_collection(self, delete_if_exists: bool = False) -> bool:
        """
        Crea (o recupera) la colección vectorial

        La métrica usada es similitud coseno, como se vio en clase.

        Args:
            delete_if_exists: Borrar la colección anterior antes de crear

        Returns:
            True si la colección quedó lista
        """
        if delete_if_exists:
            self.delete_collection()

        # hnsw:space = cosine  ->  las distancias devueltas por ChromaDB
        # serán (1 - similitud_coseno), es decir: score = 1 - distance
        
        self.collection = self.client.get_or_create_collection(
            name=self.collection_name,
            embedding_function=None,  # pasamos los vectores explícitamente
            metadata={"hnsw:space": "cosine"}
        )

        count = self.collection.count()
        print(f"✓ Colección '{self.collection_name}' lista ({count} documentos existentes)")
        return True

    def add_documents(self, texts: List[str],
                      metadatas: List[Dict] = None,
                      ids: List[str] = None,
                      batch_size: int = 32,
                      show_progress: bool = True) -> int:
        """
        Añade documentos a la colección (los vectoriza y los indexa)

        Args:
            texts: Lista de textos a indexar
            metadatas: Metadatos de cada texto (fuente, página, sección...)
            ids: IDs personalizados (se generan automáticamente si no se pasan)
            batch_size: Tamaño del batch para generar embeddings
            show_progress: Mostrar progreso

        Returns:
            Número de documentos indexados
        """
        if not texts:
            print("No hay documentos para indexar")
            return 0

        if self.collection is None:
            self.create_collection()

        print(f"Indexando {len(texts)} documentos...")

        # Paso 4 de la fase offline: convertir chunks en vectores
        print("Generando embeddings...")
        
        embeddings = self.embedder.embed_documents(
            texts,# List of texts to embed
            batch_size=batch_size, # Batch size for embedding
            show_progress=show_progress
        )

        # Preparar metadatos: ChromaDB solo acepta str/int/float/bool y
        # rechaza diccionarios vacíos (usa None en su lugar)
        
        if metadatas is None:
            metadatas = [None for _ in texts]
        else:
            metadatas = [
                ({k: v for k, v in m.items() if isinstance(v, (str, int, float, bool))} or None)
                for m in metadatas
            ]

        if all(m is None for m in metadatas):
            metadatas = None

        # IDs únicos
        if ids is None:
            ids = [f"doc_{uuid.uuid4().hex[:12]}" for _ in texts]

        # ID aleatorio lo convierte en una cadena hexadecimal sin guione 
        # solo 12 caracteres 

        # Paso 5 de la fase offline: indexar en la base de datos vectorial.
        # Se usa upsert: si un ID ya existe, se actualiza (permite re-ejecutar
        # celdas del notebook o reindexar sin errores de IDs duplicados).
        self.collection.upsert(
            ids=ids,
            documents=texts,
            embeddings=embeddings,
            metadatas=metadatas
        )

        print(f"✓ {len(texts)} documentos indexados en '{self.collection_name}'")
        return len(texts)

    def add_documents_chunked(self, texts: List[str],
                              chunk_size: int = None,
                              chunk_overlap: int = None,
                              metadatas: List[Dict] = None,
                              **kwargs) -> int:
        """
        Añade documentos dividiéndolos primero en chunks
        (pasos 2-3 de la fase offline: preprocesamiento + chunking)

        Args:
            texts: Lista de documentos largos
            chunk_size: Tamaño de cada chunk en caracteres
            chunk_overlap: Solapamiento entre chunks (10-20% recomendado)
            metadatas: Metadatos originales de cada documento
            **kwargs: Argumentos adicionales para add_documents

        Returns:
            Número de chunks indexados
        """
        print(f"Dividiendo {len(texts)} documentos en chunks...")

        chunker = TextChunker(chunk_size, chunk_overlap)

        all_chunks = []
        all_metadatas = []

        for doc_idx, text in enumerate(tqdm(texts, desc="Procesando documentos")):
            
            chunks = chunker.split_text(text) # Split the document into chunks

            for chunk_idx, chunk in enumerate(chunks):
                all_chunks.append(chunk)

                # Combinar metadata original con la info del chunk
                chunk_metadata = (metadatas[doc_idx].copy()
                                  if metadatas and doc_idx < len(metadatas) else {})
                chunk_metadata.update({
                    "source_doc_id": doc_idx,
                    "chunk_id": chunk_idx,
                    "total_chunks": len(chunks)
                })
                all_metadatas.append(chunk_metadata)

        print(f"Total de chunks generados: {len(all_chunks)}")

        return self.add_documents(
            all_chunks,
            metadatas=all_metadatas,
            **kwargs
        )

    # ------------------------------------------------------------------
    # FASE DE CONSULTA (online)
    # ------------------------------------------------------------------

    def search(self, query: str, k: int = None,
               score_threshold: float = None,
               filter_metadata: Dict = None) -> List[Dict]:
        """
        Búsqueda semántica por similitud coseno

        Pasos 2-3 de la fase online: embedding de la pregunta +
        búsqueda Top-K de los chunks más similares.

        Args:
            query: Pregunta del usuario
            k: Número de resultados (Top-K)
            score_threshold: Umbral mínimo de similitud (0 a 1)
            filter_metadata: Filtro por metadatos, ej: {"source": "manual.pdf"}

        Returns:
            Lista de resultados con texto, score y metadata
        """
        if self.collection is None or self.collection.count() == 0:
            print("No hay documentos indexados. Ejecuta add_documents primero.")
            return []

        k = k or Config.DEFAULT_TOP_K
        score_threshold = score_threshold if score_threshold is not None else Config.DEFAULT_SCORE_THRESHOLD

        # Embedding de la consulta
        query_embedding = self.embedder.embed_query(query)

        # Búsqueda Top-K en ChromaDB
        response = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=min(k, self.collection.count()),
            where=filter_metadata,
            include=["documents", "metadatas", "distances"]
        )

        results = []
        
        for i in range(len(response["documents"][0])):
            distance = response["distances"][0][i]
            # distance = 1 - coseno  ->  score = 1 - distance
            score = 1.0 - distance

            if score < score_threshold:
                continue

            results.append({
                "id": response["ids"][0][i],
                "score": round(float(score), 4),
                "text": response["documents"][0][i],
                "metadata": response["metadatas"][0][i] or {}
            })

        return results

    def similarity_search(self, query: str, k: int = None) -> List[str]:
        """Búsqueda que retorna solo los textos encontrados"""
        results = self.search(query, k=k, score_threshold=0.0)
        return [r['text'] for r in results]

    def similarity_search_with_score(self, query: str, k: int = None) -> List[tuple]:
        """Búsqueda que retorna tuplas (texto, score)"""
        results = self.search(query, k=k, score_threshold=0.0)
        return [(r['text'], r['score']) for r in results]

    # ------------------------------------------------------------------
    # RAG (Retrieval Augmented Generation)
    # ------------------------------------------------------------------

    def build_rag_prompt(self, question: str, 
                         k: int = None,
                         system_instruction: str = None) -> tuple:
        """
        Construye el prompt aumentado de RAG: recupera los chunks más
        relevantes y los inyecta como contexto en el prompt del LLM.

        Returns:
            (prompt_completo, resultados_recuperados)
        """
        k = k or Config.RAG_CONTEXT_CHUNKS

        # 1. Retrieval: recuperar contexto relevante
        context_chunks = self.search(question,
                                     k=k, 
                                     score_threshold=0.0)

        if not context_chunks:
            return None, []

        # 2. Augmented: armar el prompt con el contexto recuperado
        context_text = "\n\n".join(
            f"[Contexto {i} - fuente: {c['metadata'].get('source', c['metadata'].get('filename', 'desconocida'))}]\n{c['text']}"
            for i, c in enumerate(context_chunks, 1)
        )

        system_instruction = system_instruction or (
            "Eres un asistente corporativo. Responde ÚNICAMENTE con base en el "
            "contexto proporcionado. Si la información no está en el contexto, "
            "indícalo claramente. Cita la fuente de cada afirmación."
        )

        prompt = (
            f"{system_instruction}\n\n"
            f"CONTEXTO:\n{context_text}\n\n"
            f"PREGUNTA: {question}\n\n"
            f"RESPUESTA:"
        )

        return prompt, context_chunks

    def generate_answer(self, question: str, k: int = None) -> Dict:
        """
        RAG completo: recuperación + generación con un LLM (OpenAI).
        Si no hay API key configurada, solo devuelve el prompt aumentado.

        Returns:
            Diccionario con la respuesta, el prompt y las fuentes usadas
        """
        prompt, context_chunks = self.build_rag_prompt(question, k=k)

        if prompt is None:
            return {"answer": "No se encontró información relevante en la base de conocimiento.",
                    "prompt": None, "sources": []}

        if not Config.OPENAI_API_KEY or Config.OPENAI_API_KEY == "tu-api-key-aqui":
            return {
                "answer": None,
                "prompt": prompt,
                "sources": [c['metadata'] for c in context_chunks],
                "note": "Sin API key de OpenAI: se devuelve solo el prompt aumentado."
            }

        # 3. Generation: el LLM responde basado en los chunks recuperados
        try:
            from openai import OpenAI
            client = OpenAI(api_key=Config.OPENAI_API_KEY)
            response = client.chat.completions.create(
                model=Config.OPENAI_CHAT_MODEL,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.2
            )
            answer = response.choices[0].message.content
            
        except Exception as e:
            answer = f"(Error al llamar al LLM: {e})"

        return {
            "answer": answer,
            "prompt": prompt,
            "sources": [c['metadata'] for c in context_chunks]
        }

    # ------------------------------------------------------------------
    # Utilidades
    # ------------------------------------------------------------------

    def get_stats(self) -> Dict:
        """Obtiene estadísticas de la colección"""
        if self.collection is None:
            return {"document_count": 0, "exists": False}

        return {
            "document_count": self.collection.count(),
            "collection_name": self.collection_name,
            "persist_dir": self.persist_dir,
            "embedding_dimension": self.embedder.dimension,
            "model_name": self.embedder.model_name,
            "exists": True
        }

    def delete_collection(self) -> bool:
        """Elimina la colección y todos sus vectores del disco"""
        try:
            self.client.delete_collection(name=self.collection_name)
            print(f"✓ Colección '{self.collection_name}' eliminada")
            self.collection = None
            return True
        except Exception:
            return False  # No existía


def main():
    """Ejemplo de uso completo con el caso de uso de la sesión"""
    print("=" * 70)
    print("MOTOR DE BÚSQUEDA SEMÁNTICA - CASO DE USO DE LA SESIÓN 6")
    print("=" * 70 + "\n")

    # Caso de uso: consultar políticas internas de una empresa
    engine = VectorSearchEngine(collection_name="demo_sesion6")
    engine.create_collection(delete_if_exists=True)

    documents = [
        "Las políticas internas de la empresa establecen que el teletrabajo está permitido dos días por semana",
        "El manual técnico indica que el mantenimiento de servidores se realiza el primer domingo de cada mes",
        "La documentación legal exige conservar los contratos de los clientes por un mínimo de cinco años",
        "El reglamento interno prohíbe compartir credenciales de acceso entre empleados",
        "La política de vacaciones otorga quince días hábiles por año trabajado",
        "El plan de continuidad del negocio se activa ante incidentes de seguridad críticos",
        "Los gastos de representación requieren aprobación del gerente de área",
        "El código de ética prohíbe aceptar regalos de proveedores mayores a cincuenta dólares",
    ]

    count = engine.add_documents(documents)
    print(f"\n✓ {count} documentos indexados\n")

    queries = [
        "¿Puedo trabajar desde casa?",
        "¿Qué dice el código de ética sobre los regalos?",
        "¿Cuántos días de vacaciones tengo al año?",
        "¿Cada cuánto se mantienen los servidores?",
    ]

    for query in queries:
        print(f"\n{'=' * 70}")
        print(f"Query: {query}")
        print('=' * 70)

        results = engine.search(query, k=2)

        for i, result in enumerate(results, 1):
            print(f"\n{i}. [Score: {result['score']:.4f}]")
            print(f"   {result['text']}")

    # Demo RAG: prompt aumentado
    print(f"\n{'=' * 70}")
    print("DEMO RAG: PROMPT AUMENTADO")
    print('=' * 70)
    question = "¿Qué normas debo cumplir como empleado?"
    rag = engine.generate_answer(question)
    print(rag["prompt"][:800] + "...\n")

    # Estadísticas
    print(f"\n{'=' * 70}")
    print("ESTADÍSTICAS")
    print('=' * 70)
    stats = engine.get_stats()
    for key, value in stats.items():
        print(f"{key}: {value}")

    # Limpiar
    engine.delete_collection()
    print("\nDemo completada")


if __name__ == "__main__":
    main()
