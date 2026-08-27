"""
Demo: Indexar y buscar en la presentación de la Sesión 6

Aplica el pipeline completo de la sesión a su propio material:
extracción (pptx) -> chunking -> embeddings -> ChromaDB -> búsqueda semántica.
"""
import sys
from pathlib import Path

# Permitir importar desde el directorio actual
sys.path.insert(0, str(Path(__file__).parent))

from read_presentation import PresentationReader
from vector_search import VectorSearchEngine


def index_presentation(pptx_path: str, collection_name: str = "presentacion_sesion6"):
    """
    Indexa el contenido de una presentación PowerPoint

    Args:
        pptx_path: Ruta al archivo PowerPoint
        collection_name: Nombre de la colección en ChromaDB

    Returns:
        VectorSearchEngine configurado
    """
    print("=" * 70)
    print("INDEXANDO PRESENTACIÓN POWERPOINT")
    print("=" * 70 + "\n")

    # Paso 1-2: extracción y limpieza del contenido
    print(f"Leyendo: {pptx_path}")
    reader = PresentationReader(pptx_path)
    slides_data = reader.read_all_slides()
    print(f"✓ {len(slides_data)} diapositivas leídas\n")

    # Preparar documentos para indexar
    documents = []
    metadatas = []

    for slide in slides_data:
        # Combinar título y contenido
        slide_text = f"{slide['title']}\n\n"

        if slide['content']:
            slide_text += "\n".join(slide['content'])

        # Solo agregar si hay contenido
        if slide_text.strip():
            documents.append(slide_text.strip())
            metadatas.append({
                "slide_number": slide['slide_number'],
                "title": slide['title'],
                "total_slides": len(slides_data)
            })

    # Pasos 3-5: chunking + embedding + indexación en ChromaDB
    print("Inicializando motor de búsqueda...")
    engine = VectorSearchEngine(collection_name=collection_name)
    engine.create_collection(delete_if_exists=True)

    print(f"\nIndexando {len(documents)} diapositivas...")
    count = engine.add_documents_chunked(
        documents,
        chunk_size=400,
        chunk_overlap=50,
        metadatas=metadatas,
        show_progress=True
    )

    print(f"\n✓ {count} chunks indexados exitosamente")

    # Mostrar estadísticas
    stats = engine.get_stats()
    print(f"\nEstadísticas de la colección:")
    print(f"  - Chunks totales: {stats.get('document_count', 0)}")
    print(f"  - Dimensión embeddings: {stats.get('embedding_dimension', 0)}")
    print(f"  - Modelo: {stats.get('model_name', 'N/A')}\n")

    return engine


def search_presentation(engine: VectorSearchEngine, interactive: bool = True):
    """
    Búsqueda en la presentación

    Args:
        engine: Motor de búsqueda configurado
        interactive: Si es True, modo interactivo. Si es False, solo ejemplos
    """
    print("=" * 70)
    print("BÚSQUEDA EN LA PRESENTACIÓN")
    print("=" * 70 + "\n")

    if not interactive:
        # Consultas de ejemplo sobre los temas de la sesión
        example_queries = [
            "¿Qué son los embeddings?",
            "¿Cómo se calcula la similitud entre vectores?",
            "¿Qué es RAG?",
            "¿Qué bases de datos vectoriales existen?",
            "¿Cómo se hace el chunking?",
        ]

        for query in example_queries:
            print(f"\n{'─' * 70}")
            print(f"Query: '{query}'")
            print('─' * 70)

            results = engine.search(query, k=3, score_threshold=0.2)

            if not results:
                print("No se encontraron resultados relevantes")
                continue

            for i, result in enumerate(results, 1):
                metadata = result['metadata']
                print(f"\n{i}. [Score: {result['score']:.4f}] "
                      f"Diapositiva {metadata.get('slide_number', '?')}")
                if metadata.get('title'):
                    print(f"   Título: {metadata['title']}")
                print(f"   {result['text'][:200]}...")

        return

    # Modo interactivo
    print("Busca información en la presentación de la sesión.")
    print("Escribe 'salir' para terminar.\n")

    while True:
        try:
            query = input("\nTu pregunta: ").strip()

            if not query:
                continue
            if query.lower() in ['salir', 'exit', 'quit']:
                break

            results = engine.search(query, k=5, score_threshold=0.2)

            if not results:
                print("\nNo se encontraron resultados relevantes.")
                continue

            print(f"\n{'═' * 70}")
            print(f"Resultados para: '{query}'")
            print('═' * 70)

            for i, result in enumerate(results, 1):
                metadata = result['metadata']
                slide = metadata.get('slide_number', '?')
                chunk_info = (f"[Chunk {metadata.get('chunk_id', 0) + 1}/"
                              f"{metadata.get('total_chunks', 1)}]")

                print(f"\n{i}. [Score: {result['score']:.4f}] Diapositiva {slide} {chunk_info}")

                if metadata.get('title'):
                    print(f"   📊 {metadata['title']}")

                text = result['text']
                if len(text) > 300:
                    text = text[:300] + "..."
                print(f"\n   {text}\n")
                print(f"   {'-' * 68}")

        except KeyboardInterrupt:
            print("\n\nInterrumpido por el usuario")
            break
        except Exception as e:
            print(f"\nError: {e}")


def main():
    """Función principal"""
    # Ruta a la presentación de la sesión (en el directorio padre)
    pptx_path = Path(__file__).parent.parent / "Sesion6_Recuperación_de_información.pptx"

    if not pptx_path.exists():
        print(f"❌ Error: No se encuentra el archivo: {pptx_path}")
        return

    try:
        # Indexar presentación
        engine = index_presentation(str(pptx_path))

        # Menú
        print("\n¿Qué deseas hacer?")
        print("1. Búsqueda interactiva")
        print("2. Ejecutar consultas de ejemplo")
        print("3. Ambas")
        print()

        choice = input("Tu elección (1-3): ").strip()

        if choice == '1':
            search_presentation(engine, interactive=True)
        elif choice == '2':
            search_presentation(engine, interactive=False)
        elif choice == '3':
            search_presentation(engine, interactive=False)
            input("\nPresiona Enter para continuar a modo interactivo...")
            search_presentation(engine, interactive=True)
        else:
            print("\n⚠️  Opción inválida")

        # Preguntar si quiere mantener la colección
        print("\n" + "=" * 70)
        keep = input("¿Mantener la colección para futuras búsquedas? (s/n): ").strip().lower()

        if keep != 's':
            engine.delete_collection()
            print("✓ Colección eliminada")
        else:
            print(f"✓ Colección '{engine.collection_name}' mantenida")
            print(f"  Quedó persistida en '{engine.persist_dir}'. "
                  "Puedes reutilizarla sin reindexar.")

        print("\n¡Hasta luego!\n")

    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
