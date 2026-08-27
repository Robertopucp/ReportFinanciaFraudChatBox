"""
Script principal - Sesión 6: Recuperación de Información (RAG)

Demos del caso de uso de la sesión: sistema de búsqueda semántica para
consultar políticas internas, manuales técnicos y documentación legal
de una empresa.
"""
from pathlib import Path
from vector_search import VectorSearchEngine
from config import Config


def demo_basica():
    """Demo 1: Búsqueda semántica básica (embeddings + similitud coseno)"""
    print("\n" + "=" * 70)
    print("DEMO 1: BÚSQUEDA SEMÁNTICA BÁSICA")
    print("=" * 70 + "\n")

    engine = VectorSearchEngine(collection_name="demo_basica")
    engine.create_collection(delete_if_exists=True)

    # Base de conocimiento empresarial (caso de uso de la sesión)
    documents = [
        "Las políticas internas permiten el teletrabajo dos días por semana con autorización del jefe directo",
        "El manual técnico establece que las contraseñas deben cambiarse cada noventa días",
        "La documentación legal exige firmar un acuerdo de confidencialidad al ingresar a la empresa",
        "El reglamento interno prohíbe el uso de dispositivos personales para datos confidenciales",
        "La política de capacitación otorga veinte horas anuales de formación pagadas por la empresa",
        "El plan de emergencia indica evacuar por las escaleras en caso de sismo",
        "Los gastos de viaje se reembolsan en un plazo máximo de quince días",
        "El código de ética prohíbe aceptar regalos de proveedores",
    ]

    print("Indexando base de conocimiento...")
    count = engine.add_documents(documents, show_progress=False)
    print(f"✓ {count} documentos indexados\n")

    queries = [
        "¿Puedo trabajar desde mi casa?",
        "¿Qué dice la empresa sobre las contraseñas?",
        "¿Qué debo hacer si tiembla?",
        "¿Me pagan los cursos de capacitación?",
    ]

    for query in queries:
        print(f"\n{'=' * 70}")
        print(f"Query: '{query}'")
        print('=' * 70)

        results = engine.search(query, k=2)

        for i, result in enumerate(results, 1):
            print(f"\n{i}. [Score: {result['score']:.4f}]")
            print(f"   {result['text']}")

    engine.delete_collection()
    print("\n✓ Demo completada\n")


def demo_chunking():
    """Demo 2: Chunking de documentos largos (Taller Parte 1)"""
    print("\n" + "=" * 70)
    print("DEMO 2: CHUNKING + BÚSQUEDA EN DOCUMENTOS LARGOS")
    print("=" * 70 + "\n")

    engine = VectorSearchEngine(collection_name="demo_chunks")
    engine.create_collection(delete_if_exists=True)

    # Manual técnico largo (se divide en chunks antes de indexar)
    manual_tecnico = """
    MANUAL TÉCNICO DE SEGURIDAD INFORMÁTICA. Capítulo 1: Acceso a sistemas.
    Todo colaborador debe usar autenticación de dos factores para acceder a
    los sistemas internos de la empresa. Las credenciales son personales e
    intransferibles. Capítulo 2: Respaldos. Los respaldos de la información
    se realizan diariamente a las tres de la mañana y se conservan copias
    por treinta días. Capítulo 3: Incidentes. Ante un incidente de seguridad,
    se debe notificar al área de TI dentro de la primera hora y aislar el
    equipo afectado de la red. Capítulo 4: Software. Solo se permite instalar
    software aprobado por el área de TI; cualquier excepción requiere una
    solicitud formal. Capítulo 5: Correo electrónico. Los adjuntos de remitentes
    desconocidos deben verificarse antes de abrirse para evitar malware.
    """

    print("Dividiendo el manual en chunks (size=200, overlap=50)...")
    count = engine.add_documents_chunked(
        [manual_tecnico],
        chunk_size=200,
        chunk_overlap=50,
        metadatas=[{"source": "manual_tecnico.pdf", "section": "seguridad"}],
        show_progress=False
    )
    print(f"✓ {count} chunks indexados\n")

    queries = [
        "¿Qué hago si hay un incidente de seguridad?",
        "¿Cada cuánto se hacen respaldos de información?",
        "¿Puedo instalar programas en mi computadora?",
    ]

    for query in queries:
        print(f"\nQuery: '{query}'\n")

        results = engine.search(query, k=2)

        for i, result in enumerate(results, 1):
            meta = result['metadata']
            print(f"{i}. [Score: {result['score']:.4f}] "
                  f"Chunk {meta.get('chunk_id', 0) + 1}/{meta.get('total_chunks', '?')}")
            print(f"   {result['text'][:150]}...")

    engine.delete_collection()
    print("\n✓ Demo completada\n")


def demo_pdfs():
    """Demo 3: Indexación de PDFs en ChromaDB persistente (Taller Parte 3)"""
    print("\n" + "=" * 70)
    print("DEMO 3: PDFs -> CHUNKS -> CHROMADB -> BÚSQUEDA")
    print("=" * 70 + "\n")

    from pdf_reader import PDFReader

    folder = Path("pdfs")
    if not folder.exists():
        print(f"❌ La carpeta '{folder}' no existe")
        return

    # Paso 1-2: extracción y limpieza de los PDFs
    print("Leyendo PDFs...")
    reader = PDFReader()
    documents = reader.read_pdf_folder(str(folder))

    if not documents:
        print("No se encontraron PDFs con texto")
        return

    textos = [doc["text"] for doc in documents]
    metadatas = [doc["metadata"] for doc in documents]

    # Paso 3-5: chunking + embedding + indexación (persistente)
    engine = VectorSearchEngine(collection_name="pdfs_empresa")
    engine.create_collection(delete_if_exists=True)

    count = engine.add_documents_chunked(
        textos,
        chunk_size=Config.CHUNK_SIZE,
        chunk_overlap=Config.CHUNK_OVERLAP,
        metadatas=metadatas,
        show_progress=True
    )
    print(f"✓ {count} chunks indexados en ChromaDB (persistente)\n")

    queries = [
        "¿Cuáles son los productos del catálogo?",
        "¿Qué es la inteligencia artificial según el documento?",
    ]

    for query in queries:
        print(f"\nQuery: '{query}'\n")
        results = engine.search(query, k=3)

        for i, result in enumerate(results, 1):
            meta = result['metadata']
            print(f"{i}. [Score: {result['score']:.4f}] 📄 {meta.get('filename', 'N/A')}")
            print(f"   {result['text'][:180]}...")

    print(f"\n💾 La colección '{engine.collection_name}' quedó guardada en "
          f"'{engine.persist_dir}'. Puedes reutilizarla en el notebook sin reindexar.")
    print("\n✓ Demo completada\n")


def demo_rag():
    """Demo 4: RAG - prompt aumentado y generación (si hay API key)"""
    print("\n" + "=" * 70)
    print("DEMO 4: RAG (RETRIEVAL AUGMENTED GENERATION)")
    print("=" * 70 + "\n")

    engine = VectorSearchEngine(collection_name="demo_rag")
    engine.create_collection(delete_if_exists=True)

    documents = [
        "La política de teletrabajo permite dos días remotos por semana previa coordinación con el jefe directo",
        "El horario laboral es de lunes a viernes de 9:00 a 18:00 con una hora de almuerzo",
        "Las vacaciones se programan con treinta días de anticipación y son quince días hábiles al año",
        "El seguro médico cubre al trabajador y a un familiar directo",
    ]
    engine.add_documents(documents, show_progress=False)

    question = "¿Cuántos días de vacaciones me corresponden y con cuánta anticipación debo pedirlos?"

    print(f"Pregunta del usuario: {question}\n")
    result = engine.generate_answer(question)

    print("─" * 70)
    print("CONTEXTO RECUPERADO (Retrieval):")
    for i, source in enumerate(result["sources"], 1):
        print(f"  {i}. {source}")
    print("─" * 70)

    if result["answer"] is not None:
        print("\nRESPUESTA GENERADA (Generation):")
        print(f"  {result['answer']}\n")
    else:
        print("\n(No hay API key de OpenAI configurada: se muestra el prompt aumentado)")
        print("─" * 70)
        print(result["prompt"][:1000])
        print("─" * 70)

    engine.delete_collection()
    print("\n✓ Demo completada\n")


def demo_interactiva():
    """Demo 5: Búsqueda interactiva en la base de conocimiento"""
    print("\n" + "=" * 70)
    print("DEMO 5: BÚSQUEDA INTERACTIVA")
    print("=" * 70 + "\n")

    engine = VectorSearchEngine(collection_name="demo_interactiva")
    engine.create_collection(delete_if_exists=True)

    knowledge_base = [
        "El teletrabajo está permitido dos días por semana",
        "Las contraseñas deben cambiarse cada noventa días",
        "Los respaldos se realizan diariamente a las 3:00 am",
        "El horario laboral es de 9:00 a 18:00",
        "Las vacaciones son quince días hábiles al año",
        "El código de ética prohíbe aceptar regalos de proveedores",
        "La capacitación anual es de veinte horas pagadas",
        "Los gastos de viaje se reembolsan en quince días",
    ]

    engine.add_documents(knowledge_base, show_progress=False)
    print(f"✓ {len(knowledge_base)} políticas indexadas\n")
    print("Pregunta sobre las políticas de la empresa.")
    print("Escribe 'salir' para terminar.\n")

    while True:
        try:
            query = input("Tu pregunta: ").strip()

            if not query:
                continue
            if query.lower() in ['salir', 'exit', 'quit']:
                break

            results = engine.search(query, k=3)

            if not results:
                print("No se encontraron resultados relevantes.\n")
                continue

            print(f"\nResultados encontrados:\n")
            for i, result in enumerate(results, 1):
                print(f"{i}. [{result['score']:.3f}] {result['text']}")
            print()

        except KeyboardInterrupt:
            print("\n\nInterrumpido por el usuario")
            break
        except Exception as e:
            print(f"Error: {e}\n")

    engine.delete_collection()
    print("\n✓ Demo completada\n")


def show_menu():
    """Muestra el menú de opciones"""
    print("\n" + "=" * 70)
    print("SESIÓN 6 - RECUPERACIÓN DE INFORMACIÓN: EMBEDDINGS + RAG + CHROMADB")
    print("Caso de uso: búsqueda semántica en documentos empresariales")
    print("=" * 70)

    Config.print_config()

    print("\nSelecciona una demo:")
    print("1. Búsqueda semántica básica (embeddings + similitud coseno)")
    print("2. Chunking de documentos largos")
    print("3. PDFs -> ChromaDB persistente -> búsqueda")
    print("4. RAG (prompt aumentado + generación)")
    print("5. Búsqueda interactiva")
    print("6. Ejecutar todas")
    print("7. Salir")
    print()


def main():
    """Función principal"""
    while True:
        show_menu()

        try:
            choice = input("Tu elección (1-7): ").strip()

            if choice == '1':
                demo_basica()
            elif choice == '2':
                demo_chunking()
            elif choice == '3':
                demo_pdfs()
            elif choice == '4':
                demo_rag()
            elif choice == '5':
                demo_interactiva()
            elif choice == '6':
                demo_basica()
                demo_chunking()
                demo_rag()
            elif choice == '7':
                print("\n¡Hasta luego!\n")
                break
            else:
                print("\n⚠️  Opción inválida. Intenta de nuevo.\n")

            input("\nPresiona Enter para continuar...")

        except KeyboardInterrupt:
            print("\n\n¡Hasta luego!\n")
            break
        except Exception as e:
            print(f"\n❌ Error: {e}\n")
            import traceback
            traceback.print_exc()
            input("Presiona Enter para continuar...")


if __name__ == "__main__":
    main()
