"""
Script para verificar que todo esté correctamente instalado
(Sesión 6 - Recuperación de Información con ChromaDB)
"""
import sys
from pathlib import Path


def check_python_version():
    """Verifica la versión de Python"""
    print("Verificando versión de Python...")
    version = sys.version_info
    if version.major == 3 and version.minor >= 9:
        print(f"  ✓ Python {version.major}.{version.minor}.{version.micro}")
        return True
    else:
        print(f"  ✗ Python {version.major}.{version.minor}.{version.micro} (se requiere 3.9+)")
        return False


def check_dependencies():
    """Verifica que las dependencias estén instaladas"""
    print("\nVerificando dependencias...")

    dependencies = {
        "chromadb": "ChromaDB (base de datos vectorial)",
        "sentence_transformers": "Sentence Transformers (embeddings)",
        "torch": "PyTorch",
        "numpy": "NumPy",
        "tqdm": "TQDM",
        "dotenv": "python-dotenv",
        "pypdf": "PyPDF (lectura de PDFs)",
        "pptx": "python-pptx (lectura de PowerPoint)",
    }

    all_ok = True
    for module, name in dependencies.items():
        try:
            __import__(module)
            print(f"  ✓ {name}")
        except ImportError:
            print(f"  ✗ {name} - NO INSTALADO")
            all_ok = False

    if not all_ok:
        print("\n  Solución: pip install -r requirements.txt")

    return all_ok


def check_chromadb():
    """Verifica que ChromaDB funcione (persistencia en disco, sin Docker)"""
    print("\nVerificando ChromaDB...")

    try:
        import chromadb
        import tempfile
        import shutil

        tmp_dir = tempfile.mkdtemp(prefix="chroma_test_")
        client = chromadb.PersistentClient(path=tmp_dir)
        collection = client.get_or_create_collection(
            name="test_setup",
            embedding_function=None,
            metadata={"hnsw:space": "cosine"}
        )
        collection.add(
            ids=["t1"],
            documents=["prueba de conexión"],
            embeddings=[[0.1] * 384],
        )
        count = collection.count()
        shutil.rmtree(tmp_dir, ignore_errors=True)

        print(f"  ✓ ChromaDB {chromadb.__version__} funciona (colección de prueba creada con {count} doc)")
        return True

    except Exception as e:
        print(f"  ✗ Error con ChromaDB: {e}")
        return False


def check_embeddings_model():
    """Verifica que se pueda cargar el modelo de embeddings"""
    print("\nVerificando modelo de embeddings...")

    try:
        from embeddings import EmbeddingModel

        print("  Cargando modelo (puede tomar un momento la primera vez)...")
        embedder = EmbeddingModel()

        # Probar embedding
        test_text = "Prueba de embedding"
        embedding = embedder.embed_text(test_text)

        print(f"  ✓ Modelo cargado: {embedder.model_name}")
        print(f"  ✓ Dimensión: {len(embedding)}")

        return True

    except Exception as e:
        print(f"  ✗ Error al cargar modelo: {e}")
        return False


def check_files():
    """Verifica que todos los archivos necesarios existan"""
    print("\nVerificando archivos del proyecto...")

    required_files = [
        "config.py",
        "embeddings.py",
        "vector_search.py",
        "pdf_reader.py",
        "read_presentation.py",
        "main.py",
        "requirements.txt",
        "Taller_Sesion6.ipynb",
        ".env",
    ]

    all_ok = True
    for file in required_files:
        path = Path(file)
        if path.exists():
            print(f"  ✓ {file}")
        else:
            print(f"  ✗ {file} - NO ENCONTRADO")
            all_ok = False

    return all_ok


def main():
    """Función principal"""
    print("=" * 70)
    print("VERIFICACIÓN DE INSTALACIÓN - SESIÓN 6 (RAG + CHROMADB)")
    print("=" * 70 + "\n")

    checks = [
        ("Python", check_python_version),
        ("Archivos", check_files),
        ("Dependencias", check_dependencies),
        ("ChromaDB", check_chromadb),
        ("Embeddings", check_embeddings_model),
    ]

    results = {}

    for name, check_func in checks:
        try:
            results[name] = check_func()
        except Exception as e:
            print(f"\n  ✗ Error inesperado: {e}")
            results[name] = False

    # Resumen
    print("\n" + "=" * 70)
    print("RESUMEN")
    print("=" * 70 + "\n")

    all_passed = True
    for name, passed in results.items():
        status = "✓ OK" if passed else "✗ FALLO"
        print(f"  {status:8} - {name}")
        if not passed:
            all_passed = False

    print("\n" + "=" * 70)

    if all_passed:
        print("\n🎉 ¡Todo está correctamente configurado!")
        print("\nPuedes ejecutar:")
        print("  jupyter notebook Taller_Sesion6.ipynb   (recomendado: probar celda a celda)")
        print("  python main.py                          (demos del taller)")
        print("  python demo_presentacion.py             (buscar dentro de la presentación de la sesión)")
    else:
        print("\n⚠️  Hay algunos problemas que resolver.")
        print("\nConsulta el README.md o QUICKSTART.md para más información.")

    print()


if __name__ == "__main__":
    main()
