# Cómo buscar en PDFs (Sesión 6 — Taller Práctico Parte 1 y 3)

## 🚀 Inicio Rápido

### 1. Instalar la librería de PDFs

```bash
pip install pypdf
```

### 2. Crear carpeta con PDFs

```bash
# Crear carpeta
mkdir pdfs

# Copiar tus PDFs a esta carpeta
```

El proyecto ya incluye dos PDFs de ejemplo en `pdfs/`:
- `CATÁLOGO DE PRODUCTOS 2025.pdf`
- `INCYTU_18-012.pdf` (documento sobre inteligencia artificial)

### 3. Buscar en PDFs con el notebook (recomendado)

Abre `Taller_Sesion6.ipynb` con Jupyter:

```bash
jupyter notebook Taller_Sesion6.ipynb
```

Y ejecuta la sección **"Parte 3: PDFs → ChromaDB → Búsqueda semántica"**.
Puedes cambiar la consulta en la celda `CONSULTAS` y volver a ejecutarla
sin reindexar (la colección queda persistida en `./chroma_db`).

### 4. Buscar en PDFs por consola

```bash
python main.py
# Elegir la opción 3: PDFs -> ChromaDB persistente -> búsqueda
```

## 🔧 Personalizar la búsqueda

En el notebook (o en un script):

```python
from vector_search import VectorSearchEngine
from pdf_reader import PDFReader

# 1. Leer PDFs (extracción y limpieza)
reader = PDFReader()
documentos = reader.read_pdf_folder("./pdfs")

# 2. Indexar con chunking (size y overlap configurables)
engine = VectorSearchEngine(collection_name="mis_pdfs")
engine.create_collection(delete_if_exists=True)

engine.add_documents_chunked(
    [doc["text"] for doc in documentos],
    chunk_size=500,          # 200-500 tokens recomendado
    chunk_overlap=50,        # 10% de overlap
    metadatas=[doc["metadata"] for doc in documentos],
)

# 3. Buscar
resultados = engine.search("¿Qué es la inteligencia artificial?", k=5)
for r in resultados:
    print(f"[{r['score']:.3f}] {r['metadata'].get('filename')}")
    print(f"  {r['text'][:200]}...")
```

## 💡 Consejos

- **Metadatos**: cada chunk conserva `filename`, `size`, `chunk_id`, etc.
  Puedes filtrar por archivo: `engine.search(query, filter_metadata={"filename": "INCYTU_18-012.pdf"})`
- **Persistencia**: la colección queda en `./chroma_db`. Para reindexar desde
  cero usa `engine.create_collection(delete_if_exists=True)` o borra la carpeta.
- **PDFs escaneados (imágenes)**: `pypdf` no extrae texto de imágenes.
  Necesitarías OCR (p. ej. `pytesseract`) para esos casos.
- **Chunking**: reduce `chunk_size` a ~200 para respuestas más precisas,
  auméntalo a ~800 para más contexto por fragmento.
- **Idioma**: el modelo por defecto (`paraphrase-multilingual-MiniLM-L12-v2`)
  funciona bien en español. Si tus PDFs están en inglés puedes usar
  `all-MiniLM-L6-v2` en `.env` (más ligero y rápido).
