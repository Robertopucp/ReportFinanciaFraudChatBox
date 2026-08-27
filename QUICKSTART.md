# Inicio Rápido — Sesión 6: Recuperación de Información (RAG + ChromaDB)

## Instalación en 3 pasos (sin Docker, sin servidores)

### 1. Instalar dependencias

```bash
pip install -r requirements.txt
```

> Si solo quieres el notebook y ya tienes el resto instalado:
> `pip install chromadb jupyter sentence-transformers`

### 2. Verificar la instalación

```bash
python check_setup.py
```

### 3. Probar el notebook (recomendado, celda a celda)

```bash
jupyter notebook Taller_Sesion6.ipynb
```

O las demos por consola:

```bash
python main.py                  # Menú con 5 demos del taller
python demo_presentacion.py     # Buscar dentro de la presentación de la sesión
```

## Uso programático rápido

```python
from vector_search import VectorSearchEngine

# Crear motor (ChromaDB persistente, sin Docker)
engine = VectorSearchEngine(collection_name="demo")
engine.create_collection(delete_if_exists=True)

# Indexar (los embeddings se generan automáticamente)
docs = ["Los embeddings representan texto como vectores",
        "La similitud coseno mide el ángulo entre vectores",
        "RAG combina recuperación y generación"]
engine.add_documents(docs)

# Buscar por significado (no por palabras exactas)
results = engine.search("¿cómo se mide la cercanía entre textos?", k=2)
for r in results:
    print(f"{r['score']:.3f} - {r['text']}")

# Limpiar
engine.delete_collection()
```

## Comandos útiles

```bash
# Probar componentes individuales
python embeddings.py        # Embeddings + similitud coseno
python vector_search.py     # Motor completo con ChromaDB
python pdf_reader.py        # Extracción de texto de PDFs
python read_presentation.py # Extracción de texto de PowerPoint
python check_setup.py       # Diagnóstico de la instalación
```

## Solución de problemas

### Error al cargar el modelo de embeddings
La primera vez el modelo se descarga de HuggingFace (requiere internet).
Si ya está descargado, funciona offline. Verifica con:
```bash
python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2')"
```

### Error de ChromaDB
```bash
pip install --upgrade chromadb
```
La base de datos se guarda en `./chroma_db` (carpeta local, sin servidor).
Si quieres empezar de cero, borra esa carpeta.

### Falta algún paquete
```bash
pip install -r requirements.txt
python check_setup.py
```

## Próximos pasos

1. Ejecuta el notebook `Taller_Sesion6.ipynb` celda a celda
2. Lee el [README.md](README.md) para entender la arquitectura RAG
3. Cambia los PDFs en `pdfs/` y vuelve a indexar
4. Prueba el otro modelo de embeddings (`all-MiniLM-L6-v2`) en `.env`
5. Agrega tu API key de OpenAI en `.env` para activar la generación en la demo RAG
