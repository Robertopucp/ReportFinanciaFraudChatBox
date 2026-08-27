"""
Lector de PDFs para búsqueda vectorial
Extrae texto de archivos PDF
"""
from pathlib import Path
from typing import List, Dict
import re


class PDFReader:
    """Lee y extrae texto de archivos PDF"""

    def __init__(self):
        """Inicializa el lector de PDF"""
        try:
            from pypdf import PdfReader
            self.PdfReader = PdfReader
        except ImportError:
            raise ImportError(
                "pypdf no está instalado. Instálalo con: pip install pypdf"
            )

    def extract_text_from_pdf(self, pdf_path: str) -> str:
        """
        Extrae texto de un archivo PDF

        Args:
            pdf_path: Ruta al archivo PDF

        Returns:
            Texto extraído del PDF
        """
        try:
            reader = self.PdfReader(pdf_path)
            text = ""

            for page in reader.pages:
                page_text = page.extract_text()
                if page_text:
                    text += page_text + "\n\n"

            # Limpiar texto
            text = self.clean_text(text)
            return text

        except Exception as e:
            print(f"Error al leer {pdf_path}: {e}")
            return ""

    def clean_text(self, text: str) -> str:
        """
        Limpia el texto extraído del PDF

        Args:
            text: Texto a limpiar

        Returns:
            Texto limpio
        """
        # Eliminar múltiples espacios
        text = re.sub(r'\s+', ' ', text)

        # Eliminar múltiples saltos de línea
        text = re.sub(r'\n\s*\n', '\n\n', text)

        # Eliminar espacios al inicio y final
        text = text.strip()

        return text

    def read_pdf_folder(self, folder_path: str, recursive: bool = False) -> List[Dict]:
        """
        Lee todos los PDFs de una carpeta

        Args:
            folder_path: Ruta a la carpeta con PDFs
            recursive: Si True, busca en subcarpetas también

        Returns:
            Lista de diccionarios con texto y metadata de cada PDF
        """
        folder = Path(folder_path)

        if not folder.exists():
            raise FileNotFoundError(f"La carpeta {folder_path} no existe")

        # Buscar archivos PDF
        if recursive:
            pdf_files = list(folder.rglob("*.pdf"))
        else:
            pdf_files = list(folder.glob("*.pdf"))

        if not pdf_files:
            print(f"⚠️  No se encontraron archivos PDF en {folder_path}")
            return []

        print(f"Encontrados {len(pdf_files)} archivos PDF")

        documents = []

        for pdf_file in pdf_files:
            print(f"  Leyendo: {pdf_file.name}...", end=" ")

            text = self.extract_text_from_pdf(str(pdf_file))

            if text:
                documents.append({
                    "text": text,
                    "metadata": {
                        "filename": pdf_file.name,
                        "filepath": str(pdf_file),
                        "size": pdf_file.stat().st_size,
                        "extension": ".pdf"
                    }
                })
                print(f"✓ ({len(text)} caracteres)")
            else:
                print("✗ (sin texto)")

        print(f"\n✓ Total documentos procesados: {len(documents)}")
        return documents

    def get_pdf_info(self, pdf_path: str) -> Dict:
        """
        Obtiene información de un PDF

        Args:
            pdf_path: Ruta al PDF

        Returns:
            Diccionario con información del PDF
        """
        try:
            reader = self.PdfReader(pdf_path)

            info = {
                "num_pages": len(reader.pages),
                "metadata": reader.metadata if reader.metadata else {},
                "filepath": pdf_path
            }

            return info

        except Exception as e:
            return {"error": str(e)}


def main():
    """Función de prueba"""
    import sys

    print("="*70)
    print("LECTOR DE PDFs - PRUEBA")
    print("="*70 + "\n")

    # Verificar si se proporcionó una carpeta
    if len(sys.argv) > 1:
        folder_path = sys.argv[1]
    else:
        folder_path = input("Ingresa la ruta de la carpeta con PDFs: ").strip()

    if not folder_path:
        print("No se proporcionó una carpeta. Usando carpeta actual.")
        folder_path = "."

    # Crear lector
    reader = PDFReader()

    # Leer PDFs
    try:
        documents = reader.read_pdf_folder(folder_path, recursive=False)

        if documents:
            print("\n" + "="*70)
            print("DOCUMENTOS LEÍDOS")
            print("="*70 + "\n")

            for i, doc in enumerate(documents, 1):
                print(f"{i}. {doc['metadata']['filename']}")
                print(f"   Tamaño: {doc['metadata']['size']:,} bytes")
                print(f"   Caracteres: {len(doc['text']):,}")

                # Mostrar primeros 200 caracteres
                preview = doc['text'][:200].replace('\n', ' ')
                print(f"   Vista previa: {preview}...")
                print()

    except FileNotFoundError as e:
        print(f"❌ Error: {e}")
    except Exception as e:
        print(f"❌ Error inesperado: {e}")


if __name__ == "__main__":
    main()
