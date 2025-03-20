# all the needed pipinstallations:
!pip uninstall weaviate-client -y
!pip install weaviate-client==4.4.0
# Core dependencies
pip install weaviate PyPDF2 pdfplumber tqdm

# For table extraction (optional)
pip install camelot-py[cv] opencv-python-headless ghostscript

# For embedding (if not using Weaviate's built-in)
pip install sentence-transformers

# For LLM integration (optional)
pip install openai

"""
Complete RAG Workshop Example: Technical Specs Q&A with Weaviate
This script demonstrates a full RAG pipeline with Weaviate Embedded (free option)
"""

import weaviate
from weaviate.embedded import EmbeddedOptions
import PyPDF2
import pdfplumber
import os
import re
from typing import Dict, List, Tuple, Optional
import json
from tqdm import tqdm

# Try to import optional dependencies
try:
    import camelot
    CAMELOT_AVAILABLE = True
except ImportError:
    CAMELOT_AVAILABLE = False
    print("Camelot not available. Table extraction will be limited.")

try:
    from openai import OpenAI
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False
    print("OpenAI not available. Using placeholder LLM function.")

# ================= STEP 1: PDF PROCESSING FUNCTIONS =================

def extract_text_from_pdf(pdf_path: str) -> Dict[int, str]:
    """Extract text content page by page from a PDF."""
    text_by_page = {}
    
    # Using PyPDF2 for basic text extraction
    with open(pdf_path, 'rb') as file:
        reader = PyPDF2.PdfReader(file)
        for page_num in range(len(reader.pages)):
            page = reader.pages[page_num]
            text_by_page[page_num] = page.extract_text()
    
    # If PyPDF2 fails to extract good text, try pdfplumber
    if any(not text for text in text_by_page.values()):
        with pdfplumber.open(pdf_path) as pdf:
            for page_num, page in enumerate(pdf.pages):
                if not text_by_page.get(page_num, "").strip():
                    text_by_page[page_num] = page.extract_text() or ""
    
    return text_by_page

def extract_tables_from_pdf(pdf_path: str) -> Dict[int, List[str]]:
    """Extract tables page by page from PDF."""
    tables_by_page = {}
    
    if not CAMELOT_AVAILABLE:
        print("Skipping table extraction - camelot not available")
        return tables_by_page
    
    try:
        # Camelot works well for tables with clear borders
        tables = camelot.read_pdf(pdf_path, pages='all', flavor='lattice')
        
        for table in tables:
            page_num = table.page - 1  # Camelot uses 1-indexed pages
            if page_num not in tables_by_page:
                tables_by_page[page_num] = []
            
            # Convert table to string representation
            table_str = table.df.to_csv(index=False)
            tables_by_page[page_num].append(table_str)
    except Exception as e:
        print(f"Warning: Table extraction failed with error: {e}")
    
    return tables_by_page

def detect_diagrams(pdf_path: str) -> Dict[int, List[Tuple[int, int, int, int]]]:
    """
    Detect potential diagrams based on image density and text sparsity.
    Returns dictionary with page numbers as keys and lists of bounding boxes.
    """
    diagram_regions = {}
    
    try:
        with pdfplumber.open(pdf_path) as pdf:
            for page_num, page in enumerate(pdf.pages):
                # Extract images
                images = page.images
                
                # If there are images, consider them potential diagrams
                if images:
                    diagram_regions[page_num] = []
                    for img in images:
                        # Get bounding box
                        x0, y0, x1, y1 = img['x0'], img['top'], img['x1'], img['bottom']
                        diagram_regions[page_num].append((x0, y0, x1, y1))
    except Exception as e:
        print(f"Warning: Diagram detection failed with error: {e}")
    
    return diagram_regions

def chunk_text(text: str, chunk_size: int = 1000, overlap: int = 200) -> List[str]:
    """Split text into overlapping chunks of approximately chunk_size characters."""
    if not text or chunk_size <= 0:
        return []
    
    # Split by natural breaks first
    paragraphs = [p for p in re.split(r'\n{2,}', text) if p.strip()]
    
    chunks = []
    current_chunk = []
    current_size = 0
    
    for paragraph in paragraphs:
        # If adding this paragraph exceeds chunk size and we already have content
        if current_size + len(paragraph) > chunk_size and current_chunk:
            # Join the accumulated paragraphs and add to chunks
            chunks.append(' '.join(current_chunk))
            
            # Start a new chunk with overlap
            overlap_size = 0
            overlap_chunk = []
            
            # Add paragraphs from the end until we reach desired overlap
            for p in reversed(current_chunk):
                overlap_chunk.insert(0, p)
                overlap_size += len(p)
                if overlap_size >= overlap:
                    break
            
            current_chunk = overlap_chunk
            current_size = sum(len(p) for p in current_chunk)
        
        # Add the current paragraph
        current_chunk.append(paragraph)
        current_size += len(paragraph)
    
    # Add the last chunk if not empty
    if current_chunk:
        chunks.append(' '.join(current_chunk))
    
    return chunks

def process_pdf(pdf_path: str, chunk_size: int = 1000, overlap: int = 200) -> List[Dict]:
    """
    Process a PDF file to extract text, tables, and diagram locations.
    Returns a list of chunk documents with metadata.
    """
    # Extract content
    text_by_page = extract_text_from_pdf(pdf_path)
    tables_by_page = extract_tables_from_pdf(pdf_path)
    diagrams_by_page = detect_diagrams(pdf_path)
    
    # Create document chunks
    doc_chunks = []
    
    # Process text and tables by page
    all_pages = sorted(set(list(text_by_page.keys()) + 
                          list(tables_by_page.keys()) + 
                          list(diagrams_by_page.keys())))
    
    for page_num in all_pages:
        # Get page content
        page_text = text_by_page.get(page_num, "")
        page_tables = tables_by_page.get(page_num, [])
        page_diagrams = diagrams_by_page.get(page_num, [])
        
        # Process text chunks
        if page_text:
            text_chunks = chunk_text(page_text, chunk_size, overlap)
            for i, chunk in enumerate(text_chunks):
                doc_chunks.append({
                    "content": chunk,
                    "content_type": "text",
                    "page_num": page_num,
                    "chunk_num": i,
                    "source": os.path.basename(pdf_path)
                })
        
        # Process tables
        for i, table in enumerate(page_tables):
            doc_chunks.append({
                "content": table,
                "content_type": "table",
                "page_num": page_num,
                "chunk_num": i,
                "source": os.path.basename(pdf_path)
            })
        
        # Process diagrams (just record their presence)
        if page_diagrams:
            doc_chunks.append({
                "content": f"This page contains {len(page_diagrams)} diagram(s) or images.",
                "content_type": "diagram_reference",
                "page_num": page_num,
                "source": os.path.basename(pdf_path)
            })
    
    return doc_chunks

# ================= STEP 2: WEAVIATE FUNCTIONS =================

def get_weaviate_client():
    """Get a Weaviate client using the embedded option (completely free)."""
    print("Starting embedded Weaviate instance...")
    client = weaviate.Client(
        embedded_options=EmbeddedOptions()
    )
    return client

def create_tech_spec_schema(client):
    """Create the TechSpec schema in Weaviate if it doesn't exist."""
    # Check if class already exists
    try:
        client.schema.get("TechSpec")
        print("TechSpec class already exists")
        return
    except Exception as e:
        pass  # Create it
    
    # Define the schema
    class_obj = {
        "class": "TechSpec",
        "vectorizer": "text2vec-transformers",  # Using the built-in vectorizer
        "moduleConfig": {
            "text2vec-transformers": {
                "vectorizeClassName": False
            }
        },
        "properties": [
            {
                "name": "content",
                "dataType": ["text"],
                "description": "The actual content of the document chunk",
                "moduleConfig": {
                    "text2vec-transformers": {
                        "skip": False,
                        "vectorizePropertyName": False
                    }
                }
            },
            {
                "name": "content_type",
                "dataType": ["text"],
                "description": "Type of content: text, table, diagram_reference",
                "moduleConfig": {
                    "text2vec-transformers": {
                        "skip": True  # Don't include in vector calculation
                    }
                }
            },
            {
                "name": "page_num",
                "dataType": ["int"],
                "description": "Page number in the source document",
                "moduleConfig": {
                    "text2vec-transformers": {
                        "skip": True
                    }
                }
            },
            {
                "name": "chunk_num",
                "dataType": ["int"],
                "description": "Chunk number within the page",
                "moduleConfig": {
                    "text2vec-transformers": {
                        "skip": True
                    }
                }
            },
            {
                "name": "source",
                "dataType": ["text"],
                "description": "Source document filename",
                "moduleConfig": {
                    "text2vec-transformers": {
                        "skip": True
                    }
                }
            }
        ]
    }
    
    # Create the class
    client.schema.create_class(class_obj)
    print("Created TechSpec class in Weaviate")

def import_chunks_to_weaviate(client, chunks):
    """Import processed document chunks into Weaviate."""
    # Get a batch configuration
    batch_size = 100
    batch = client.batch.configure(batch_size=batch_size)
    
    print(f"Importing {len(chunks)} chunks to Weaviate...")
    
    # Import chunks in batch
    with batch:
        for i, chunk in enumerate(tqdm(chunks)):
            # Add to batch
            client.batch.add_data_object(
                data_object=chunk,
                class_name="TechSpec"
            )
    
    print("Completed importing chunks to Weaviate")

def process_and_import_pdf(pdf_path, client):
    """Process a PDF and import it to Weaviate."""
    # Ensure schema exists
    create_tech_spec_schema(client)
    
    # Process the PDF
    print(f"Processing PDF: {pdf_path}")
    chunks = process_pdf(pdf_path)
    print(f"Generated {len(chunks)} chunks")
    
    # Import to Weaviate
    import_chunks_to_weaviate(client, chunks)
    print("Import complete!")
    
    return len(chunks)

# ================= STEP 3: QUERY FUNCTIONS =================

def query_tech_specs(client, query_text, limit=5):
    """
    Perform a basic vector search in Weaviate.
    """
    result = (
        client.query
        .get("TechSpec", ["content", "content_type", "page_num", "source"])
        .with_near_text({"concepts": [query_text]})
        .with_limit(limit)
        .do()
    )
    
    if "data" in result and "Get" in result["data"] and "TechSpec" in result["data"]["Get"]:
        return result["data"]["Get"]["TechSpec"]
    return []

def hybrid_query_tech_specs(client, query_text, limit=5):
    """
    Perform a hybrid search combining vector similarity and BM25 keyword search.
    """
    result = (
        client.query
        .get("TechSpec", ["content", "content_type", "page_num", "source"])
        .with_hybrid(
            query=query_text,
            properties=["content"],
            alpha=0.5  # Balance between vector (0) and keyword (1)
        )
        .with_limit(limit)
        .do()
    )
    
    if "data" in result and "Get" in result["data"] and "TechSpec" in result["data"]["Get"]:
        return result["data"]["Get"]["TechSpec"]
    return []

def filtered_query_tech_specs(client, query_text, content_type=None, source=None, limit=5):
    """
    Query tech specs with optional filters for content type and source.
    """
    # Start building the query
    query_builder = (
        client.query
        .get("TechSpec", ["content", "content_type", "page_num", "source"])
        .with_near_text({"concepts": [query_text]})
        .with_limit(limit)
    )
    
    # Add filters if specified
    where_filter = {}
    
    if content_type:
        where_filter["path"] = ["content_type"]
        where_filter["operator"] = "Equal"
        where_filter["valueText"] = content_type
        
        # Apply the filter
        query_builder = query_builder.with_where(where_filter)
    
    if source:
        where_filter = {
            "path": ["source"],
            "operator": "Equal",
            "valueText": source
        }
        # Apply the filter
        query_builder = query_builder.with_where(where_filter)
    
    # Execute the query
    result = query_builder.do()
    
    if "data" in result and "Get" in result["data"] and "TechSpec" in result["data"]["Get"]:
        return result["data"]["Get"]["TechSpec"]
    return []

# ================= STEP 4: LLM INTEGRATION =================

def generate_answer(query, retrieved_chunks):
    """
    Generate an answer to a query using retrieved chunks and an LLM.
    Falls back to a simple summarization if OpenAI is not available.
    """
    if not OPENAI_AVAILABLE:
        # Fallback simple answer without LLM
        answer = f"Based on the retrieved information, I found {len(retrieved_chunks)} relevant sections."
        answer += "\n\nKey information includes:\n"
        for i, chunk in enumerate(retrieved_chunks[:3]):
            answer += f"- From page {chunk['page_num']}: {chunk['content'][:150]}...\n"
        return answer
    
    # If OpenAI is available:
    # Set your API key in environment variable or directly here
    client = OpenAI()
    
    # Combine the retrieved chunks
    context = "\n\n".join([f"CHUNK {i+1} (Page {chunk['page_num']}, {chunk['content_type']}):\n{chunk['content']}" 
                         for i, chunk in enumerate(retrieved_chunks)])
    
    # Create the prompt
    prompt = f"""
    You are an expert assistant for technical specifications. 
    
    CONTEXT INFORMATION:
    {context}
    
    QUESTION: {query}
    
    Based only on the context information provided, answer the question.
    If the context doesn't contain the answer, say "I don't have enough information to answer this question."
    Cite the page numbers where you found relevant information.
    
    ANSWER:
    """
    
    # Get response from LLM
    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": "You are an expert technical documentation assistant."},
            {"role": "user", "content": prompt}
        ],
        temperature=0.0
    )
    
    return response.choices[0].message.content.strip()

# ================= STEP 5: COMPLETE RAG PIPELINE =================

def answer_tech_spec_question(client, query, limit=5):
    """
    End-to-end RAG pipeline for answering technical specification questions.
    """
    # Step 1: Retrieve relevant chunks
    print(f"Query: {query}")
    retrieved_chunks = hybrid_query_tech_specs(client, query, limit=limit)
    
    if not retrieved_chunks:
        return {
            "answer": "No relevant information found in the technical specifications.",
            "sources": []
        }
    
    # Step 2: Generate answer
    print("Generating answer...")
    answer = generate_answer(query, retrieved_chunks)
    
    # Step 3: Format the response
    response = {
        "answer": answer,
        "sources": [
            {
                "source": chunk["source"],
                "page": chunk["page_num"],
                "content_type": chunk["content_type"]
            } for chunk in retrieved_chunks
        ]
    }
    
    return response

# ================= MAIN EXECUTION =================

def main():
    """Main workshop execution."""
    print("Welcome to the RAG Workshop for Technical Specs Q&A with Weaviate!")
    print("=" * 80)
    
    # Check for PDF path
    pdf_path = input("Enter the path to your technical spec PDF (or press Enter for demo): ")
    if not pdf_path:
        pdf_path = "LEXI-R4-SARA-R4_ATCommands_UBX-17003787.pdf"
        print(f"Using demo file: {pdf_path}")
        # In a real workshop, you'd provide a sample file for users
        
    # Initialize Weaviate client
    client = get_weaviate_client()
    
    # Process and import PDF
    process_and_import_pdf(pdf_path, client)
    
    # Interactive query loop
    print("\nYour technical spec is now ready for questions!")
    print("Type 'exit' to quit.\n")
    
    while True:
        query = input("\nGenerate an AT command sequence that will attach the device to an LTE network using eDRX with 81seconds cycle interval, periodically send 100 bytes of data using HTTPs, and immediately release the connection using RAI ")
        if query.lower() in ('exit', 'quit', 'q'):
            break
            
        # Get answer
        response = answer_tech_spec_question(client, query)
        
        # Display answer
        print("\n" + "=" * 40)
        print("ANSWER:")
        print(response["answer"])
        print("\nSOURCES:")
        for src in response["sources"]:
            print(f"- {src['source']}, Page {src['page']}, Type: {src['content_type']}")
        print("=" * 40)
    
    print("\nThank you for using the Technical Spec RAG Workshop!")

if __name__ == "__main__":
    main()
Last edited 20 hours ago