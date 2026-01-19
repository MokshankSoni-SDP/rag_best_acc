import os
import json
from pathlib import Path
from unstructured.cleaners.core import clean, clean_non_ascii_chars

# --- CONFIGURATION ---
INPUT_FILE = "INFORMATION TECHNOLOGY.pdf"  # Update this to your file
OUTPUT_FILE = "raw_data.json"
# ---------------------

def load_and_structure_file(file_path):
    file_path = Path(file_path)
    file_ext = file_path.suffix.lower()
    
    print(f"🔄 Processing: {file_path.name}...")
    
    elements = []
    try:
        if file_ext == ".pdf":
            from unstructured.partition.pdf import partition_pdf
            print("   👉 Using PDF Partitioner (hi_res + Tables)...")
            # CRITICAL UPDATE: infer_table_structure=True forces it to look inside tables
            elements = partition_pdf(
                filename=str(file_path), 
                strategy="hi_res", 
                infer_table_structure=True,
                chunking_strategy="by_title" # This helps keep related text together
            )
        elif file_ext == ".txt":
            from unstructured.partition.text import partition_text
            elements = partition_text(filename=str(file_path))
        elif file_ext == ".docx":
            from unstructured.partition.docx import partition_docx
            elements = partition_docx(filename=str(file_path))
        else:
            print(f"❌ Unsupported format: {file_ext}")
            return []

    except Exception as e:
        print(f"❌ Error during partition: {e}")
        return []

    print(f"   📊 Raw elements found: {len(elements)}")

    structured_blocks = []
    for element in elements:
        # Safety check
        if not hasattr(element, "text") or not element.text: 
            continue

        # 1. STOP REMOVING HEADERS
        # In this PDF, "Semester 3" or "Subject Name" might be a Header. We need them.
        # We only skip 'Footer' (page numbers) to avoid noise.
        if element.category == "Footer": 
            continue 

        # 2. LIGHTER CLEANING
        # Don't lowercase or remove bullets aggressively
        text = clean(element.text, extra_whitespace=True, dashes=True, bullets=False)
        text = clean_non_ascii_chars(text)

        # 3. STOP REMOVING SHORT TEXT
        # Syllabus codes like "CO1" or "Unit 1" are short but vital.
        if len(text) < 2: 
            continue

        # 4. HANDLE TABLES
        # If it's a table, we want to flag it or just treat it as text
        elem_type = element.category
        if elem_type == "Table":
            text = "[TABLE DATA] " + text

        # Save Safe Metadata
        page_num = getattr(element.metadata, "page_number", 1) if hasattr(element, "metadata") else 1

        block = {
            "type": elem_type,
            "text": text,
            "metadata": {"source": file_path.name, "page": page_num}
        }
        structured_blocks.append(block)

    return structured_blocks

if __name__ == "__main__":
    if os.path.exists(INPUT_FILE):
        blocks = load_and_structure_file(INPUT_FILE)
        
        if blocks:
            with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
                json.dump(blocks, f, indent=2, ensure_ascii=False)
            print(f"✅ Success! Saved {len(blocks)} raw blocks to '{OUTPUT_FILE}'")
        else:
            print("❌ No text extracted.")
    else:
        print(f"❌ File '{INPUT_FILE}' not found.")