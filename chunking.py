import json
import re

# --- CONFIGURATION ---
INPUT_FILE = "raw_data.json"
OUTPUT_FILE = "semantic_chunks.json"

def is_header(text):
    """
    Returns True if the text looks like a standard section header.
    """
    # Pattern 1: Numbered headers (1. Introduction, [1], 2.1 Analysis)
    if re.match(r'^\[?\d+\]?(\.\d+)*\.?\s', text):
        return True
    
    # Pattern 2: Short, emphatic text (e.g., "EXECUTIVE SUMMARY")
    if len(text) < 60 and (text.isupper() or text.istitle()):
        # Exclude common noise like dates or page numbers
        if not re.search(r'\d{4}', text) and not re.match(r'Page \d+', text): 
            return True
            
    return False

def extract_subject_name(text):
    """
    Specific logic to capture the course name from the syllabus file.
    Looks for: "SUBJECT: <NAME>"
    """
    match = re.search(r'SUBJECT\s*:\s*(.*)', text, re.IGNORECASE)
    if match:
        return match.group(1).strip()
    return None

def create_semantic_chunks(raw_blocks):
    chunks = []
    
    # State tracking
    current_subject = "General Introduction" # Default context
    
    current_chunk = {
        "section_title": "General",
        "content": "",
        "page_numbers": set(),
        "source": "",
        "subject_context": current_subject
    }
    
    for block in raw_blocks:
        text = block['text']
        page = block['metadata']['page']
        source = block['metadata']['source']
        
        # 1. CHECK FOR SUBJECT CHANGE (Global Context Switch)
        # If we see "SUBJECT: HARDWARE WORKSHOP", we switch the entire context.
        new_subject = extract_subject_name(text)
        if new_subject:
            # Save the previous chunk if it has content
            if current_chunk["content"].strip():
                current_chunk["page_numbers"] = list(current_chunk["page_numbers"])
                chunks.append(current_chunk)
            
            # Update the Global Subject
            current_subject = new_subject
            
            # Start a fresh chunk for the new subject
            current_chunk = {
                "section_title": "Course Introduction",
                "content": f"Subject: {current_subject}", # Start with the name
                "page_numbers": {page},
                "source": source,
                "subject_context": current_subject
            }
            continue # Skip appending the "SUBJECT: ..." line again to avoid duplicate noise

        # 2. CHECK FOR SECTION HEADERS (Local Section Switch)
        if is_header(text):
            # Save previous chunk
            if current_chunk["content"].strip():
                current_chunk["page_numbers"] = list(current_chunk["page_numbers"])
                chunks.append(current_chunk)
            
            # Start new chunk
            current_chunk = {
                "section_title": text,
                "content": "",
                "page_numbers": {page},
                "source": source,
                "subject_context": current_subject # Carry over the active subject
            }
        
        # 3. APPEND CONTENT
        # Just add the text normally. We will inject the context in the final step.
        current_chunk["content"] += " " + text
        current_chunk["page_numbers"].add(page)
        current_chunk["source"] = source
        # Update context just in case (e.g. for the very first block)
        current_chunk["subject_context"] = current_subject

    # Save the last chunk
    if current_chunk["content"].strip():
        current_chunk["page_numbers"] = list(current_chunk["page_numbers"])
        chunks.append(current_chunk)
        
    # --- POST-PROCESSING: INJECT CONTEXT ---
    # This is the "Secret Sauce" for Q8, Q9, Q10.
    # We prefix EVERY chunk with its subject name.
    for chunk in chunks:
        # If the text doesn't already start with the Subject name...
        if chunk["subject_context"] not in chunk["content"][:50]:
            # ...Prepend it!
            # Example: "Subject: HARDWARE WORKSHOP - [1] ELECTRONIC COMPONENTS..."
            chunk["content"] = f"Subject: {chunk['subject_context']} - {chunk['content']}"

    return chunks

if __name__ == "__main__":
    print("🧠 Starting Context-Aware Semantic Chunking...")
    
    try:
        with open(INPUT_FILE, "r", encoding="utf-8") as f:
            raw_data = json.load(f)
            
        final_chunks = create_semantic_chunks(raw_data)
        
        with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
            json.dump(final_chunks, f, indent=2, ensure_ascii=False)
            
        print(f"✅ Created {len(final_chunks)} chunks with Context Injection.")
        print(f"📂 Saved to '{OUTPUT_FILE}'")
        
        # Verify it worked
        print("\n🔍 Verification: Checking for injected subjects...")
        count = 0
        for chunk in final_chunks:
            if "Subject:" in chunk["content"]:
                count += 1
        print(f"   {count} chunks have 'Subject:' injected successfully.")
            
    except FileNotFoundError:
        print(f"❌ Error: Could not find '{INPUT_FILE}'.")