import re

def chunk_text(text, max_tokens=150, overlap=30):
    """
    Split text into overlapping chunks for better context preservation.
    Overlap ensures important context isn't lost at chunk boundaries.
    
    Args:
        text: Input text to chunk
        max_tokens: Maximum words per chunk
        overlap: Words to overlap between chunks
    """
    # Clean the text
    text = re.sub(r'\s+', ' ', text).strip()
    
    # Split into sentences
    sentences = re.split(r'(?<=[.!?])\s+', text)
    
    chunks = []
    current_chunk = []
    current_len = 0
    
    for sentence in sentences:
        sentence = sentence.strip()
        if not sentence:
            continue
        
        words = sentence.split()
        word_count = len(words)
        
        # If single sentence is too long, split it
        if word_count > max_tokens:
            # Flush current chunk first
            if current_chunk:
                chunks.append(' '.join(current_chunk))
                current_chunk = []
                current_len = 0
            
            # Split long sentence
            for i in range(0, len(words), max_tokens - overlap):
                part = ' '.join(words[i:i + max_tokens])
                if part.strip():
                    chunks.append(part.strip())
        
        # Check if adding this sentence exceeds limit
        elif current_len + word_count > max_tokens:
            # Save current chunk
            if current_chunk:
                chunks.append(' '.join(current_chunk))
            
            # Start new chunk with overlap from previous
            if overlap > 0 and current_chunk:
                overlap_words = ' '.join(current_chunk).split()[-overlap:]
                current_chunk = overlap_words + [sentence]
                current_len = len(overlap_words) + word_count
            else:
                current_chunk = [sentence]
                current_len = word_count
        else:
            current_chunk.append(sentence)
            current_len += word_count
    
    # Don't forget the last chunk
    if current_chunk:
        chunks.append(' '.join(current_chunk))
    
    # Filter out very short chunks
    chunks = [c.strip() for c in chunks if len(c.split()) >= 5]
    
    return chunks
