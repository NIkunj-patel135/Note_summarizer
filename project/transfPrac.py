from transformers import pipeline, AutoTokenizer
import torch
import re
import time

start_time = time.time()
device = 0 if torch.cuda.is_available() else -1

summarizer = pipeline("summarization", model="sshleifer/distilbart-cnn-12-6", device=device)

ARTICLE = """ Lorem ipsum
Lorem ipsum dolor sit amet, consectetur adipiscing
elit. Nunc ac faucibus odio.
Vestibulum neque massa, scelerisque sit amet ligula eu, congue molestie mi. Praesent ut
varius sem. Nullam at porttitor arcu, nec lacinia nisi. Ut ac dolor vitae odio interdum
condimentum. Vivamus dapibus sodales ex, vitae malesuada ipsum cursus
convallis. Maecenas sed egestas nulla, ac condimentum orci. Mauris diam felis,
vulputate ac suscipit et, iaculis non est. Curabitur semper arcu ac ligula semper, nec luctus
nisl blandit. Integer lacinia ante ac libero lobortis imperdiet. Nullam mollis convallis ipsum,
ac accumsan nunc vehicula vitae. Nulla eget justo in felis tristique fringilla. Morbi sit amet
tortor quis risus auctor condimentum. Morbi in ullamcorper elit. Nulla iaculis tellus sit amet
mauris tempus fringilla.
Maecenas mauris lectus, lobortis et purus mattis, blandit dictum tellus.
 Maecenas non lorem quis tellus placerat varius.
Lorem ipsum
Lorem ipsum dolor sit amet, consectetur adipiscing
elit. Nunc ac faucibus odio.
Vestibulum neque massa, scelerisque sit amet ligula eu, congue molestie mi. Praesent ut
varius sem. Nullam at porttitor arcu, nec lacinia nisi. Ut ac dolor vitae odio interdum
condimentum. Vivamus dapibus sodales ex, vitae malesuada ipsum cursus
convallis. Maecenas sed egestas nulla, ac condimentum orci. Mauris diam felis,
vulputate ac suscipit et, iaculis non est. Curabitur semper arcu ac ligula semper, nec luctus
nisl blandit. Integer lacinia ante ac libero lobortis imperdiet. Nullam mollis convallis ipsum,
ac accumsan nunc vehicula vitae. Nulla eget justo in felis tristique fringilla. Morbi sit amet
tortor quis risus auctor condimentum. Morbi in ullamcorper elit. Nulla iaculis tellus sit amet
mauris tempus fringilla.
Maecenas mauris lectus, lobortis et purus mattis, blandit dictum tellus.
 Maecenas non lorem quis tellus placerat varius.
Lorem ipsum
Lorem ipsum dolor sit amet, consectetur adipiscing
elit. Nunc ac faucibus odio.
Vestibulum neque massa, scelerisque sit amet ligula eu, congue molestie mi. Praesent ut
varius sem. Nullam at porttitor arcu, nec lacinia nisi. Ut ac dolor vitae odio interdum
condimentum. Vivamus dapibus sodales ex, vitae malesuada ipsum cursus
convallis. Maecenas sed egestas nulla, ac condimentum orci. Mauris diam felis,
vulputate ac suscipit et, iaculis non est. Curabitur semper arcu ac ligula semper, nec luctus
nisl blandit. Integer lacinia ante ac libero lobortis imperdiet. Nullam mollis convallis ipsum,
ac accumsan nunc vehicula vitae. Nulla eget justo in felis tristique fringilla. Morbi sit amet
tortor quis risus auctor condimentum. Morbi in ullamcorper elit. Nulla iaculis tellus sit amet
mauris tempus fringilla.
Maecenas mauris lectus, lobortis et purus mattis, blandit dictum tellus.
 Maecenas non lorem quis tellus placerat varius.
Lorem ipsum
Lorem ipsum dolor sit amet, consectetur adipiscing
elit. Nunc ac faucibus odio.
Vestibulum neque massa, scelerisque sit amet ligula eu, congue molestie mi. Praesent ut
varius sem. Nullam at porttitor arcu, nec lacinia nisi. Ut ac dolor vitae odio interdum
condimentum. Vivamus dapibus sodales ex, vitae malesuada ipsum cursus
convallis. Maecenas sed egestas nulla, ac condimentum orci. Mauris diam felis,
vulputate ac suscipit et, iaculis non est. Curabitur semper arcu ac ligula semper, nec luctus
nisl blandit. Integer lacinia ante ac libero lobortis imperdiet. Nullam mollis convallis ipsum,
ac accumsan nunc vehicula vitae. Nulla eget justo in felis tristique fringilla. Morbi sit amet
tortor quis risus auctor condimentum. Morbi in ullamcorper elit. Nulla iaculis tellus sit amet
mauris tempus fringilla.
Maecenas mauris lectus, lobortis et purus mattis, blandit dictum tellus.
 Maecenas non lorem quis tellus placerat varius.
Lorem ipsum
Lorem ipsum dolor sit amet, consectetur adipiscing
elit. Nunc ac faucibus odio.
Vestibulum neque massa, scelerisque sit amet ligula eu, congue molestie mi. Praesent ut
varius sem. Nullam at porttitor arcu, nec lacinia nisi. Ut ac dolor vitae odio interdum
condimentum. Vivamus dapibus sodales ex, vitae malesuada ipsum cursus
convallis. Maecenas sed egestas nulla, ac condimentum orci. Mauris diam felis,
vulputate ac suscipit et, iaculis non est. Curabitur semper arcu ac ligula semper, nec luctus
nisl blandit. Integer lacinia ante ac libero lobortis imperdiet. Nullam mollis convallis ipsum,
ac accumsan nunc vehicula vitae. Nulla eget justo in felis tristique fringilla. Morbi sit amet
tortor quis risus auctor condimentum. Morbi in ullamcorper elit. Nulla iaculis tellus sit amet
mauris tempus fringilla.
Maecenas mauris lectus, lobortis et purus mattis, blandit dictum tellus.
 Maecenas non lorem quis tellus placerat varius.
Lorem ipsum
Lorem ipsum dolor sit amet, consectetur adipiscing
elit. Nunc ac faucibus odio.
Vestibulum neque massa, scelerisque sit amet ligula eu, congue molestie mi. Praesent ut
varius sem. Nullam at porttitor arcu, nec lacinia nisi. Ut ac dolor vitae odio interdum
condimentum. Vivamus dapibus sodales ex, vitae malesuada ipsum cursus
convallis. Maecenas sed egestas nulla, ac condimentum orci. Mauris diam felis,
vulputate ac suscipit et, iaculis non est. Curabitur semper arcu ac ligula semper, nec luctus
nisl blandit. Integer lacinia ante ac libero lobortis imperdiet. Nullam mollis convallis ipsum,
ac accumsan nunc vehicula vitae. Nulla eget justo in felis tristique fringilla. Morbi sit amet
tortor quis risus auctor condimentum. Morbi in ullamcorper elit. Nulla iaculis tellus sit amet
mauris tempus fringilla.
Maecenas mauris lectus, lobortis et purus mattis, blandit dictum tellus.
 Maecenas non lorem quis tellus placerat varius.
Lorem ipsum
Lorem ipsum dolor sit amet, consectetur adipiscing
elit. Nunc ac faucibus odio.
Vestibulum neque massa, scelerisque sit amet ligula eu, congue molestie mi. Praesent ut
varius sem. Nullam at porttitor arcu, nec lacinia nisi. Ut ac dolor vitae odio interdum
condimentum. Vivamus dapibus sodales ex, vitae malesuada ipsum cursus
convallis. Maecenas sed egestas nulla, ac condimentum orci. Mauris diam felis,
vulputate ac suscipit et, iaculis non est. Curabitur semper arcu ac ligula semper, nec luctus
nisl blandit. Integer lacinia ante ac libero lobortis imperdiet. Nullam mollis convallis ipsum,
ac accumsan nunc vehicula vitae. Nulla eget justo in felis tristique fringilla. Morbi sit amet
tortor quis risus auctor condimentum. Morbi in ullamcorper elit. Nulla iaculis tellus sit amet
mauris tempus fringilla.
Maecenas mauris lectus, lobortis et purus mattis, blandit dictum tellus.
 Maecenas non lorem quis tellus placerat varius.
Lorem ipsum
Lorem ipsum dolor sit amet, consectetur adipiscing
elit. Nunc ac faucibus odio.
Vestibulum neque massa, scelerisque sit amet ligula eu, congue molestie mi. Praesent ut
varius sem. Nullam at porttitor arcu, nec lacinia nisi. Ut ac dolor vitae odio interdum
condimentum. Vivamus dapibus sodales ex, vitae malesuada ipsum cursus
convallis. Maecenas sed egestas nulla, ac condimentum orci. Mauris diam felis,
vulputate ac suscipit et, iaculis non est. Curabitur semper arcu ac ligula semper, nec luctus
nisl blandit. Integer lacinia ante ac libero lobortis imperdiet. Nullam mollis convallis ipsum,
ac accumsan nunc vehicula vitae. Nulla eget justo in felis tristique fringilla. Morbi sit amet
tortor quis risus auctor condimentum. Morbi in ullamcorper elit. Nulla iaculis tellus sit amet
mauris tempus fringilla.
Maecenas mauris lectus, lobortis et purus mattis, blandit dictum tellus.
 Maecenas non lorem quis tellus placerat varius.
Lorem ipsum
Lorem ipsum dolor sit amet, consectetur adipiscing
elit. Nunc ac faucibus odio.
Vestibulum neque massa, scelerisque sit amet ligula eu, congue molestie mi. Praesent ut
varius sem. Nullam at porttitor arcu, nec lacinia nisi. Ut ac dolor vitae odio interdum
condimentum. Vivamus dapibus sodales ex, vitae malesuada ipsum cursus
convallis. Maecenas sed egestas nulla, ac condimentum orci. Mauris diam felis,
vulputate ac suscipit et, iaculis non est. Curabitur semper arcu ac ligula semper, nec luctus
nisl blandit. Integer lacinia ante ac libero lobortis imperdiet. Nullam mollis convallis ipsum,
ac accumsan nunc vehicula vitae. Nulla eget justo in felis tristique fringilla. Morbi sit amet
tortor quis risus auctor condimentum. Morbi in ullamcorper elit. Nulla iaculis tellus sit amet
mauris tempus fringilla.
Maecenas mauris lectus, lobortis et purus mattis, blandit dictum tellus.
 Maecenas non lorem quis tellus placerat varius.
Lorem ipsum
Lorem ipsum dolor sit amet, consectetur adipiscing
elit. Nunc ac faucibus odio.
Vestibulum neque massa, scelerisque sit amet ligula eu, congue molestie mi. Praesent ut
varius sem. Nullam at porttitor arcu, nec lacinia nisi. Ut ac dolor vitae odio interdum
condimentum. Vivamus dapibus sodales ex, vitae malesuada ipsum cursus
convallis. Maecenas sed egestas nulla, ac condimentum orci. Mauris diam felis,
vulputate ac suscipit et, iaculis non est. Curabitur semper arcu ac ligula semper, nec luctus
nisl blandit. Integer lacinia ante ac libero lobortis imperdiet. Nullam mollis convallis ipsum,
ac accumsan nunc vehicula vitae. Nulla eget justo in felis tristique fringilla. Morbi sit amet
tortor quis risus auctor condimentum. Morbi in ullamcorper elit. Nulla iaculis tellus sit amet
mauris tempus fringilla.
Maecenas mauris lectus, lobortis et purus mattis, blandit dictum tellus.
 Maecenas non lorem quis tellus placerat varius.

"""

tokenizer = AutoTokenizer.from_pretrained("sshleifer/distilbart-cnn-12-6")


def chunk_text(text, max_tokens=900, overlap=50):  # max_tokens for safety margin
   
    tokens = tokenizer.encode(text, add_special_tokens=False)
    chunks = []
    start = 0

    while start < len(tokens):
        # Calculate end position
        end = min(start + max_tokens, len(tokens))
        
        # Extract chunk tokens
        chunk_tokens = tokens[start:end]
        
        # Decode to text
        chunk_text = tokenizer.decode(chunk_tokens, skip_special_tokens=True)
        
        # Verify the chunk doesn't exceed limits when re-tokenized
        # (Sometimes decoding and re-encoding can change token count slightly)
        verification_tokens = tokenizer.encode(chunk_text, add_special_tokens=True)
        if len(verification_tokens) > 1024:
            # If it exceeds, reduce the chunk size
            reduced_end = start + max_tokens - 100  # Reduce by 100 tokens
            chunk_tokens = tokens[start:reduced_end]
            chunk_text = tokenizer.decode(chunk_tokens, skip_special_tokens=True)
        
        chunks.append(chunk_text)
        
        # Move start position with overlap
        start = end - overlap
        
        # Ensure we don't get stuck in infinite loop
        if start >= end - overlap:
            start = end

    return chunks


def summarize(chunks):
    final_summaries = []

    for i, chunk in enumerate(chunks):
        try:
            # Calculate token count for this chunk
            num_tokens = len(tokenizer.encode(chunk, add_special_tokens=True))
            print(f"Chunk {i+1}: {num_tokens} tokens")
            
            # # Ensure chunk is within model limits
            # if num_tokens > 1024:
            #     print(f"Warning: Chunk {i+1} has {num_tokens} tokens, skipping...")
            #     continue
            
            # Calculate appropriate summary lengths
            max_len = min(int(num_tokens * 0.3), 142)  # DistilBART max summary length is ~142
            min_len = max(10, int(max_len * 0.5))  # Ensure reasonable minimum
            
            print(f"Summarizing chunk {i+1} with max_len={max_len}, min_len={min_len}")
            
            summary = summarizer(chunk, max_length=max_len, min_length=min_len, do_sample=False)
            final_summaries.append(summary[0]['summary_text'])
            
        except Exception as e:
            print(f"Error processing chunk {i+1}: {e}")
            continue

    return " ".join(final_summaries)


def clean_text(text: str) -> str:
    if not text:
        return ""

    # Normalize spaces (replace non-breaking spaces etc.)
    text = text.replace("\xa0", " ").replace("\u200b", " ")

    # Remove bullet points and special symbols
    text = re.sub(r"[•·●▪◆■●★▶►‐–—−●]", " ", text)

    # Remove multiple hyphens/underscores (page breaks or separators)
    text = re.sub(r"[-_]{2,}", " ", text)

    # Remove extra newlines (convert to space)
    text = re.sub(r"\n+", " ", text)

    # Remove isolated numbers (page numbers, references)
    text = re.sub(r"\b\d+\b", " ", text)

    # Remove weird multiple punctuation (e.g., "!!!", "...", "??")
    text = re.sub(r"([!?.,]){2,}", r"\1", text)

    # Collapse multiple spaces
    text = re.sub(r"\s{2,}", " ", text).strip()

    return text


# Main execution
cleaned_text = clean_text(ARTICLE)
print(f"Original text length: {len(tokenizer.encode(cleaned_text))} tokens")

chunks = chunk_text(cleaned_text, max_tokens=900, overlap=50)
print(f"Created {len(chunks)} chunks")

summary = summarize(chunks)
print("\nFinal Summary:")
print(summary)

end_time = time.time()
elapsed_time = end_time - start_time
print(f"\nTime taken: {elapsed_time:.4f} seconds")