import json
import re
import sys
from pathlib import Path

def is_valid_bhojpuri_sentence(text: str) -> bool:
    """
    Applies strict heuristics to filter out Wikipedia stubs, 
    useless fragments, and scraping artifacts.
    """
    text = text.strip()
    
    # Rule 1: Drop extremely short or artificially long sentences
    if len(text) < 15 or len(text) > 400:
        return False
        
    # Rule 2: Drop Wikipedia calendar/date stubs and generic lists
    wiki_stub_patterns = [
        r"ग्रेगरियन कैलेंडर",
        r"जनवरी-मार्च",
        r"अप्रैल-जून",
        r"जुलाई-सितंबर",
        r"अक्टूबर-दिसंबर",
        r"अज्ञात तिथि",
        r"निधन तिहुआर",
        r"छुट्टी अउरी खास महत्व"
    ]
    for pattern in wiki_stub_patterns:
        if re.search(pattern, text):
            return False
            
    # Rule 3: Drop strings that are purely English/Gibberish (e.g., "व्हत् थे हेल्ल् इस् थिस्")
    # If the text has an unusually high density of English characters, drop it.
    english_chars = len(re.findall(r'[a-zA-Z]', text))
    if english_chars > (len(text) * 0.3):
        return False
        
    # Rule 4: Drop sentences starting with orphaned punctuation like ") "
    if re.match(r'^[\)\}\]\.\,]', text):
        return False

    return True

def process_corpus(input_path: str, output_path: str):
    print(f"Loading corpus from {input_path}...")
    
    total_rows = 0
    kept_rows = 0
    
    with open(input_path, 'r', encoding='utf-8') as infile, \
         open(output_path, 'w', encoding='utf-8') as outfile:
         
        for line in infile:
            total_rows += 1
            try:
                row = json.loads(line)
                text = row.get("cleaned_text", "")
                
                if is_valid_bhojpuri_sentence(text):
                    outfile.write(json.dumps(row, ensure_ascii=False) + '\n')
                    kept_rows += 1
                    
            except json.JSONDecodeError:
                print(f"Skipping malformed JSON on line {total_rows}")
                continue

    dropped_rows = total_rows - kept_rows
    print("-" * 30)
    print("FILTERING COMPLETE")
    print(f"Total Rows Processed: {total_rows}")
    print(f"Rows Kept:            {kept_rows} ({(kept_rows/total_rows)*100:.2f}%)")
    print(f"Rows Dropped:         {dropped_rows} ({(dropped_rows/total_rows)*100:.2f}%)")
    print("-" * 30)

if __name__ == "__main__":
    # Point this to where your corpus_clean.jsonl is located
    INPUT_FILE = "corpus_clean.jsonl"
    OUTPUT_FILE = "corpus_filtered.jsonl"
    
    if not Path(INPUT_FILE).exists():
        print(f"Error: {INPUT_FILE} not found. Please check your path.")
        sys.exit(1)
        
    process_corpus(INPUT_FILE, OUTPUT_FILE)