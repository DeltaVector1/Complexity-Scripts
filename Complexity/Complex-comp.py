import json
import argparse
import spacy
from tqdm import tqdm
from pathlib import Path
from multiprocessing import Pool, cpu_count

MAX_PARTICIPLES = 2
MAX_NESTED_CLAUSES = 2
COMPLEXITY_THRESHOLD = 128  # Adjust as needed
ANALYZE_FIELD = "text"

nlp = spacy.load("en_core_web_sm", disable=["ner", "textcat"])

def analyze_complexity(text):
    doc = nlp(text)
    analysis = {
        "has_passive_voice": False,
        "max_nested_clauses": 0,
        "longest_sentence": 0,
        "num_participle_phrases": 0,
        "num_sentences": len(list(doc.sents)),
        "is_complex": False,
        "reasons": [],
        "score": 0,
    }

    for sent in doc.sents:
        for token in sent:
            if token.dep_ in ["nsubjpass", "auxpass"]:
                if not analysis["has_passive_voice"]:
                    analysis["reasons"].append("Contains passive voice")
                    analysis["has_passive_voice"] = True
                    analysis["score"] += 1
                break

        sent_participles = sum(1 for token in sent if token.tag_ in ["VBG", "VBN"] and token.dep_ in ["ROOT", "advcl", "acl"])
        sent_clauses = sum(1 for token in sent if token.dep_ in ["ccomp", "xcomp", "advcl", "relcl"])

        analysis["max_nested_clauses"] = max(analysis["max_nested_clauses"], sent_clauses)
        analysis["longest_sentence"] = max(analysis["longest_sentence"], len(sent))
        analysis["num_participle_phrases"] = max(analysis["num_participle_phrases"], sent_participles)

        if sent_participles > MAX_PARTICIPLES:
            analysis["reasons"].append(f"Too many participle phrases ({sent_participles}) in one sentence")
            analysis["score"] += (sent_participles - MAX_PARTICIPLES)

        if sent_clauses > MAX_NESTED_CLAUSES:
            analysis["reasons"].append(f"Deeply nested clauses ({sent_clauses}) in one sentence")
            analysis["score"] += (sent_clauses - MAX_NESTED_CLAUSES)

    analysis["is_complex"] = analysis["score"] > 0
    return analysis

def process_line(args):
    """ Process a single JSONL entry """
    line, filter_out_complex = args
    obj = json.loads(line)
    
    if ANALYZE_FIELD in obj and obj[ANALYZE_FIELD]:
        obj["complexity_analysis"] = analyze_complexity(obj[ANALYZE_FIELD])
        complexity_score = obj["complexity_analysis"]["score"]
    else:
        obj["complexity_analysis"] = {
            "is_complex": False,
            "reasons": [f"MISSING FIELD {ANALYZE_FIELD}"],
            "has_passive_voice": False,
            "max_nested_clauses": 0,
            "longest_sentence": 0,
            "num_participle_phrases": 0,
            "score": 0
        }
        complexity_score = 0

    obj["total_complexity_score"] = complexity_score
    obj["is_complex"] = complexity_score > COMPLEXITY_THRESHOLD

    if not filter_out_complex or complexity_score <= COMPLEXITY_THRESHOLD:
        return json.dumps(obj, ensure_ascii=False)
    
    return None  # Skip saving this log (filtered out due to high complexity)

def process_jsonl(input_file, output_file, filter_out_complex, num_workers):
    input_path = Path(input_file)

    with input_path.open('r') as f_in, open(output_file, 'w') as f_out:
        total_lines = sum(1 for _ in f_in)
        f_in.seek(0)

        with Pool(num_workers) as pool:
            with tqdm(total=total_lines, desc="Processing logs") as pbar:
                for result in pool.imap_unordered(process_line, ((line, filter_out_complex) for line in f_in), chunksize=10):
                    if result:
                        f_out.write(result + '\n')
                    pbar.update(1)

def main():
    parser = argparse.ArgumentParser(description=f"Analyze and filter {ANALYZE_FIELD} field in JSONL file.")
    parser.add_argument('input_file', type=str, help='Input JSONL file path')
    parser.add_argument('-f', '--filter', action='store_true', help="Filter out logs exceeding complexity threshold.")
    parser.add_argument('-w', '--workers', type=int, default=cpu_count(), help="Number of worker processes (default: max CPU cores)")
    args = parser.parse_args()

    input_path = Path(args.input_file)
    output_path = input_path.with_stem(f"{input_path.stem}_enriched")

    if not input_path.exists():
        raise RuntimeError(f"File '{input_path}' does not exist")

    print(f"Processing '{input_path}' with {args.workers} workers (Output: '{output_path}')\n")
    process_jsonl(input_path, output_path, args.filter, args.workers)
    print("DONE")

if __name__ == "__main__":
    main()