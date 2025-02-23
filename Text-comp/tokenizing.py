from transformers import AutoTokenizer
import orjson  # for speed
from tqdm import tqdm
from multiprocessing import Pool

input_file = "filtered-ass.jsonl"
output_file = "tokenized-ass.jsonl"
model_name = "microsoft/phi-4"  # Change this to whatever HF model you're using
max_tokens = 16384


# Load your tokenizer only once for each worker
def init_worker():
    global tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)


def process_line(line):
    try:
        record = orjson.loads(line)
        content = record.get("content", "")

        if not content:  # Skip entries with blank content
            return None

        # Tokenize and check length
        token_count = len(tokenizer.encode(content, add_special_tokens=False))
        if token_count <= max_tokens:
            return orjson.dumps(record).decode("utf-8")
    except Exception:
        return None  # Skip problematic entries


def main():
    with open(input_file, "r") as infile:
        lines = infile.readlines()

    num_workers = 12  # Use all those 12 cores you're so proud of
    with Pool(num_workers, initializer=init_worker) as pool:
        results = list(
            tqdm(
                pool.imap(process_line, lines),
                desc="Filtering based on token limit",
                total=len(lines),
            )
        )

    with open(output_file, "w") as outfile:
        for result in results:
            if result:
                outfile.write(result + "\n")


if __name__ == "__main__":
    main()