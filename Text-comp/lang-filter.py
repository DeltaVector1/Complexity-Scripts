from langdetect import detect
import json
from tqdm import tqdm
from multiprocessing import Pool

input_file = "ass-pruned.jsonl"
output_file = "filtered-ass.jsonl"

def process_line(line):
    try:
        record = json.loads(line)
        text = record.get("content", "")
        if detect(text) == "en":  # Keep only English
            return json.dumps(record)
    except Exception:
        # If detection fails, skip the line
        return None


def main():
    with open(input_file, "r") as infile:
        lines = infile.readlines()

    # Use 8 workers, happy now?
    num_workers = 8
    with Pool(num_workers) as pool:
        results = list(
            tqdm(pool.imap(process_line, lines), desc="Filtering entries", total=len(lines))
        )

    # Write the filtered results back
    with open(output_file, "w") as outfile:
        for result in results:
            if result:  # Only write non-skipped lines
                outfile.write(result + "\n")


if __name__ == "__main__":
    main()