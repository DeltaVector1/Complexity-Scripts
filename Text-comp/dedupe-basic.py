import orjson
from tqdm import tqdm

input_file = "tokenized-ass.jsonl"
output_file = "deduped_ass.jsonl"

def main():
    seen_contents = set()  # Store unique content
    unique_records = []

    with open(input_file, "r") as infile:
        for line in tqdm(infile, desc="Deduplicating"):
            record = orjson.loads(line)
            content = record.get("content", "")

            if content not in seen_contents:
                seen_contents.add(content)
                unique_records.append(record)

    with open(output_file, "w") as outfile:
        for record in unique_records:
            outfile.write(orjson.dumps(record).decode("utf-8") + "\n")


if __name__ == "__main__":
    main()