from rapidfuzz import fuzz, process
import orjson
from multiprocessing import Pool, Manager
from tqdm import tqdm

input_file = "deduped_ass.jsonl"
output_file = "filtered_file.jsonl"
similarity_threshold = 85  # Percentage threshold for similarity
num_workers = 12  # Use your available cores
batch_size = 1000  # Number of records per chunk


def is_similar(new_content, seen_contents):
    """
    Check for similarity to already-seen contents using RapidFuzz.
    """
    matches = process.extract(
        new_content, seen_contents, scorer=fuzz.ratio, limit=1
    )  # Check against limited candidates
    if matches and matches[0][1] >= similarity_threshold:
        return True
    return False


def process_chunk(chunk, shared_seen_contents, lock):
    """
    Deduplicate a chunk of records.
    """
    local_seen = set()  # A local set to avoid duplicates within this chunk
    unique_records = []  # List of unique records to return
    skipped_records = 0  # Counter for skipped records

    for line in chunk:
        try:
            record = orjson.loads(line)
            content = record.get("content", "")

            if not content:
                # Skip records with empty content
                skipped_records += 1
                continue

            with lock:
                if content in shared_seen_contents:
                    # Already globally seen; skip this record
                    skipped_records += 1
                    continue

            # Perform fuzzy matching locally
            if not is_similar(content, local_seen):
                local_seen.add(content)
                unique_records.append(record)
            else:
                # Fuzzy match too similar; skip record
                skipped_records += 1
        except Exception as e:
            print(f"Error processing record: {e}")
            skipped_records += 1

    with lock:
        # Update globally shared content with locally seen unique ones
        shared_seen_contents.update(local_seen)

    print(f"Chunk processed. Unique records: {len(unique_records)}, Skipped records: {skipped_records}")
    return unique_records


def main():
    # Read all lines from the input file
    with open(input_file, "r") as infile:
        lines = infile.readlines()

    # Split the lines into chunks for multiprocessing
    chunks = [lines[i : i + batch_size] for i in range(0, len(lines), batch_size)]

    # Set up shared memory using Manager
    manager = Manager()
    shared_seen_contents = manager.list()  # Shared content tracker
    lock = manager.Lock()

    # Use multiprocessing to process each chunk
    with Pool(num_workers) as pool:
        results = list(
            tqdm(
                pool.starmap(
                    process_chunk,
                    [(chunk, shared_seen_contents, lock) for chunk in chunks],
                ),
                desc="Multiprocessing fuzzy deduplication",
                total=len(chunks),
            )
        )

    # Flatten all the unique records from the multiprocessing results
    filtered_records = [record for chunk_results in results for record in chunk_results]

    print(f"Total unique records after processing: {len(filtered_records)}")

    # Write the deduplicated records to the output file
    with open(output_file, "w") as outfile:
        for record in filtered_records:
            outfile.write(orjson.dumps(record).decode("utf-8") + "\n")


if __name__ == "__main__":
    main()