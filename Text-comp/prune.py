import json

input_file = "ass.jsonl"
output_file = "ass-pruned.jsonl"

with open(input_file, "r") as infile, open(output_file, "w") as outfile:
    for line in infile:
        record = json.loads(line)
        pruned_record = {key: record[key] for key in ("id", "title", "content") if key in record}
        outfile.write(json.dumps(pruned_record) + "\n")