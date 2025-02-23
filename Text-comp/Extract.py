import json

def filter_jsonl(input_file, output_file):
    with open(input_file, 'r') as infile, open(output_file, 'w') as outfile:
        for idx, line in enumerate(infile, 1):
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                print(f"Line {idx} in {input_file} is garbage JSON: {line.strip()}")
                continue

            evaluation = obj.get("evaluation")
            if (isinstance(evaluation, int) and 3 <= evaluation <= 6) or (
                isinstance(evaluation, dict) and 3 <= evaluation.get("rating", 0) <= 6):
                outfile.write(json.dumps(obj) + '\n')
            else:
                print(f"Line {idx} skipped. Evaluation doesn't match criteria or is nonsense: {evaluation}")

filter_jsonl("rated_file-final.jsonl", "output.jsonl")