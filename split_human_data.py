import os

input_path = "data/games.jsonl"
output_dir = "data/human_batches"
lines_per_file = 100  # 1 million per file

os.makedirs(output_dir, exist_ok=True)

def split_file(input_file, output_directory, chunk_size):
    with open(input_file, "r") as infile:
        file_count = 0
        outfile = open(os.path.join(output_directory, f"games_part_{file_count}.jsonl"), "w")
        for i, line in enumerate(infile):
            if i > 0 and i % chunk_size == 0:
                outfile.close()
                file_count += 1
                outfile = open(os.path.join(output_directory, f"games_part_{file_count}.jsonl"), "w")
            outfile.write(line)
        outfile.close()

split_file(input_path, output_dir, lines_per_file)