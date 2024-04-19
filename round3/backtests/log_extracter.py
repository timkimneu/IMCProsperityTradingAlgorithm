import json
import re
import argparse

def process_log_file(file_path, output_file_path):
    extracted_logs = []

    with open(file_path, 'r') as file:
        log_entries = file.read()

    # Split the log entries into individual logs
    individual_logs = log_entries.split("|")
    for log in individual_logs:
        #print(log)
        extracted_logs.append(log)

    # Write the extracted logs to a new file
    with open(output_file_path, 'w') as file:
        for log in extracted_logs:
            file.write(log + "\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='log file to analyze')
    parser.add_argument('file_path', type=str, help="file_path to log file")
    parser.add_argument('output_file_path', type=str, help="file_path to output the log file")
    args = parser.parse_args()
    # Replace 'path_to_your_log_file.txt' with the actual file path
    process_log_file(args.file_path, args.output_file_path)
    print("done")
