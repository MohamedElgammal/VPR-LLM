import os
import subprocess
import csv
import re 
import pandas as pd

from vpr_llm import CommandModificationApp

# Base command parameters
script = "python vpr_llm.py"
testcases_dir = "testcases/"
vpr_filename = "error.log"
api_token = ""

seed = 1
temperature = 0.5
max_tokens = 1000
error_lines = 10
#mode = "rag"
mode = ""
llm_output_csv = ""
vpr_output_csv = ""
input_help = "help_srcs/help.txt"
#llm_model = "llama-3.1-8b-instant"
llm_model = ""
embedding_model = "thenlper/gte-large"
#top_k_retrieve = 3
top_k_retrieve = 0

script_dir = os.path.dirname(os.path.abspath(__file__))  # Get the script's directory

def extract_vpr_command(text):
    # Regular expression pattern to match the command starting with 'vpr' and containing *.xml
    pattern = r"\b(?:\S*?vpr\S*)\b[^\n]*?\.xml[^\n]*"

    # Search for the command using the regular expression
    match = re.search(pattern, text)

    # If a match is found, return the command
    if match:
        vtr_home = os.getenv('VTR_HOME', '')
        command_with_prefix = f"{vtr_home}{match.group(0).strip().strip('`')}"
        return command_with_prefix  # Strip any leading or trailing spaces
    else:
        return None

def get_last_log_and_command(csv_file):
    """Reads the last row of a CSV file and extracts 'Log File' and 'Modified Command'."""
    with open(csv_file, newline="", encoding="utf-8") as csvfile:
        reader = csv.DictReader(csvfile)
        last_row = None
        for row in reader:
            last_row = row  # Keep overwriting until the last row

    if last_row:
        return last_row["Log File"], last_row["Modified Command"]
    else:
        return None, None  # Handle empty files

def run_command_and_check_log(directory, command):
    """Changes to the specified directory, runs the given command,
    and checks if 'vpr_stdout.log' contains the string 'VPR succeeded'."""
    
    # Change to the given directory
    try:
        print(f"Changing directory to: {directory}")
        full_directory = os.path.abspath(os.path.join(script_dir, directory))  # Convert relative to absolute
        os.chdir(full_directory)
    except FileNotFoundError:
        print(f"Directory {full_directory} not found.")
        return False, None, f"Directory {full_directory} not found."
    
    # Run the command
    result = None
    try:
        print(f"Running cmd: {command}")
        result = subprocess.run(command, shell=True, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    except subprocess.CalledProcessError as e:
        print(f"Command '{command}' failed.")
        return False, None, result.stderr.replace(",","") if result else None
    
    # Check if 'vpr_stdout.log' exists and contains 'VPR succeeded'
    log_file = "vpr_stdout.log"
    if os.path.exists(log_file):
        embedding_model_safe = embedding_model.replace("/", "_")
        new_filename = f"seed_{seed}_temp_{temperature}_tokens_{max_tokens}_errors_{error_lines}_mode_{mode}_llm_{llm_model}_embed_{embedding_model_safe}_topk_{top_k_retrieve}.log"
        os.rename(log_file, new_filename)
        with open(new_filename, "r") as file:
            content = file.read()
            if "VPR succeeded" in content:
                # Extract WL (Wire Length)
                wl_match = re.search(r"Total wirelength: (\d+), average net length:", content)
                wl = wl_match.group(1) if wl_match else None

                # Extract Channel width
                channel_width_match = re.search(r"Circuit successfully routed with a channel width factor of (\d+)", content)
                channel_width = channel_width_match.group(1) if channel_width_match else None

                # Extract CPD (Critical Path Delay)
                cpd_match = re.search(r"Final critical path delay \(least slack\): (\d+\.\d+) ns, Fmax: (\d+\.\d+) MHz", content)
                cpd = cpd_match.group(1) if cpd_match else None

                grid_size_match = re.search(r"FPGA sized to \d+ x \d+: (\d+) grid tiles", content)
                grid_size = grid_size_match.group(1) if grid_size_match else None

                return True, {"Channel Width": channel_width, "Grid Size": grid_size, "WL": wl, "CPD": cpd}, result.stderr.replace(",","") if result else None
    
    return False, None, f"{log_file} does not exist"

def process_csv_and_run_commands(input_csv, output_csv):
    """Process the CSV file, run the commands, and update the output CSV with success and metrics."""
    with open(input_csv, newline="", encoding="utf-8") as csvfile:
        reader = csv.DictReader(csvfile)
        fieldnames = reader.fieldnames + ["LLM Error", "VPR Error", "Success"] + ["Channel Width", "Grid Size", "WL", "CPD"]  # Add extra columns for success and metrics
        rows = list(reader)
    # Write the updated rows to the output CSV
    with open(output_csv, mode="w", newline="", encoding="utf-8") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()

        for row in rows:
            # Get 'Log File' and 'Modified Command' from each row
            testcase_dir, llm_output = row["Log File"], row["Modified Command"]
            
            if testcase_dir and llm_output:
                # Extract the VPR command
                modified_command = extract_vpr_command(llm_output)
                print("Command:", modified_command) 

                # Run command and check the log
                passed, metrics, vpr_error = run_command_and_check_log(testcase_dir, modified_command)
                
                # Add 'Success' and metrics to the row
                row["VPR Error"] = vpr_error

                row["Success"] = "True" if passed else "False"
                row["WL"] = metrics["WL"] if metrics and metrics["WL"] else None
                row["Channel Width"] = metrics["Channel Width"] if metrics and metrics["Channel Width"] else None
                row["CPD"] = metrics["CPD"] if metrics and metrics["CPD"] else None
                row["Grid Size"] = metrics["Grid Size"] if metrics and metrics["Grid Size"] else None
        
            writer.writerow(row)
            print(f"Results logged to: {output_csv}\n")



os.makedirs("output/", exist_ok=True)

# Iterate over directories under testcases/
leaf_dirs = []
for root, dirs, files in os.walk(testcases_dir, topdown=True):
    if not dirs:
        leaf_dirs.append(root)

for mode in ["rag"]:
    for llm_model in ["llama-3.1-8b-instant"]:
        for top_k_retrieve in [5]:
            for subdir in sorted(leaf_dirs):
                subdir_path = subdir
                log_file = os.path.join(subdir_path, vpr_filename)
                embedding_model_safe = embedding_model.replace("/", "_")
                help_name = input_help.split('/')[1].strip(".txt")
                llm_output_csv = f"output/llm_seed_{seed}_temp_{temperature}_tokens_{max_tokens}_errors_{error_lines}_mode_{mode}_llm_{llm_model}_embed_{embedding_model_safe}_topk_{top_k_retrieve}_{help_name}.csv"
                vpr_output_csv = f"output/vpr_seed_{seed}_temp_{temperature}_tokens_{max_tokens}_errors_{error_lines}_mode_{mode}_llm_{llm_model}_embed_{embedding_model_safe}_topk_{top_k_retrieve}_{help_name}.csv"

                other_args = f"{input_help} groq {llm_model}  {api_token} {llm_output_csv} --seed {seed} --temperature {temperature} --max-tokens {max_tokens} --error-lines {error_lines} --mode {mode} --embedding-model {embedding_model} --top-k-retrieve {top_k_retrieve}"
                   
                if os.path.isdir(subdir_path) and os.path.isfile(log_file):
                    command = f"{script} {log_file} {other_args}"
                    print(f"Running: {command}")
                    result = subprocess.run(command, shell=True, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)

                    df = pd.read_csv(llm_output_csv)
                    df["LLM Error"] = df["LLM Error"].astype("object")
                    df.at[df.index[-1], "LLM Error"] = result.stdout.replace(",","") + result.stderr.replace(",","")
                    df.to_csv(llm_output_csv, index=False)
                
            process_csv_and_run_commands(llm_output_csv, vpr_output_csv)

