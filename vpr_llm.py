import sys
import openai
import groq
import deepseek
import csv
import os
import argparse
from help_retriever import HelpRetriever

import tiktoken

num_input_tokens = 0
num_output_tokens = 0

class HelpTextManager:
    @staticmethod
    def read_help_file(help_file_path):
        """
        Read the help file containing command-line options.
        """
        try:
            with open(help_file_path, 'r') as file:
                return file.read()
        except FileNotFoundError:
            print(f"Error: The file '{help_file_path}' was not found.")
        except Exception as e:
            print(f"An error occurred: {e}")
        return None

    @staticmethod
    def extract_relevant_help(help_text, options):
        """
        Extract the relevant parts of the help text based on the proposed options.
        """
        relevant_help = []
        for option in options:
            if option in help_text:
                relevant_help.append(help_text[help_text.find(option):help_text.find("\n\n", help_text.find(option))])
        return "\n\n".join(relevant_help)


class LogFileProcessor:
    @staticmethod
    def extract_command_and_error(log_file_path, error_lines=1):
        """
        Extract the command and error from the log file.
        """
        command = None
        error = None

        try:
            with open(log_file_path, 'r') as file:
                lines = file.readlines()

                # Extract the command
                for i, line in enumerate(lines):
                    if "VPR was run with the following command-line:" in line:
                        if i + 1 < len(lines):
                            command = lines[i + 1].strip()
                        break

                # Extract the error (last `error_lines` lines of the file)
                if lines:
                    error = "\n".join(line.strip() for line in lines[-error_lines:])

        except FileNotFoundError:
            print(f"Error: The file '{log_file_path}' was not found.")
        except Exception as e:
            print(f"An error occurred: {e}")

        return command, error


class LLMClient:
    def __init__(self, provider, api_key):
        """
        Initialize the LLM client based on the selected provider.
        """
        self.provider = provider
        self.api_key = api_key
        self.client = self._initialize_client()

    def _initialize_client(self):
        if self.provider == "openai":
            return openai.OpenAI(api_key=self.api_key)
        elif self.provider == "groq":
            return groq.Groq(api_key=self.api_key)
        elif self.provider == "deepseek":
            return deepseek.DeepSeek(api_key=self.api_key)
        else:
            raise ValueError(f"Unsupported provider: {self.provider}")

    def ask_llm(self, model_name, messages, seed=1, temperature=0.3, max_tokens=500):
        """
        Generalized method to ask the LLM for a response.
        """
        global num_input_tokens
        global num_output_tokens

        try:
            response = self.client.chat.completions.create(
                model=model_name,
                messages=messages,
                max_tokens=max_tokens,
                seed=seed,
                temperature=temperature
            )
            num_input_tokens += response.usage.prompt_tokens
            num_output_tokens += response.usage.completion_tokens
            return response.choices[0].message.content.strip()
        except Exception as e:
            print(f"An error occurred while calling the {self.provider} API: {e}")
            return None


class CommandModifier:
    def __init__(self, client, provider, model_name, help_text, embedding_model, top_k_retrieve):
        self.client = client
        self.provider = provider
        self.model_name = model_name
        self.help_text = help_text
        self.embedding_model = embedding_model
        self.top_k_retrieve = top_k_retrieve

    def basic_mode(self, command, error, seed=1, temperature=0.3, max_tokens=500):
        """
        Ask the LLM for a modified command in "basic" mode.
        """
        system_message = (
            f"You are an expert in the VTR tool, specializing in debugging and command-line optimizations. "
            f"You have access to the following VTR command-line help text:\n\n{self.help_text}\n\n"
            f"Use this information to suggest minimal yet effective changes to fix errors encountered during a VTR run. "
            f"Your response must include only:\n"
            f"1. The modified command\n"
            f"2. A brief explanation of why the changes were made\n\n"
            f"Do not alter file names. Only increase channel width or grid size if other solutions are insufficient."
        )

        user_message = (
            f"The VTR tool was run with the following command:\n\n{command}\n\n"
            f"However, the following error occurred:\n\n{error}\n\n"
            f"Your task is to modify the command to resolve the error while preserving the original file names. "
            f"Prioritize fixing settings (e.g., effort levels, optimization parameters) over increasing device size (e.g., channel width, grid size), unless absolutely necessary. "
            f"Provide only the modified command and a brief explanation of the changes made."
        )
        '''
        system_message = (
            f"You are a helpful assistant familiar with the VTR tool. "
            f"Here is the help text for the VTR command-line options:\n\n{self.help_text}\n\n"
            f"Use this information to suggest changes to VTR commands when errors occur."
        )

        user_message = (
            f"The VTR tool was run with the following command:\n\n{command}\n\n"
            f"However, the following error occurred:\n\n{error}\n\n"
            f"Based on the error and the available command-line options, suggest a modified command to resolve the error. "
            f"Provide only the modified command and a brief explanation of why the change was made."
        )
        '''

        return self.client.ask_llm(
            self.model_name,
            [
                {"role": "system", "content": system_message},
                {"role": "user", "content": user_message}
            ],
            seed=seed,
            temperature=temperature,
            max_tokens=max_tokens
        )

    def cot_mode(self, command, error, seed=1, temperature=0.3, max_tokens=500):
        """
        Ask the LLM for a modified command using Chain of Thought (CoT) reasoning.
        """
        # Step 1: Propose all possible options related to the error
        step1_system_message = (
            f"You are a helpful assistant familiar with the VTR tool. "
            f"Here is the help text for the VTR command-line options:\n\n{self.help_text}\n\n"
            f"Use this information to suggest changes to VTR commands when errors occur."
        )

        step1_user_message = (
            f"The VTR tool was run with the following command:\n\n{command}\n\n"
            f"However, the following error occurred:\n\n{error}\n\n"
            f"Based on the error and the available command-line options, list all relevant options that can be added or changed that could resolve the error."
        )

        step1_output = self.client.ask_llm(
            self.model_name,
            [
                {"role": "system", "content": step1_system_message},
                {"role": "user", "content": step1_user_message}
            ],
            seed=seed,
            temperature=temperature,
            max_tokens=max_tokens
        )
        print("Step 1 Output:\n", step1_output)

        # Step 2: Shortlist the most relevant options
        proposed_options = [line.strip() for line in step1_output.split("\n") if line.strip()]
        relevant_help = HelpTextManager.extract_relevant_help(self.help_text, proposed_options)

        step2_system_message = (
            f"You are a helpful assistant familiar with the VTR tool. "
            f"Here is the relevant help text for the proposed options:\n\n{relevant_help}\n\n"
            f"Use this information to shortlist the most relevant options."
        )

        step2_user_message = (
            f"The VTR tool was run with the following command:\n\n{command}\n\n"
            f"However, the following error occurred:\n\n{error}\n\n"
            f"Here are the possible command-line options to resolve the error:\n\n{step1_output}\n\n"
            f"From the above options, shortlist the most relevant ones that are likely to solve the issue."
        )

        step2_output = self.client.ask_llm(
            self.model_name,
            [
                {"role": "system", "content": step2_system_message},
                {"role": "user", "content": step2_user_message}
            ],
            seed=seed,
            temperature=temperature,
            max_tokens=max_tokens
        )
        print("Step 2 Output:\n", step2_output)

        # Step 3: Evaluate the shortlisted options and generate a modified command
        shortlisted_options = [line.strip() for line in step2_output.split("\n") if line.strip()]
        relevant_help = HelpTextManager.extract_relevant_help(self.help_text, shortlisted_options)

        step3_system_message = (
            f"You are a helpful assistant familiar with the VTR tool. "
            f"Here is the relevant help text for the shortlisted options:\n\n{relevant_help}\n\n"
            f"Use this information to evaluate the options and generate a modified command."
        )

        step3_user_message = (
            f"The VTR tool was run with the following command:\n\n{command}\n\n"
            f"However, the following error occurred:\n\n{error}\n\n"
            f"Here are the shortlisted options to resolve the error:\n\n{step2_output}\n\n"
            f"Evaluate these options and generate a modified command that is most likely to resolve the error. "
            f"Provide only the modified command and a brief explanation of why the change was made."
        )

        return self.client.ask_llm(
            self.model_name,
            [
                {"role": "system", "content": step3_system_message},
                {"role": "user", "content": step3_user_message}
            ],
            seed=seed,
            temperature=temperature,
            max_tokens=max_tokens
        )


    def eval_n_temp_mode(self, command, error, seed=1, num_temp=3, max_tokens=500):
        """
        Ask the LLM for a modified command using multiple temperatures and evaluate the best response.
        """
        # Generate evenly spaced temperatures between 0 and 1
        temperatures = [i * (1 / (num_temp - 1)) for i in range(num_temp)]

        # Collect responses for each temperature
        responses = []
        for temp in temperatures:
            response = self.basic_mode(command, error, seed=seed, temperature=temp, max_tokens=max_tokens)
            if response:
                responses.append(response)
                print(f"Response at temperature {temp}:\n{response}\n")

        if not responses:
            print("No valid responses were generated.")
            return None

        # Ask the LLM to evaluate and pick the best response
        system_message = (
            f"You are a helpful assistant familiar with the VTR tool. "
        )

        user_message = (
                "The VTR tool was run with the following command:\n\n"
                f"{command}\n\n"
                "However, the following error occurred:\n\n"
                f"{error}\n\n"
                f"Here are {len(responses)} possible modified commands to resolve the error along with the explanation of each of them:\n\n"
                + "\n\n".join(map(str, responses)) + "\n\n"
                "From the proposed modified command, provide only the best modified command to resolve the error and a brief explanation of why it was chosen."
        )

        return self.client.ask_llm(
            self.model_name,
            [
                {"role": "system", "content": system_message},
                {"role": "user", "content": user_message}
            ],
            seed=seed,
            temperature=0.3,  # Use a moderate temperature for evaluation
            max_tokens=max_tokens
        )

    def eval_n_seed_mode(self, command, error, temperature=0.3, num_seed=3, max_tokens=500):
        """
        Ask the LLM for a modified command using multiple seeds and evaluate the best response.
        """
        # Generate evenly spaced temperatures between 0 and 1
        seeds = list(range(1, num_seed + 1))

        # Collect responses for each temperature
        responses = []
        for seed in seeds:
            response = self.basic_mode(command, error, seed=seed, temperature=temperature, max_tokens=max_tokens)
            if response:
                responses.append(response)
                print(f"Response at seed {seed}:\n{response}\n")

        if not responses:
            print("No valid responses were generated.")
            return None

        # Ask the LLM to evaluate and pick the best response
        system_message = (
            f"You are a helpful assistant familiar with the VTR tool. "
        )

        user_message = (
                "The VTR tool was run with the following command:\n\n"
                f"{command}\n\n"
                "However, the following error occurred:\n\n"
                f"{error}\n\n"
                f"Here are {len(responses)} possible modified commands to resolve the error along with the explanation of each of them:\n\n"
                + "\n\n".join(map(str, responses)) + "\n\n"
                "From the proposed modified command, provide only the best modified command to resolve the error and a brief explanation of why it was chosen."
        )

        return self.client.ask_llm(
            self.model_name,
            [
                {"role": "system", "content": system_message},
                {"role": "user", "content": user_message}
            ],
            seed=seed,
            temperature=0.3,  # Use a moderate temperature for evaluation
            max_tokens=max_tokens
        )

    def rag_mode(self, command, error, seed=1, temperature=0.3, max_tokens=500):
        """
        Ask the LLM for a modified command in "RAG" mode.
        """
        user_message = (
            f"The VTR tool was run with the following command:\n\n{command}\n\n"
            f"However, the following error occurred:\n\n{error}\n\n"
            f"Your task is to modify the command to resolve the error while preserving the original file names. "
            f"Prioritize fixing settings (e.g., effort levels, optimization parameters) over increasing device size (e.g., channel width, grid size), unless absolutely necessary. "
            f"Provide only the modified command and a brief explanation of the changes made."
        )

        vtr_help_retriever = HelpRetriever(self.help_text, self.embedding_model)
        retrieved_help = vtr_help_retriever.retrieve_help_text(user_message, self.top_k_retrieve)

        system_message = (
            f"You are an expert in the VTR tool, specializing in debugging and command-line optimizations. "
            f"You have access to the following VTR command-line help text:\n\n{retrieved_help}\n\n"
            f"Use this information to suggest minimal yet effective changes to fix errors encountered during a VTR run. "
            f"Your response must include only:\n"
            f"1. The modified command\n"
            f"2. A brief explanation of why the changes were made\n\n"
            f"Do not alter file names. Only increase channel width or grid size if other solutions are insufficient."
        )

        return self.client.ask_llm(
            self.model_name,
            [
                {"role": "system", "content": system_message},
                {"role": "user", "content": user_message}
            ],
            seed=seed,
            temperature=temperature,
            max_tokens=max_tokens
        )

    def rag_cot_mode(self, command, error, seed=1, temperature=0.3, max_tokens=500):
        """
        Ask the LLM for a modified command using Chain of Thought (CoT) reasoning with RAG.
        """

        step1_user_message = (
            f"The VTR tool was run with the following command:\n\n{command}\n\n"
            f"However, the following error occurred:\n\n{error}\n\n"
            f"Based on the error and the available command-line options, list all relevant options that can be added or changed that could resolve the error."
        )

        vtr_help_retriever = HelpRetriever(self.help_text, self.embedding_model)
        retrieved_help = vtr_help_retriever.retrieve_help_text(step1_user_message, self.top_k_retrieve)

        # Step 1: Propose all possible options related to the error
        step1_system_message = (
            f"You are a helpful assistant familiar with the VTR tool. "
            f"Here is the help text for the VTR command-line options:\n\n{retrieved_help}\n\n"
            f"Use this information to suggest changes to VTR commands when errors occur."
        )

        step1_output = self.client.ask_llm(
            self.model_name,
            [
                {"role": "system", "content": step1_system_message},
                {"role": "user", "content": step1_user_message}
            ],
            seed=seed,
            temperature=temperature,
            max_tokens=max_tokens
        )
        print("Step 1 Output:\n", step1_output)

        # Step 2: Shortlist the most relevant options
        #proposed_options = [line.strip() for line in step1_output.split("\n") if line.strip()]
        #relevant_help = HelpTextManager.extract_relevant_help(self.help_text, proposed_options)

        step2_system_message = (
            f"You are a helpful assistant familiar with the VTR tool. "
            f"Here is the relevant help text for the proposed options:\n\n{retrieved_help}\n\n"
            f"Use this information to shortlist the most relevant options."
        )

        step2_user_message = (
            f"The VTR tool was run with the following command:\n\n{command}\n\n"
            f"However, the following error occurred:\n\n{error}\n\n"
            f"Here are the possible command-line options to resolve the error:\n\n{step1_output}\n\n"
            f"From the above options, shortlist the most relevant ones that are likely to solve the issue."
        )

        step2_output = self.client.ask_llm(
            self.model_name,
            [
                {"role": "system", "content": step2_system_message},
                {"role": "user", "content": step2_user_message}
            ],
            seed=seed,
            temperature=temperature,
            max_tokens=max_tokens
        )
        print("Step 2 Output:\n", step2_output)

        # Step 3: Evaluate the shortlisted options and generate a modified command
        #shortlisted_options = [line.strip() for line in step2_output.split("\n") if line.strip()]
        #relevant_help = HelpTextManager.extract_relevant_help(self.help_text, shortlisted_options)

        step3_system_message = (
            f"You are a helpful assistant familiar with the VTR tool. "
            f"Here is the relevant help text for the shortlisted options:\n\n{retrieved_help}\n\n"
            f"Use this information to evaluate the options and generate a modified command.i MAke sure that the proposed command is a valid VPR command."
        )

        step3_user_message = (
            f"The VTR tool was run with the following command:\n\n{command}\n\n"
            f"However, the following error occurred:\n\n{error}\n\n"
            f"Here are the shortlisted options to resolve the error:\n\n{step2_output}\n\n"
            f"Evaluate these options and generate a modified command that is most likely to resolve the error. "
            f"Provide only the modified command and a brief explanation of why the change was made."
        )

        return self.client.ask_llm(
            self.model_name,
            [
                {"role": "system", "content": step3_system_message},
                {"role": "user", "content": step3_user_message}
            ],
            seed=seed,
            temperature=temperature,
            max_tokens=max_tokens
        )


class ExperimentLogger:
    @staticmethod
    def log_experiment(csv_file_path, log_file_name, provider, model_name, temperature, max_tokens, seed, error_lines, modified_command, mode, embedding_model, top_k):
        """
        Log the experiment details to a CSV file.
        """
        file_exists = os.path.isfile(csv_file_path)

        with open(csv_file_path, mode='a', newline='') as file:
            writer = csv.writer(file)

            if not file_exists:
                writer.writerow(["Log File", "LLM Provider", "LLM Model", "Mode", "Embedding Model", "Top-K Retrieved Blocks", "Temperature", "Max Tokens", "Seed", "Error Lines", "Modified Command", "Input Tokens", "Output Tokens", "LLM Error"])

            writer.writerow([log_file_name, provider, model_name, mode, embedding_model, top_k, temperature, max_tokens, seed, error_lines, modified_command, num_input_tokens, num_output_tokens])


class CommandModificationApp:
    def __init__(self):
        self.parser = argparse.ArgumentParser(description="Automate command modification using LLMs.")
        self._setup_arguments()
        num_input_tokens = 0
        num_output_tokens = 0


    def _setup_arguments(self):
        self.parser.add_argument("log_file_path", help="Path to the log file containing the command and error.")
        self.parser.add_argument("help_file_path", help="Path to the help file containing command-line options.")
        self.parser.add_argument("provider", choices=["openai", "groq", "deepseek"], help="LLM provider (openai, groq, deepseek).")
        self.parser.add_argument("model_name", help="Name of the LLM model to use.")
        self.parser.add_argument("api_key", help="API key for the LLM provider.")
        self.parser.add_argument("csv_file_path", help="Path to the CSV file to log experiment details.")
        self.parser.add_argument("--seed", type=int, default=1, help="Seed for reproducibility (default: 1).")
        self.parser.add_argument("--temperature", type=float, default=0.3, help="Temperature for LLM response (default: 0.3).")
        self.parser.add_argument("--max-tokens", type=int, default=500, help="Maximum number of tokens in the LLM response (default: 500).")
        self.parser.add_argument("--error-lines", type=int, default=1, help="Number of lines from the end of the log file to consider as the error (default: 1).")
        self.parser.add_argument("--mode", default="basic", choices=["basic", "cot", "eval_n_temp", "eval_n_seed","rag", "rag_cot"], help="Mode of operation (default: basic).")
        self.parser.add_argument("--num-temp", type=int, default=3, help="Number of temperatures to use in 'eval_n_temp' mode (default: 3).")
        self.parser.add_argument("--num-seed", type=int, default=3, help="Number of seeds to use in 'eval_n_seed' mode (default: 3).")
        self.parser.add_argument("--embedding-model", choices=[
           "sentence-transformers/all-MiniLM-L12-v2",
            "sentence-transformers/all-mpnet-base-v2",
            "nomic-ai/nomic-embed-text-v1",
            "sentence-transformers/msmarco-bert-base-dot-v5",
            "BAAI/bge-base-en",
            "BAAI/bge-large-en",
            "BAAI/bge-large-en-v1.5",
            "intfloat/e5-large-v2",
            "thenlper/gte-large"
            ], default="thenlper/gte-large", help="For RAG modes, the option determines the embedding model used")
        self.parser.add_argument("--top-k-retrieve", type=int, default=3, help="For RAG modes, the option determines the number of retrieved chunks.")

    def run(self):
        args = self.parser.parse_args()

        # Extract command and error
        command, error = LogFileProcessor.extract_command_and_error(args.log_file_path, args.error_lines)
        if not command or not error:
            print("Failed to extract command or error from the log file.")
            sys.exit(1)

        # Read help file
        help_text = HelpTextManager.read_help_file(args.help_file_path)
        if not help_text:
            print("Failed to read the help file.")
            sys.exit(1)

        # Initialize LLM client
        try:
            client = LLMClient(args.provider, args.api_key)
        except ValueError as e:
            print(e)
            sys.exit(1)

        # Initialize command modifier
        modifier = CommandModifier(client, args.provider, args.model_name, help_text, args.embedding_model, args.top_k_retrieve)

        # Dispatch to the appropriate mode
        if args.mode == "basic":
            modified_command = modifier.basic_mode(command, error, args.seed, args.temperature, args.max_tokens)
        elif args.mode == "cot":
            modified_command = modifier.cot_mode(command, error, args.seed, args.temperature, args.max_tokens)
        elif args.mode == "eval_n_temp":
            modified_command = modifier.eval_n_temp_mode(command, error, args.seed, args.num_temp, args.max_tokens)
        elif args.mode == "eval_n_seed":
            modified_command = modifier.eval_n_seed_mode(command, error, args.temperature, args.num_seed, args.max_tokens)
        elif args.mode == "rag":
            modified_command = modifier.rag_mode(command, error, args.seed, args.temperature, args.max_tokens)
        elif args.mode == "rag_cot":
            modified_command = modifier.rag_cot_mode(command, error, args.seed, args.temperature, args.max_tokens)
        else:
            print(f"Error: Unsupported mode '{args.mode}'.")
            sys.exit(1)

        #if modified_command:
        if True:
            print("Modified Command Suggestion:")
            print(modified_command)

            # Log the experiment
            log_file_name = os.path.basename(args.log_file_path)
            #log_file_name = os.path.dirname(args.log_file_path)
            ExperimentLogger.log_experiment(
                args.csv_file_path, log_file_name, args.provider, args.model_name, args.temperature, args.max_tokens, args.seed, args.error_lines, modified_command, args.mode, args.embedding_model, args.top_k_retrieve
            )
            print(f"Experiment logged to {args.csv_file_path}.")
        else:
            print("Failed to get a modified command suggestion from the LLM.")


if __name__ == "__main__":
    app = CommandModificationApp()
    app.run()
