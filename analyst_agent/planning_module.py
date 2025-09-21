import os
import subprocess
import json
from typing import List
from openai import OpenAI
from prompts.prompts import COMPLETION_TASK_TEMPLATE
from workflows.unified_agent_class import UnifiedAgent
from workflows.agent_logic import agent_logic_loop

# Class for the Planning Module, enriched with LocAgent - responsible for context-aware plan generation.
class PlanningModule:
    def __init__(self, repo_path: str, output_folder: str, api_key: str = None, separate_api_key_mode: bool = False):
        self.repo_path = repo_path
        self.output_folder = output_folder
        self.relevant_files = []
        self.agent = UnifiedAgent(
            prompt_template="user_prompt_planning_module",
            model_name="openrouter/qwen/qwen-2.5-72b-instruct:free",
            api_key=api_key,
        )
        self.separate_api_key_mode = separate_api_key_mode


    # lists available files in the local repo
    def get_file_list_string(self, allowed_extensions=None) -> str:
        """
        Returns a formatted string listing all file paths in the repo,
        using paths relative to the repo root.
        """
        file_list = []
        for root, _, files in os.walk(self.repo_path):
            for file in files:
                if allowed_extensions and not any(file.endswith(ext) for ext in allowed_extensions):
                    continue
                # Get path relative to repo_path instead of absolute path
                abs_path = os.path.join(root, file)
                rel_path = os.path.relpath(abs_path, self.repo_path)
                file_list.append(rel_path)
        return "\n".join(file_list)

    def run_loc_agent(self, instance_id: str, query: str):
        """
        Runs the LocAgent tool to identify relevant code files based on the user query.
        """
        # Get current script directory
        script_dir = os.path.dirname(os.path.abspath(__file__))

        # Get parent directory
        parent_dir = os.path.dirname(script_dir)
        # command to create a subprocess for LocAgent analysis
        if self.separate_api_key_mode:
            individual_api_key = 'True'
        else:
            individual_api_key = 'False'
        
        command = [
            "python", "-m", "LocAgent.auto_search_main",
            "--localize",
            "--output_folder", self.output_folder,
            "--query", query,
            "--repo", self.repo_path,
            "--num_samples", '1',
            "--model", "openrouter/qwen/qwen-2.5-coder-32b-instruct:free",
            "--log_level", "INFO",
            "--instance_id", instance_id,
            "--separate_keys", individual_api_key
        ]
        subprocess.run(command, cwd=parent_dir, check=True)
        # Parse the output
        result_path = os.path.join(self.output_folder, "loc_outputs.jsonl")
        if not os.path.exists(result_path):
            print("No output file found. Check LocAgent logs.")
            return

        with open(result_path, "r") as f:
            lines = f.readlines()
        
        if not lines:
            print("No localization results found.")
            return
        
        # Extract the relevant files from the last output line
        last_output = json.loads(lines[-1])
        self.relevant_files = last_output.get("raw_output_loc", [])
        # LocAgent may return "ERROR" if it fails to analyze the codebase
        if self.relevant_files == "ERROR":
            self.relevant_files = ["Codebase analysis failed. Use the search tool to find relevant context!"]


    # main function to generate a plan
    def generate_plan(self, instance_id: str, query: str, detected_intents) -> str:
        """
        Uses the UnifiedAgent to generate a detailed, actionable plan based on the user's query and relevant files.
        """
        # Include the available files in the prompt
        code_files = self.get_file_list_string()

        instance_data = {
            "instance_id": instance_id,
            "problem_statement": query,
            "repo": self.repo_path, # Custom repo passed by the devloper
            "base_commit": "latest",
            "patch": ""
        }


        # Pass relevant files and code files to the agent for plan generation
        final_output = agent_logic_loop(
            agent_llm=self.agent,
            instance_data_dict=instance_data,
            agent_specific_prompt=COMPLETION_TASK_TEMPLATE.substitute(intent_detected=detected_intents, relevant_code_summary=self.relevant_files[0], available_files=code_files)
        )
        print("[ANALYST] Final output from the planning module:\n", final_output)

        # Return the final output to the Analyst Agent's main logic
        return final_output

    # main function to execute the planning module
    def execute(self, instance_id: str, query: str, detected_intents) -> str:
        """
        Full execution process: running LocAgent and generating the plan.
        """
        # Run LocAgent to identify relevant files
        self.run_loc_agent(instance_id, query)

        if self.relevant_files:
            plan = self.generate_plan(instance_id, query, detected_intents)
        else:
            print("Relevant files analysis fiailed, proceeding with a default response form LocAgent.")
            self.relevant_files = ["Codebase analysis failed. Use the search tool to find relevant context!"]
            plan = self.generate_plan(instance_id, query, detected_intents)

        if plan:
            return plan
        else:
            print("No plan generated.")
            return None

