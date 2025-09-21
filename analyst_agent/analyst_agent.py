import re
import json
import os
import queue
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from analyst_agent.clarification_module import ClarificationModule
from analyst_agent.planning_module import PlanningModule
from analyst_agent.mode_determination import OpenRouterModeClassifier
from prompts.prompts import RESULT_CONSOLIDATION_INSTRUCTION
from workflows.unified_agent_class import UnifiedAgent
# reused iterative loop for agent logic
from workflows.agent_logic import agent_logic_loop

class AnalystAgent:
    def __init__(self, model_name: str = "openrouter/qwen/qwen-2.5-coder-32b-instruct:free", repo_path: str = "", output_folder: str = "/mnt/d/wsl_workspace/MYSYSTEMCOPY/loc_agent_outputs_resources/LocAgentResults", shared_memory = None, api_key: str = None, separate_api_key_mode: bool = False):
        """
        Initializes the AnalystAgent with the necessary components for clarification and planning.

        :param model_name: The model to be used by both the clarification and planning modules.
        :param prompt_template: The prompt template for the clarification module.
        :param repo_path: The path to the code repository.
        :param output_folder: The folder for storing the output from LocAgent during the planning phase.
        :param shared_memory: A shared memory object for inter-agent communication.
        :param api_key: The API key for accessing the OpenRouter API.
        """
        self.model_name = model_name
        self.repo_path = repo_path
        self.output_folder = output_folder
        self.shared_memory = shared_memory
        self.queue = queue.Queue()
        self.mode = None # store the mode of the analyst agent to access it in the Streamlit application

        # Initialize ModeDetermination to classify task mode (clarification or planning)
        self.mode_classifier = OpenRouterModeClassifier(api_key=api_key)

        # Initialize the clarification and planning modules
        self.clarification_module = ClarificationModule(model_name=model_name, api_key=api_key)
        self.planning_module = PlanningModule(repo_path=repo_path, output_folder=output_folder, api_key=api_key, separate_api_key_mode=separate_api_key_mode)
        
        # Intent classification model will be initialized when needed
        self.intent_model = None
        self.intent_tokenizer = None
        self.predefined_intents = [
            "debug_code",
            "add_feature",
            "write_tests",
            "document_code"
        ]

    def load_intent_classification_model(self):
        """
        Load the fine-tuned intent classification model.
        """
        if self.intent_model is None:
            # Load the model only if it's not already loaded
            try:
                parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
                model_path = os.path.join(parent_dir, "train", "fine_tuned_codebert")
                print(f"Loading intent classification model from {model_path}")
                self.intent_tokenizer = AutoTokenizer.from_pretrained(model_path)
                self.intent_model = AutoModelForSequenceClassification.from_pretrained(model_path)
                print("Intent classification model loaded successfully.")
            except Exception as e:
                print(f"Error loading intent classification model: {e}")
                # Fallback to operating without the model
                self.intent_model = None
                self.intent_tokenizer = None

    def classify_intent(self, query):
        """
        Classify the intent of a user query using the fine-tuned model.
        
        :param query: The user query to classify
        :return: List of detected intents
        """
        # Ensure model is loaded
        if self.intent_model is None or self.intent_tokenizer is None:
            self.load_intent_classification_model()
            
            # If still None after loading attempt, retuern a signal for analyst agent
            if self.intent_model is None:
                return "<Intent_Classification_Failed>"
        
        # Tokenize the input
        inputs = self.intent_tokenizer(query, return_tensors="pt", truncation=True, padding="max_length", max_length=256)
        
        # Run the model inference
        self.intent_model.eval()
        with torch.no_grad():
            outputs = self.intent_model(**inputs)
        
        # Get raw logits and apply sigmoid for multi-label classification
        logits = outputs.logits
        scores = torch.sigmoid(logits).squeeze().tolist()
        print(f"Intent classification scores: {list(zip(self.predefined_intents, scores))}")

        # Determine which intents are detected based on a threshold
        threshold = 0.3
        detected_intents = [intent for intent, score in zip(self.predefined_intents, scores) if score > threshold]

        if not detected_intents:
            detected_intents = "<No_Intent_Detected_with_Confidence>"

        return detected_intents

    def process_task(self, task: dict, instance_id: str) -> str:
        """
        Process a task by determining whether it requires clarification or planning,
        and then run the appropriate module.

        :param task: The task dictionary containing the problem statement and other data.
        :param instance_id: A unique identifier for the instance of the task.
        :return: The final output from either the clarification or planning module.
        """
        # Extract the problem statement from the task
        problem_statement = task.get("content", "")

        # Use the ModeDetermination component to classify the task
        mode = self.mode_classifier.detect_mode(problem_statement)
        print(f"Detected mode: {mode}")
        self.mode = mode

        result = None
        if mode == "clarification":
            print("Clarification mode activated.")
            result = self.clarification_module.run_clarification(problem_statement, self.repo_path, instance_id)
            print(f"[ANALYST] Clarification result:\n{result}")
        elif mode == "completion":
            print("Planning mode activated.")
            
            # Classify the intent using the fine-tuned model
            detected_intents = self.classify_intent(problem_statement)
            # print(f"Detected intents: {detected_intents}")
            
            # Execute planning with the enhanced context from detected intents
            result = self.planning_module.execute(instance_id, problem_statement, detected_intents=detected_intents)
        else:
            raise ValueError(f"Unknown mode: {mode}")
        
        # If we have a shared memory and the result contains a plan, initialize the subtasks
        if mode == "completion" and self.shared_memory:
            # Extract the JSON plan from the result
            json_pattern = r'\[\s*\{.*?\}\s*\]'
            match = re.search(json_pattern, result, re.DOTALL)
            
            if match:
                try:
                    tasks_plan = json.loads(match.group(0))

                    # Initialize the subtasks tracking
                    self.shared_memory.write_message({
                        "type": "subtasks_init",
                        "sender": "analyst_agent",
                        "tasks_plan": tasks_plan
                    })
                except json.JSONDecodeError as e:
                    print(f"Error parsing JSON plan: {e}")

        return result


    def consolidate_results(self, original_task: str = None) -> str:
        """
        Collect all task results from the agent queue and consolidate them into CRUD operations format.
        
        Returns:
            Consolidated CRUD operations as a JSON string
        """
        print(f"[ANALYST] Processing collected agent results")
        
        # Collect all outputs from the queue (containing results)
        agent_outputs = {}
        
        # Process all messages currently in the queue
        while not self.queue.empty():
            try:
                message = self.queue.get_nowait()
                
                # Skip messages that aren't task results
                if message.get("type") != "task_result":
                    continue
                    
                sender = message.get("sender")
                content = message.get("content", "")
                
                print(f"[ANALYST] Processing result from {sender}")
                agent_outputs[sender] = content
                    
            except queue.Empty:
                break
        
        formatted_outputs = None

        # Check if we got any outputs
        if not agent_outputs:
            print(f"[ANALYST] Warning: No agent outputs found in queue")
            formatted_outputs = "ALL AGENTS FAILED TO COMPLETE THEIR TASKS"
        else:
            # Format the collected outputs for the consolidation prompt
            formatted_outputs = "\n\n".join([
                f"=== {agent.upper()} OUTPUT ===\n{content}"
                for agent, content in agent_outputs.items()
            ])
        
        final_prompt = RESULT_CONSOLIDATION_INSTRUCTION.substitute(original_assignment=original_task, agent_outputs=formatted_outputs)
        print(f"[ANALYST] constructing final message to developer...")
        # Continue the conversation of the planning module's agent
        consolidation_agent = self.planning_module.agent        

        # Prepare the instance data for agent_logic_loop with CRUD prompt as problem statement
        instance_data = {
            "instance_id": "crud_consolidation",
            "problem_statement": final_prompt,
            "repo": self.repo_path,
            "base_commit": "latest",
            "patch": ""
        }

        # Use agent_logic_loop to process the consolidation task
        final_message_to_developer = agent_logic_loop(
            agent_llm=consolidation_agent,
            instance_data_dict=instance_data,
            agent_specific_prompt=final_prompt,
            previous_conversation=consolidation_agent.messages
        )
        
        return final_message_to_developer
