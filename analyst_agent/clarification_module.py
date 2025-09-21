# CogniCollab
# analyst_agent/clarification_module.py

from workflows.unified_agent_class import UnifiedAgent
from workflows.agent_logic import agent_logic_loop
from prompts.prompts import CLARIFICATION_TASK_INSTRUCTION

# ClarificationModule is responsible for handling the clarification tasks - explaining code segments and providing context.
class ClarificationModule:
    def __init__(self, model_name: str = "openrouter/qwen/qwen-2.5-coder-32b-instruct:free", api_key: str = None):
        """
        Initializes the ClarificationModule.
        
        :param model_name: The name of the model to be used (e.g., "openrouter/qwen/qwen-2.5-coder-32b-instruct:free").
        :param prompt_template: The prompt template used for clarification, default is "myuser_prompt".
        """
        self.model_name = model_name
        self.prompt_template = "myuser_prompt"
        # initialize the underlying LLM
        self.agent = UnifiedAgent(
            prompt_template=self.prompt_template,
            model_name=self.model_name,
            api_key=api_key,
        )

    def run_clarification(self, problem_statement: str, repo_path: str, instance_id: str) -> str:
        """
        Executes the clarification task for a given problem statement.
        
        :param problem_statement: The problem or task that needs clarification.
        :param repo_path: The repository path (used for context).
        :param base_commit: The commit reference for the repository.
        :param patch: Any specific patch or modification to consider.
        :return: The final clarification response from the agent.
        """
        # Prepare the instance_data
        instance_data = {
            "instance_id": instance_id,
            "problem_statement": problem_statement,
            "repo": repo_path,
            "base_commit": "latest",
            "patch": ""
        }

        # Call the agent logic loop with clarification task instruction
        final_output = agent_logic_loop(agent_llm=self.agent, instance_data_dict=instance_data, agent_specific_prompt=CLARIFICATION_TASK_INSTRUCTION)

        return final_output
