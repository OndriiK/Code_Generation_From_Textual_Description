import os
import os.path
import litellm
from openai import OpenAI
import sys

# The prompt loading logic from the LocAgent system is reused to load the user Jinja2 templates for agents
from LocAgent.util.prompts.prompt import PromptManager
from LocAgent.plugins import LocationToolsRequirement


# UnifiedAgent encapsulates the API calls to models and the prompt management to abstract the LLM interactions
class UnifiedAgent:
    def __init__(
        self,
        prompt_template: str = None,
        model_name: str = "openrouter/qwen/qwen-2.5-coder-32b-instruct:free",
        prompt_dir: str = None,
        api_key: str = "",
    ):
        """
        Initializes a unified agent that can be specialized via a user prompt template.
        
        Args:
            prompt_template (str): Name of the user prompt template (without the .j2 extension).
            model_name (str): The model name to use with OpenRouter.
            prompt_dir (str): Directory where prompt templates are stored. If None, a default is used.
            api_key (str): API key for model querying.
        """        
        # Initialize the OpenAI client with OpenRouter API endpoint
        self.client = OpenAI(
            base_url="https://openrouter.ai/api/v1",
            api_key=api_key,
        )
        self.api_key = api_key
        self.model = model_name
        self.messages = None
        
        # Determine prompt_dir storing Jinja2 templates; if not provided, use a default relative path
        if prompt_dir is None:
            base_dir = os.path.dirname(os.path.abspath(__file__))
            prompt_dir = os.path.join(base_dir, "..", "LocAgent", "util", "prompts")

        agent_skills_docs = LocationToolsRequirement.documentation
        # Instantiate the PromptManager
        self.prompt_manager = PromptManager(prompt_dir=prompt_dir, agent_skills_docs=agent_skills_docs)
        
        # Load the common system prompt
        self.system_prompt = self.prompt_manager.system_message
        self.user_prompt = None

        # Load the agent-specific user prompt template if specified
        if prompt_template is not None:
            self.user_prompt = self.prompt_manager.load_and_render_user_prompt(prompt_template, context={"micro_agent": None})

    
    def get_system_prompt(self) -> str:
        """Returns the rendered system prompt."""
        return self.system_prompt
    
    
    def get_user_prompt(self) -> str:
        """Returns the rendered user prompt."""
        return self.user_prompt
    

    # central function for generating a response from the LLM through an API call
    def generate_completion(self, conversation: list) -> dict:
        """
        Generates a completion based on a provided conversation.
        
        Args:
            conversation (list): List of messages (each a dict with keys 'role' and 'content').
        
        Returns:
            dict: The LLM's response.
        """
        try:
            # execute the API call to get a response from the LLM
            response = litellm.completion(
                model=self.model,
                messages=conversation,
                temperature=1.0,
                stop=['</execute_ipython>'],
                api_key=self.client.api_key,
                api_base="https://openrouter.ai/api/v1"
            )
        except litellm.BadRequestError as e:
            print(f"'error': {str(e)}, 'type': 'BadRequestError'")
            return None

        # return the response from the LLM to continue the conversation
        return response
    

    # def generate_response_with_query(self, query: str) -> dict:
    #     """
    #     Convenience method to generate a response by constructing a conversation that includes
    #     the system prompt, user prompt, and the provided query.
        
    #     Args:
    #         query (str): The actual user query.
        
    #     Returns:
    #         dict: The LLM's response.
    #     """
    #     conversation = [
    #         {"role": "system", "content": self.get_system_prompt()},
    #         {"role": "user", "content": self.get_user_prompt()},
    #         {"role": "user", "content": query},
    #     ]
    #     return self.generate_completion(conversation)

 