from openai import OpenAI
import os
import litellm

# class that determines the mode of the Analyst Agent
class OpenRouterModeClassifier:
    def __init__(self, api_key: str = None):
        self.client = OpenAI(
            base_url="https://openrouter.ai/api/v1",
            api_key=api_key,
        )
        self.model = "openrouter/qwen/qwen-2.5-coder-32b-instruct:free"

    def detect_mode(self, query: str) -> str:
        """
        Classifies the given query into 'clarification' or 'completion'.

        Args:
            query (str): The user's query.

        Returns:
            str: One of 'clarification' or 'completion'.
        """
        # Define the prompt for mode determination. The LLM will classify the query into one of the two categories.
        prompt = f"""
Classify the following software engineering query into one of two categories: "clarification" or "completion".

Clarification: The query is asking to explain or better understand existing code or logic.
Completion: The query is asking to perform a task, such as modifying code, adding features, or fixing bugs.

Query: {query}
Only respond with one word: clarification or completion.
"""
        messages: list[dict] = [
            {"role": "system", "content": "You are a classification assistant."},
            {"role": "user", "content": prompt}
        ]
        try:
            # Call the OpenRouter API to classify the query
            response = litellm.completion(
                model=self.model,
                messages=messages,
                temperature=1.0,
                stop=['</execute_ipython>'],
                api_key=self.client.api_key,
                api_base="https://openrouter.ai/api/v1"
            )
        except litellm.BadRequestError as e:
            print(f"'error': {str(e)}, 'type': 'BadRequestError'")
            return None

        result = response.choices[0].message.content.strip().lower()
        return "clarification" if "clarification" in result else "completion"
