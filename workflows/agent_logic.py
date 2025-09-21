# Main iterative logic to facilitate reasoning and action steps for the agent
# This file is heavily inspired by the guiadance logic for the LocAgent's LLM, stored in LocAgent/auto_search_main.py
# Components of LocAgent are reused to parse agent outpts and execute tool calls

from typing import List
import os
import sys
from prompts.prompts import FAKE_USER_MSG_FOR_CLARIFICATION
from workflows.unified_agent_class import UnifiedAgent


from LocAgent.util.runtime.fn_call_converter import (
    convert_fncall_messages_to_non_fncall_messages
    # STOP_WORDS as NON_FNCALL_STOP_WORDS
)
from LocAgent.util.utils import *
from LocAgent.util.actions.action_parser import ResponseParser
from LocAgent.util.actions.action import ActionType
from LocAgent.util.runtime.execute_ipython import execute_ipython
from LocAgent.plugins.location_tools.repo_ops.repo_ops import (
    set_current_issue,
    reset_current_issue,
    get_current_repo_modules,
)

# Example usage:
def agent_logic_loop(agent_llm : UnifiedAgent = None, instance_data_dict: dict = None, agent_specific_prompt: str = None, previous_conversation: list[dict] = None):

    # set_current_issue(instance_data=instance_data_dict)
    fake_user_msg = "Please provide a new response with different information or approach the problem differently."
    
    # construct the initial conversation state for the agent
    if not previous_conversation:
        messages: list[dict] = [{
            "role": "system",
            "content": agent_llm.system_prompt
        }]

        if agent_llm.user_prompt:
            messages.append({
                "role": "user",
                "content": agent_llm.user_prompt
            })

        messages.append({
            "role": "user",
            "content": agent_specific_prompt,
        })

        messages.append({
            "role": "user",
            "content": instance_data_dict['problem_statement'],
        })
    # or continue the conversation to finish some interleaved objective
    else:
        messages = previous_conversation
        messages.append({
            "role": "user",
            "content": agent_specific_prompt,
        })

    tools = []  # No native function calling
    messages = convert_fncall_messages_to_non_fncall_messages(
        messages, tools, add_in_context_learning_example=False
    )

    # initialize parser from LocAgent
    parser = ResponseParser()

    message_stall_count = 0
    MESSAGE_STALL_THRESHOLD = 5

    final_output = "NO OUTPUT"

    cur_interation_num = 0
    # Set the maximum number of iterations
    max_iteration_num = 16
    last_message = None
    previous_substantive_message = None 
    finish = False
    empty_message_count = 0
    final_output = None
    fail = False
    while not finish:
        # Check if the current iteration number exceeds the maximum allowed to prevent infinite operation
        cur_interation_num += 1
        if cur_interation_num >= max_iteration_num:
            print("MAX ITERATIONS REACHED")
            break

        response = agent_llm.generate_completion(messages)

        if last_message and response.choices[0].message.content == last_message:
            # if the agent's respomses repeat, try to nudge it in the right direction
            messages.append({
                "role": "user",
                "content": "OBSERVATION:\n" + "Don't repeat your response.\n" + fake_user_msg,
            })
            continue

        # Store the current response content
        current_message = response.choices[0].message.content

        if current_message == "":
            # Detected cases where agent's context window is overloaded and prevented from responding further
            empty_message_count += 1
            if empty_message_count >= 4:
                print("AGENT'S CONTEXT WINDOW OVERLOADED. UNABLE TO COMPLETE THE TASK.")
                fail = True
                break
        
        # If the current message has substantive content, keep track of it
        if current_message and len(current_message.strip()) > 20:
            previous_substantive_message = current_message

        last_message = response.choices[0].message.content
        print(response.choices[0].message)
        messages.append(convert_to_json(response.choices[0].message))

        actions = parser.parse(response)

        if not isinstance(actions, List):
            actions = [actions]
        for action in actions:
            if action.action_type == ActionType.FINISH:
                message_stall_count = 0
                # Check if this is just a finish flag with minimal content
                if action.thought.strip() in ["", "<finish></finish>"] or len(action.thought.strip()) < 10:
                    if previous_substantive_message:
                        # Use the previous substantive message instead
                        print("Detected empty finish flag after substantive message - using previous content")
                        final_output = previous_substantive_message
                    else:
                        final_output = action.thought
                else:
                    final_output = action.thought
                finish = True # break
            elif action.action_type == ActionType.MESSAGE:
                message_stall_count += 1
                # if too many messages without action calls are detected, the agent likely needs a little help.
                if message_stall_count >= MESSAGE_STALL_THRESHOLD:
                    messages.append({"role": "user", "content": FAKE_USER_MSG_FOR_CLARIFICATION})
                    message_stall_count = 0
                # continue
            elif action.action_type == ActionType.RUN_IPYTHON:
                message_stall_count = 0
                # isolate the code to be executed
                ipython_code = action.code.strip('`')

                # execute the tool call
                function_response = execute_ipython(ipython_code)
                try:
                    function_response = eval(function_response)
                except SyntaxError:
                    function_response = function_response
                if not isinstance(function_response, str):
                    function_response = str(function_response)

                
                # if not tools:

                # append the function response to the agent's conversation as an observation
                messages.append({
                    "role": "user",
                    "content": "OBSERVATION:\n" + function_response,
                })

                # else:
                #     messages.append({
                #         "role": "tool",
                #         "tool_call_id": action.tool_call_id,
                #         "name": action.function_name,
                #         "content": "OBSERVATION:\n" + function_response,
                #     })

            else:
                print("ACTION TYPE ERROR")


    # store the conversation state for possible continuation
    agent_llm.messages = messages
    if fail:
        return "Agent FAILED to complete the task."
    return final_output
