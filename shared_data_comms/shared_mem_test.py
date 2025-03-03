import threading
from shared_mem_com import SharedMemory, CommunicationLeader, Agent
import queue
import time

def run_leader(shared_memory, agent_queues):
    """Starts the communication leader."""
    leader = CommunicationLeader(shared_memory, agent_queues)
    leader.start()
    return leader

def run_agents(shared_memory, agent_queues, model_path):
    """Starts the agents."""
    agents = [Agent(name, shared_memory, model_path) for name in agent_queues.keys()]
    for agent in agents:
        agent.start()
    return agents

def send_task_to_shared_memory(shared_memory, content, target_agent):
    """Send a test task to the shared memory."""
    message = {
        "type": "task",
        "target_agent": target_agent,
        "content": content
    }
    shared_memory.write_message(message)
    print(f"[TEST] Task added to shared memory: {message}")
    print(f"[TEST] Current shared memory state: {shared_memory.messages}")

def test_shared_memory_system():
    """Test the shared memory communication system."""
    # shared_memory = SharedMemory()
    # agent_queues = {"intent_manager": queue.Queue()}
    model_path = "/mnt/d/wsl_workspace/fine_tuned_bert_v4"  # Replace with actual model path

    # # Start the communication leader
    # leader = run_leader(shared_memory, agent_queues)

    # # Start the agents
    # agents = run_agents(shared_memory, agent_queues, model_path)

    # time.sleep(1)  # Allow time for initialization

    # # Add a test task to the shared memory
    # send_task_to_shared_memory(shared_memory, "Refactor data collector files and implement processing and storing data", "intent_manager")

    # # Allow time for processing
    # try:
    #     time.sleep(5)  # Wait for the task to be processed
    # finally:
    #     # Shutdown the system
    #     leader.shutdown()
    #     for agent in agents:
    #         agent.shutdown()
    #     leader.join()
    #     for agent in agents:
    #         agent.join()

    # Initialize shared memory
    shared_memory = SharedMemory()

    # Create agents first
    agents = {
        "intent_manager": Agent("intent_manager", shared_memory, model_path),
    }

    # Start agents
    for agent in agents.values():
        agent.start()

    # Create agent_queues dictionary using agents' task_queue references
    agent_queues = {name: agent.task_queue for name, agent in agents.items()}

    # Create and start the Communication Leader
    leader = CommunicationLeader(shared_memory, agent_queues)
    leader.start()

    # Example interaction
    agents["intent_manager"].send_message("Refactor data collector files and implement processing and storing data", target_agent="intent_manager", message_type="task")

    print("[Main] Active threads before shutdown:", threading.enumerate())

    # Allow time for processing
    try:
        time.sleep(5)
    finally:
        # Shutdown all threads
        leader.shutdown()
        for agent in agents.values():
            agent.shutdown()
        print("[Main] All are shutdown")
        leader.join()
        print("[Main] Leader joined")
        for agent in agents.values():
            agent.join()
        print("[Main] Active threads after join attempt:", threading.enumerate())


if __name__ == "__main__":
    test_shared_memory_system()
