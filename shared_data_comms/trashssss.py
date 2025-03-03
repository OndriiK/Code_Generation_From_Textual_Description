import threading

def producer(condition, messages):
    with condition:
        messages.append("Message from producer")
        print("[PRODUCER] Message added. Notifying...")
        condition.notify()

def consumer(condition, messages):
    with condition:
        while not messages:
            print("[CONSUMER] Waiting for messages...")
            condition.wait()
        print("[CONSUMER] Got message:", messages.pop(0))

if __name__ == "__main__":
    condition = threading.Condition()
    messages = []

    consumer_thread = threading.Thread(target=consumer, args=(condition, messages))
    producer_thread = threading.Thread(target=producer, args=(condition, messages))

    consumer_thread.start()
    producer_thread.start()

    consumer_thread.join()
    producer_thread.join()
