import requests
import json
import random

# API Configuration
API_URL = "https://api.stackexchange.com/2.3/questions"
PARAMS = {
    "order": "desc",
    "sort": "votes",
    "tagged": "python",  # Adjust tags for specific intents
    "site": "stackoverflow",
    "pagesize": 100  # Number of questions per request
}

# Predefined intents and their related tags
intent_tags = {
    "debug_code": ["debugging", "error", "bug"],
    "add_feature": ["feature", "implementation", "enhancement"],
    "write_tests": ["testing", "unit-testing", "test"],
    "optimize_performance": ["optimization", "performance", "speed"],
    "document_code": ["documentation", "readme", "comments"]
}

# Helper function to map tags to intents
def map_tags_to_intent(tags):
    for intent, related_tags in intent_tags.items():
        if any(tag in related_tags for tag in tags):
            return intent
    return None  # If no tag matches, skip

# Fetch data from Stack Overflow
def fetch_stackoverflow_data():
    dataset = []
    for page in range(1, 21):  # Fetch 20 pages of data
        print(f"Fetching page {page}...")
        PARAMS["page"] = page
        response = requests.get(API_URL, params=PARAMS)
        if response.status_code != 200:
            print("Failed to fetch data:", response.status_code)
            continue
        data = response.json().get("items", [])
        for item in data:
            # intent = map_tags_to_intent(item.get("tags", []))
            # if intent:
            #     dataset.append({"text": item["title"], "label": intent})
            print(item["title"])
            print("\n")
    return dataset

# Save dataset to a JSON file
def save_dataset(dataset, file_name="stackoverflow_intent_dataset.json"):
    with open(file_name, "w") as f:
        json.dump(dataset, f, indent=4)
    print(f"Dataset saved to {file_name}")

# Main function
if __name__ == "__main__":
    dataset = fetch_stackoverflow_data()
    # save_dataset(dataset)
