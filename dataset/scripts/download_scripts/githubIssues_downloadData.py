import requests
import json
import time

# GitHub API configurations
GITHUB_SEARCH_URL = "https://api.github.com/search/repositories"
GITHUB_ISSUES_URL = "https://api.github.com/repos/{owner}/{repo}/issues"
HEADERS = {
    "Authorization": "token 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",  # Replace with your token
    "Accept": "application/vnd.github.v3+json"
}

# Predefined intents and related GitHub labels
intent_labels = {
    "debug_code": ["bug", "debugging"],
    "add_feature": ["enhancement", "feature request"],
    "write_tests": ["testing", "unit tests"],
    "optimize_performance": ["performance", "optimization"],
    "document_code": ["documentation", "docs"]
}

# Map labels to intents
def map_labels_to_intent(labels):
    for intent, related_labels in intent_labels.items():
        if any(label.get("name", "").lower() in related_labels for label in labels):
            return intent
    return None

# Fetch repositories based on a search query
def fetch_repositories(query, max_repos=10):
    print(f"Searching for repositories with query: {query}")
    params = {"q": query, "sort": "stars", "order": "desc", "per_page": max_repos}
    response = requests.get(GITHUB_SEARCH_URL, headers=HEADERS, params=params)
    if response.status_code != 200:
        print(f"Failed to fetch repositories: {response.status_code}")
        return []
    repos = response.json().get("items", [])
    return [{"owner": repo["owner"]["login"], "name": repo["name"]} for repo in repos]

# Fetch issues for a given repository
def fetch_issues(owner, repo, max_issues=50):
    print(f"Fetching issues for repository: {owner}/{repo}")
    dataset = []
    for page in range(1, (max_issues // 50) + 2):  # Adjust pagination
        url = GITHUB_ISSUES_URL.format(owner=owner, repo=repo)
        params = {"state": "all", "per_page": 50, "page": page}
        response = requests.get(url, headers=HEADERS, params=params)
        if response.status_code != 200:
            print(f"Failed to fetch issues for {owner}/{repo}: {response.status_code}")
            break
        issues = response.json()
        for issue in issues:
            if "pull_request" in issue:  # Skip pull requests
                continue
            # intent = map_labels_to_intent(issue.get("labels", []))
            # if intent:
            #     dataset.append({"text": issue["title"], "label": intent})
            print(issue["title"])
            print("\n")
        time.sleep(1)  # To avoid hitting rate limits
    return dataset

# Main function: Batch process multiple repositories
def batch_process_repositories(query, max_repos=10, max_issues_per_repo=50, output_file="github_dataset.json"):
    repos = fetch_repositories(query, max_repos=max_repos)
    all_data = []
    for repo in repos:
        data = fetch_issues(repo["owner"], repo["name"], max_issues=max_issues_per_repo)
        all_data.extend(data)
    with open(output_file, "w") as f:
        json.dump(all_data, f, indent=4)
    print(f"Dataset saved to {output_file}")

# Execute the batch processing
if __name__ == "__main__":
    # Adjust the search query to find random repositories
    batch_process_repositories(query="topic:machine-learning", max_repos=10, max_issues_per_repo=50)
