import os
import requests
import logging
import json
import time

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

# GitHub API token
GITHUB_TOKEN = os.getenv("GITHUB_TOKEN")
if not GITHUB_TOKEN:
    raise ValueError("GitHub token not found in environment variables.")

HEADERS = {
    "Authorization": f"token {GITHUB_TOKEN}",
    "Accept": "application/vnd.github.v3+json"
}

RATE_LIMIT_SLEEP = 60
MAX_RETRIES = 5

def handle_rate_limit(response):
    """Handle API rate limits."""
    if response.status_code == 403 and "X-RateLimit-Remaining" in response.headers:
        remaining = int(response.headers.get("X-RateLimit-Remaining", 0))
        if remaining == 0:
            reset_time = int(response.headers.get("X-RateLimit-Reset", time.time()))
            sleep_time = max(0, reset_time - time.time())
            logging.warning(f"Rate limit exceeded. Sleeping for {sleep_time:.2f} seconds.")
            time.sleep(sleep_time)

def api_request(url, method="GET", retries=MAX_RETRIES, **kwargs):
    """Send a GitHub API request with retries."""
    for attempt in range(retries):
        try:
            response = requests.request(method, url, headers=HEADERS, **kwargs)
            if response.status_code == 403:
                handle_rate_limit(response)
                continue
            elif response.status_code >= 400:
                logging.error(f"Error {response.status_code}: {response.text}")
            else:
                return response
        except requests.RequestException as e:
            logging.error(f"Request error: {e}")
        sleep_time = 2 ** attempt
        logging.info(f"Retrying in {sleep_time} seconds...")
        time.sleep(sleep_time)
    raise Exception(f"Failed to fetch {url} after {retries} retries.")

def fetch_repositories(query, language="Python", per_page=10, page=1):
    """Search repositories matching the query."""
    url = f"https://api.github.com/search/repositories?q={query}+language:{language}&sort=stars&order=desc&per_page={per_page}&page={page}"
    response = api_request(url)
    return response.json().get("items", []) if response else []

def fetch_commit_messages(repo_full_name, per_page=150):
    """Fetch commit messages from a repository."""
    url = f"https://api.github.com/repos/{repo_full_name}/commits?per_page={per_page}"
    response = api_request(url)
    commits = response.json() if response else []
    messages = []
    
    # Filter and collect commit messages
    for commit in commits:
        try:
            commit_message = commit["commit"]["message"].strip()
            if any(keyword in commit_message.lower() for keyword in ["debug", "fix", "optimize", "document", "refactor"]):
                messages.append(commit_message)
        except KeyError:
            logging.warning(f"Missing commit message in commit: {commit}")
    return messages

if __name__ == "__main__":
    query = "blockchain"  # Search query for repositories
    language = "Python"  # Programming language filter
    max_repositories = 20  # Number of repositories to process
    processed_count = 0

    # Directory to save commit messages
    save_path = "/mnt/d/wsl_workspace/commit_messages2/blockchain"
    os.makedirs(save_path, exist_ok=True)

    # Fetch repositories
    repos = fetch_repositories(query, language, per_page=5)
    
    for repo in repos:
        if processed_count >= max_repositories:
            break

        repo_name = repo["full_name"]
        logging.info(f"Fetching commit messages from repository: {repo_name}")
        
        # Fetch and save commit messages
        commit_messages = fetch_commit_messages(repo_name)
        if commit_messages:
            output_file = os.path.join(save_path, f"{repo_name.replace('/', '_')}_commits.json")
            with open(output_file, "w", encoding="utf-8") as f:
                json.dump(commit_messages, f, indent=4)
            logging.info(f"Saved {len(commit_messages)} commit messages to {output_file}.")
        
        processed_count += 1
