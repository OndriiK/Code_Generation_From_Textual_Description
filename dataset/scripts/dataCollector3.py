import os
import time
import requests
from datetime import datetime
import logging

# Configure logging for the script
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

# Retrieve GitHub API token from environment variable
GITHUB_TOKEN = "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"
if not GITHUB_TOKEN:
    raise ValueError("GitHub token not found in environment variables.")  # Ensure secure handling of the token

# Headers for GitHub API requests
HEADERS = {
    "Authorization": f"token {GITHUB_TOKEN}",  # Authenticate requests
    "Accept": "application/vnd.github.v3+json"  # Specify API version
}

# Constants for API rate limits
RATE_LIMIT_SLEEP = 60  # Sleep duration in seconds if rate limit is exceeded
MAX_RETRIES = 5  # Maximum number of retries for API requests

# Function to handle rate limits
# If the API returns a 403 due to rate limiting, this function calculates sleep time and pauses execution.
def handle_rate_limit(response):
    if response.status_code == 403 and "X-RateLimit-Remaining" in response.headers:
        remaining = int(response.headers.get("X-RateLimit-Remaining", 0))
        if remaining == 0:
            reset_time = int(response.headers.get("X-RateLimit-Reset", time.time()))
            sleep_time = max(0, reset_time - time.time())
            logging.warning(f"Rate limit exceeded. Sleeping for {sleep_time:.2f} seconds.")
            time.sleep(sleep_time)

# Function to handle API requests with retries
# Implements exponential backoff for handling temporary API failures.
def api_request(url, method="GET", retries=MAX_RETRIES, **kwargs):
    for attempt in range(retries):
        try:
            response = requests.request(method, url, headers=HEADERS, **kwargs)
            if response.status_code == 403:
                handle_rate_limit(response)  # Handle rate-limiting issues
                continue
            elif response.status_code >= 400:
                logging.error(f"Error {response.status_code}: {response.text}")  # Log client/server errors
            else:
                return response  # Successful response
        except requests.RequestException as e:
            logging.error(f"Request error: {e}")  # Log network-related errors
        sleep_time = 2 ** attempt  # Exponential backoff for retries
        logging.info(f"Retrying in {sleep_time} seconds...")
        time.sleep(sleep_time)
    raise Exception(f"Failed to fetch {url} after {retries} retries.")  # Raise exception after exhausting retries

# Function to search for repositories matching a query
def search_repositories(query, language="Python", per_page=10, page=1):
    url = f"https://api.github.com/search/repositories?q={query}+language:{language}&sort=stars&order=desc&per_page={per_page}&page={page}"
    response = api_request(url)
    return response.json().get("items", []) if response else []  # Return the list of repositories

# Function to fetch and save repository files (Python scripts)
def download_repository_files(repo_full_name, path="", save_path="repositories"):
    url = f"https://api.github.com/repos/{repo_full_name}/contents/{path}"
    response = api_request(url)
    if response.status_code == 200:
        repo_contents = response.json()
        os.makedirs(save_path, exist_ok=True)

        for item in repo_contents:
            try:
                if item["type"] == "file" and item["name"].endswith(".py"):  # Target only Python files
                    file_url = item["download_url"]
                    file_response = api_request(file_url)
                    if file_response.status_code == 200:
                        file_path = os.path.join(save_path, item["path"])
                        os.makedirs(os.path.dirname(file_path), exist_ok=True)
                        with open(file_path, "w", encoding="utf-8") as file:
                            file.write(file_response.text)
                            logging.info(f"Downloaded: {file_path}")  # Log successful downloads
                elif item["type"] == "dir":
                    # Recursively handle directories
                    download_repository_files(repo_full_name, path=item["path"], save_path=save_path)
            except Exception as e:
                logging.error(f"Error processing item '{item['name']}': {e}")  # Handle file-specific errors

# Fetch commits for a repository (limited to the most recent ones)
def fetch_commits(repo_full_name, per_page=10):
    url = f"https://api.github.com/repos/{repo_full_name}/commits?per_page={per_page}"
    response = api_request(url)
    return response.json() if response else []  # Return commit details

# Fetch commit history for a specific file in the repository
def get_commits(repo_full_name, file_path, per_page=10):
    url = f"https://api.github.com/repos/{repo_full_name}/commits?path={file_path}&per_page={per_page}"
    response = api_request(url)
    return response.json() if response else []  # Return list of commits for the file

# Fetch details and patch information for a specific commit
def get_commit_diff(repo_full_name, commit_sha):
    url = f"https://api.github.com/repos/{repo_full_name}/commits/{commit_sha}"
    response = api_request(url)
    if response.status_code == 200:
        commit_data = response.json()
        return commit_data.get("files", [])  # Return list of files affected by the commit
    return []

# Process a repository to extract error-fix commit pairs with structured output
def process_repository(repo_full_name, save_path="error_fix_data"):
    url = f"https://api.github.com/repos/{repo_full_name}/contents"
    response = api_request(url)
    if not response or response.status_code != 200:
        return

    repo_contents = response.json()
    for item in repo_contents:
        if item["type"] == "file" and item["name"].endswith(".py"):  # Target only Python files
            file_path = item["path"]
            logging.info(f"Processing file: {file_path}")

            commits = get_commits(repo_full_name, file_path)
            for commit in commits:
                try:
                    commit_message = commit["commit"]["message"]
                    commit_sha = commit["sha"]

                    # Look for commits with fix-related keywords
                    if "fix" in commit_message.lower() or "bug" in commit_message.lower():
                        diff_files = get_commit_diff(repo_full_name, commit_sha)
                        for diff_file in diff_files:
                            if diff_file["filename"] == file_path and "patch" in diff_file:
                                patch = diff_file["patch"]
                                logging.info(f"Fix Commit: {commit_sha}, Message: {commit_message}")

                                os.makedirs(save_path, exist_ok=True)
                                output_file = os.path.join(
                                    save_path, f"{repo_full_name.replace('/', '_')}_{commit_sha}.json"
                                )
                                # Save structured output in JSON format
                                structured_data = {
                                    "file": file_path,
                                    "commit_sha": commit_sha,
                                    "commit_message": commit_message,
                                    "patch": patch
                                }
                                with open(output_file, "w", encoding="utf-8") as f:
                                    import json
                                    json.dump(structured_data, f, indent=4)
                                    logging.info(f"Saved structured data for commit {commit_sha} to {output_file}")
                                break  # Stop after processing the first valid diff
                except Exception as e:
                    logging.error(f"Error processing commit '{commit['sha']}': {e}")  # Handle commit-specific errors


def fetch_commits(repo_full_name, per_page=10):
    url = f"https://api.github.com/repos/{repo_full_name}/commits?per_page={per_page}"
    response = api_request(url)
    return response.json() if response else []


# Main script execution starts here
if __name__ == "__main__":
    query = "machine learning"  # Query for searching repositories
    language = "Python"  # Target programming language
    max_repositories = 1  # Limit on the number of repositories to process
    processed_count = 0

    # Search for repositories matching the query
    results = search_repositories(query, language, per_page=5)

    for repo in results:
        if processed_count >= max_repositories:
            break

        repo_name = repo["full_name"]
        logging.info(f"Processing repository: {repo_name}")
        
        commits = fetch_commits(repo_name)
        for commit in commits:
            commit_message = commit["commit"]["message"]
            print(f"Commit message: {commit_message}")

        process_repository(repo_name, save_path="error_fix_data")  # Process each repository
        processed_count += 1
