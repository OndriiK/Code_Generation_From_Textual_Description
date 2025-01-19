import requests
import os
from datetime import datetime

# GitHub API Token and Headers
GITHUB_TOKEN = "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"
HEADERS = {
    "Authorization": f"token {GITHUB_TOKEN}",
    "Accept": "application/vnd.github.v3+json"
}

# Function to search repositories
def search_repositories(query, language="Python", per_page=10, page=1):
    url = f"https://api.github.com/search/repositories?q={query}+language:{language}&sort=stars&order=desc&per_page={per_page}&page={page}"
    response = requests.get(url, headers=HEADERS)
    if response.status_code == 200:
        return response.json().get("items", [])
    else:
        print(f"Failed to search repositories: {response.status_code}, {response.text}")
        return []

# Function to fetch commits from a repository
def download_repository_files(repo_full_name, path="", save_path="repositories"):
    try:
        url = f"https://api.github.com/repos/{repo_full_name}/contents/{path}"
        response = requests.get(url, headers=HEADERS)
        
        if response.status_code == 200:
            repo_contents = response.json()
            os.makedirs(save_path, exist_ok=True)
            
            for item in repo_contents:
                try:
                    if item["type"] == "file" and item["name"].endswith(".py"):
                        file_url = item["download_url"]
                        file_response = requests.get(file_url)
                        if file_response.status_code == 200:
                            file_path = os.path.join(save_path, item["path"])
                            os.makedirs(os.path.dirname(file_path), exist_ok=True)
                            with open(file_path, "w", encoding="utf-8") as file:
                                file.write(file_response.text)
                                print(f"Downloaded: {file_path}")
                        else:
                            print(f"Failed to download file: {item['name']} - Status Code: {file_response.status_code}")
                    elif item["type"] == "dir":
                        # Recursive call for subdirectories
                        download_repository_files(repo_full_name, path=item["path"], save_path=save_path)
                except Exception as e:
                    print(f"Error processing item '{item['name']}': {e}")
        else:
            print(f"Failed to fetch repository contents at {path}: {response.status_code}, {response.text}")
    except Exception as e:
        print(f"Error accessing repository contents for '{repo_full_name}' at path '{path}': {e}")


# Fetch commits with error handling
def fetch_commits(repo_full_name, per_page=10):
    try:
        url = f"https://api.github.com/repos/{repo_full_name}/commits?per_page={per_page}"
        response = requests.get(url, headers=HEADERS)
        if response.status_code == 200:
            return response.json()
        else:
            print(f"Failed to fetch commits for {repo_full_name}: {response.status_code}, {response.text}")
            return []
    except Exception as e:
        print(f"Error fetching commits for '{repo_full_name}': {e}")
        return []


def get_commits(repo_full_name, file_path, per_page=10):
    """Fetch commit history for a specific file."""
    try:
        url = f"https://api.github.com/repos/{repo_full_name}/commits?path={file_path}&per_page={per_page}"
        response = requests.get(url, headers=HEADERS)
        if response.status_code == 200:
            return response.json()
        else:
            print(f"Failed to fetch commits for {file_path}: {response.status_code}, {response.text}")
            return []
    except Exception as e:
        print(f"Error fetching commits for file '{file_path}': {e}")
        return []

def get_commit_diff(repo_full_name, commit_sha):
    """Fetch the diff/patch for a specific commit."""
    try:
        url = f"https://api.github.com/repos/{repo_full_name}/commits/{commit_sha}"
        response = requests.get(url, headers=HEADERS)
        if response.status_code == 200:
            commit_data = response.json()
            # Return patch data if available
            if "files" in commit_data:
                return commit_data["files"]
        else:
            print(f"Failed to fetch commit diff for {commit_sha}: {response.status_code}, {response.text}")
            return []
    except Exception as e:
        print(f"Error fetching diff for commit '{commit_sha}': {e}")
        return []

def process_repository(repo_full_name, save_path="error_fix_data"):
    """Process a repository to extract error-fix pairs."""
    try:
        # Fetch repository contents
        url = f"https://api.github.com/repos/{repo_full_name}/contents"
        response = requests.get(url, headers=HEADERS)
        if response.status_code != 200:
            print(f"Failed to fetch repository contents for {repo_full_name}: {response.status_code}")
            return

        repo_contents = response.json()
        for item in repo_contents:
            if item["type"] == "file" and item["name"].endswith(".py"):
                file_path = item["path"]
                print(f"Processing file: {file_path}")

                # Get commit history for the file
                commits = get_commits(repo_full_name, file_path, per_page=10)
                for commit in commits:
                    try:
                        commit_message = commit["commit"]["message"]
                        commit_sha = commit["sha"]

                        # Check for fix-related commit
                        if "fix" in commit_message.lower() or "bug" in commit_message.lower():
                            # Fetch diff/patch for the commit
                            diff_files = get_commit_diff(repo_full_name, commit_sha)
                            for diff_file in diff_files:
                                if diff_file["filename"] == file_path and "patch" in diff_file:
                                    # Extract patch (changes)
                                    patch = diff_file["patch"]
                                    print(f"Fix Commit: {commit_sha}")
                                    print(f"Message: {commit_message}")
                                    print(f"Patch:\n{patch}")

                                    # Save structured data
                                    os.makedirs(save_path, exist_ok=True)
                                    output_file = os.path.join(
                                        save_path, f"{repo_full_name.replace('/', '_')}_{commit_sha}.txt"
                                    )
                                    with open(output_file, "w", encoding="utf-8") as f:
                                        f.write(f"File: {file_path}\n")
                                        f.write(f"Commit SHA: {commit_sha}\n")
                                        f.write(f"Commit Message: {commit_message}\n")
                                        f.write(f"Patch:\n{patch}\n")
                                        print(f"Saved data for commit {commit_sha} to {output_file}")
                                    break  # Stop processing once a valid diff is found
                    except Exception as e:
                        print(f"Error processing commit '{commit['sha']}': {e}")
    except Exception as e:
        print(f"Error processing repository '{repo_full_name}': {e}")

# Main script
if __name__ == "__main__":
    query = "machine learning"
    language = "Python"
    max_repositories = 1
    processed_count = 0

    results = search_repositories(query, language, per_page=5)

    for repo in results:
        if processed_count >= max_repositories:
            break

        repo_name = repo["full_name"]
        print(f"Processing repository: {repo_name}")
        process_repository(repo_name, save_path="error_fix_data")
        processed_count += 1