import re

def extract_sentence(text, keyword):
    # Regex to find the sentence starting with the keyword
    pattern = rf'{re.escape(keyword)}.*?(?:[\.:](?=\s|$)|[!?])'

    match = re.search(pattern, text)
    return match.group(0) if match else None

# Example text
text = "Here is how you can do this. First, you need to import myfile.zip into your environment: make sure it's the right version. You are done after that."

keyword = "First, "

# Extract sentence
sentence = extract_sentence(text, keyword)

# Output result
print(sentence)
