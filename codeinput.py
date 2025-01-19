from flask import Flask, request, redirect, url_for, render_template, flash, jsonify
import os
import zipfile
import shutil

# Configuration
UPLOAD_FOLDER = "uploaded_zips"
EXTRACT_FOLDER = "extracted_code"
ALLOWED_EXTENSIONS = {"zip"}

# Initialize Flask app
app = Flask(__name__)
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER
app.secret_key = "your_secret_key"  # Needed for flash messages

def allowed_file(filename):
    """
    Check if the uploaded file has an allowed extension.
    """
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route("/", methods=["GET", "POST"])
def upload_file_and_instructions():
    """
    Handle file uploads and user instructions.
    """
    if request.method == "POST":
        # Check if the POST request has a file part
        if "file" not in request.files:
            flash("No file part")
            return redirect(request.url)

        file = request.files["file"]

        # If no file is selected
        if file.filename == "":
            flash("No selected file")
            return redirect(request.url)

        # Check if the file is allowed
        if file and allowed_file(file.filename):
            filename = file.filename
            save_path = os.path.join(app.config["UPLOAD_FOLDER"], filename)
            os.makedirs(app.config["UPLOAD_FOLDER"], exist_ok=True)
            file.save(save_path)  # Save the file to the upload folder
            flash(f"File {filename} uploaded successfully!")

            # Extract the zip file
            extract_path = extract_zip_file(save_path)

            # Get user instructions from the form
            user_instructions = request.form.get("instructions", "").strip()
            if not user_instructions:
                flash("No instructions provided.")
                return redirect(request.url)

            flash(f"Instructions received: {user_instructions}")

            # Process the extracted files and pass to the LLM
            process_codebase_and_instructions(extract_path, user_instructions)

            return f"File uploaded, instructions processed, and data passed to the LLM. Extracted to {extract_path}"

        flash("Invalid file type. Only .zip files are allowed.")
        return redirect(request.url)

    return '''
    <!doctype html>
    <title>Upload a .zip File and Provide Instructions</title>
    <h1>Upload a .zip File and Provide Instructions</h1>
    <form method=post enctype=multipart/form-data>
      <label for="file">Select a .zip file:</label><br>
      <input type=file name=file><br><br>
      <label for="instructions">Enter your instructions:</label><br>
      <textarea name="instructions" rows="4" cols="50" placeholder="Enter instructions here..."></textarea><br><br>
      <input type=submit value=Upload>
    </form>
    '''

def extract_zip_file(zip_path):
    """
    Extract the uploaded .zip file to a designated directory.
    """
    os.makedirs(EXTRACT_FOLDER, exist_ok=True)
    extract_path = os.path.join(EXTRACT_FOLDER, os.path.splitext(os.path.basename(zip_path))[0])
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(extract_path)
    print(f"Extracted .zip file to: {extract_path}")
    return extract_path

def process_codebase_and_instructions(extract_path, instructions):
    """
    Process the extracted codebase and user instructions.
    """
    # Example: Scan the codebase and prepare data for LLM processing
    code_files = []
    for root, dirs, files in os.walk(extract_path):
        for file in files:
            if file.endswith(".py"):  # Target Python files
                file_path = os.path.join(root, file)
                print(f"Processing file: {file_path}")
                with open(file_path, "r", encoding="utf-8") as f:
                    code = f.read()
                    code_files.append({"path": file_path, "content": code})
                    print(f"Read {len(code)} characters from {file_path}")

    # Example: Pass the data to the LLM instance
    data_to_send = {
        "instructions": instructions,
        "code_files": code_files
    }
    # print("Data to send to LLM:", data_to_send)
    
    # Here, you would call the LLM processing function/API
    # Example: send_to_llm(data_to_send)

@app.route("/clean", methods=["POST"])
def clean_up():
    """
    Clean up uploaded and extracted files (optional endpoint).
    """
    if os.path.exists(UPLOAD_FOLDER):
        shutil.rmtree(UPLOAD_FOLDER)
    if os.path.exists(EXTRACT_FOLDER):
        shutil.rmtree(EXTRACT_FOLDER)
    return "Temporary directories cleaned up."

if __name__ == "__main__":
    app.run(debug=True)
