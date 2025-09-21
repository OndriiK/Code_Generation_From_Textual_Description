import tkinter as tk
from tkinter import filedialog
import sys

root = tk.Tk()
root.withdraw()
root.wm_attributes("-topmost", 1)

# Open the directory selection dialog
folder_path = filedialog.askdirectory(master=root)

# Destroy the root window after selection
root.destroy()

# Print the path so it can be used in the calling script
if folder_path:
    print(folder_path)
    sys.exit(0) # Exit with success code
else:
    sys.exit(1) # Exit with error code if cancelled

