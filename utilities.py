import pandas as pd
import os

def get_participants():
    participants = pd.read_excel('./data/Participants.xlsx', header=0, skiprows=[0], index_col=0)
    participants.columns = ["Unnamed: 1", "School_group", "ID", "GroupID", "Name", "WEB_inspirations", "AI_inspirations", "Comment"]
    participants["GroupID"] = participants["ID"].apply(lambda x: str(x) if str(x).isdigit() else str(x).rstrip("AB"))
    participants["WEB_inspirations"] = participants.apply(lambda row: list_files_with_prefix(["data", "extracted_photos", "web"], f'{row["ID"]}'), axis=1)
    participants["AI_inspirations"] = participants.apply(lambda row: list_files_with_prefix(["data", "extracted_photos", "ai"], f'{row["ID"]}'), axis=1)
    participants["Final_submisions"] = participants.apply(lambda row: list_files_with_prefix(["data", "extracted_photos", "final_submissions", f'{row["GroupID"]}'], ''), axis=1)
    return participants[["Name", "ID", "GroupID", "WEB_inspirations", "AI_inspirations", "Final_submisions", "Comment"]]

def list_files_with_prefix(directory, prefix):
    try:
        # List all files in the directory
        files = os.listdir(os.path.join(*directory))
        
        # Filter files that start with the given prefix
        matching_files = [os.path.join(*directory, file) for file in files if file.startswith(prefix)]
        
        return matching_files
    except FileNotFoundError:
        return []
    except Exception as e:
        return f"An error occurred: {e}"
