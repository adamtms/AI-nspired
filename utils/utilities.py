import pandas as pd
from collections import defaultdict
import os

class DefaultDict(defaultdict):
    def __missing__(self, key):
        return self.default_factory(key)

def get_groups():
    groups = defaultdict(lambda: {
        'group_code': None,
        'people_ID': [],
        'ai_images': [],
        'web_images': [],
        'final_submissions': []
    })
    participants = get_participants()
    for _, row in participants.iterrows():
        group_code = int(row['GroupID'])
        groups[group_code]['group_code'] = group_code
        groups[group_code]['people_ID'].append(row['ID'])
        groups[group_code]['ai_images'].extend(row['AI_inspirations'])
        groups[group_code]['web_images'].extend(row['WEB_inspirations'])
        groups[group_code]['final_submissions'] = list(set(groups[group_code]['final_submissions']) | set(row['Final_submisions']))
    return pd.DataFrame.from_dict(groups, orient='index')

def get_participants():
    participants = pd.read_excel('./data/Participants.xlsx', header=0, skiprows=[0], index_col=0)
    participants.columns = ["Unnamed: 1", "School_group", "ID", "GroupID", "Name", "WEB_inspirations", "AI_inspirations", "Matrices", "Comment"]
    participants["GroupID"] = participants["ID"].apply(lambda x: str(x) if str(x).isdigit() else str(x).rstrip("ABCD"))
    participants["WEB_inspirations"] = participants.apply(lambda row: list_files_with_prefix(["data", "web"], f'{row["ID"]}'), axis=1)
    participants["AI_inspirations"] = participants.apply(lambda row: list_files_with_prefix(["data", "ai"], f'{row["ID"]}'), axis=1)
    participants["Final_submisions"] = participants.apply(lambda row: list_files_with_prefix(["data", "final_submissions", f'{row["GroupID"]}'], ''), axis=1)
    return participants[["Name", "ID", "GroupID", "WEB_inspirations", "AI_inspirations", "Final_submisions", "Matrices", "Comment"]]

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