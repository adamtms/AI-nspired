import pandas as pd
from collections import defaultdict
import os

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
        groups[group_code]['people_ID'].append(row['ParticipantID'])
        groups[group_code]['ai_images'].extend(row['AI_inspirations'])
        groups[group_code]['web_images'].extend(row['WEB_inspirations'])
        groups[group_code]['final_submissions'] = list(set(groups[group_code]['final_submissions']) | set(row['Final_submissions']))
    
    return pd.DataFrame.from_dict(groups, orient='index')

def get_participants():
    participants = pd.read_csv('data/Participants_PL_EN.csv')
    participants.columns = ["[PL] KOD", "ParticipantID","[PL] Inspiracje WEB","WEB_inspirations","[PL] Inspiracje AI","AI_inspirations","[PL] Matryce","Matrices"]
    
    # Ensure GroupID is properly extracted
    participants["GroupID"] = participants["ParticipantID"].apply(lambda x: str(x) if str(x).isdigit() else str(x).rstrip("ABCD"))
    
    # Adapt paths to new structure: data/[group_folder]/[ai, web, final]/
    participants["WEB_inspirations"] = participants.apply(lambda row: list_files_with_prefix(["data", row["GroupID"], "web"], f'{row["ParticipantID"]}'), axis=1)
    participants["AI_inspirations"] = participants.apply(lambda row: list_files_with_prefix(["data", row["GroupID"], "ai"], f'{row["ParticipantID"]}'), axis=1)
    participants["Final_submissions"] = participants.apply(lambda row: list_files_with_prefix(["data", row["GroupID"], "final"], ''), axis=1)

    return participants[["ParticipantID", "GroupID", "WEB_inspirations", "AI_inspirations", "Final_submissions", "Matrices"]]

def list_files_with_prefix(directory, prefix):
    try:
        dir_path = os.path.join(*directory)
        if not os.path.exists(dir_path):
            return []

        # List all files in the directory
        files = os.listdir(dir_path)
        
        # Filter files that start with the given prefix
        matching_files = [os.path.join(dir_path, file) for file in files if file.startswith(prefix)]
        
        return matching_files
    except Exception as e:
        return f"An error occurred: {e}"
