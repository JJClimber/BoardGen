### Function to clean and process moonboard data 
import json

def clean_moonboard_data(data):
    """
    Cleans and processes moonboard data.

    Parameters:
    data (list of dict): .json file to clean and process moonboard data. 

    Returns:
    list of dict: .json cleaned and processed moonboard data.
    """

    cleaned_data = []

    for entry in data:
        # Only keep entries with required fields
        if all(k in entry for k in ['name', 'grade', 'userGrade', 'userRating', 'repeats', 'isBenchmark', 'moves']):
            cleaned_entry = {
                'name': entry['name'].strip() if isinstance(entry['name'], str) else entry['name'],
                'grade': entry['grade'].strip() if isinstance(entry['grade'], str) else entry['grade'],
                'userGrade': entry['userGrade'].strip() if isinstance(entry['userGrade'], str) else entry['userGrade'],
                'userRating': entry['userRating'],
                'repeats': entry['repeats'],
                'isBenchmark': entry['isBenchmark'],
                'moves': entry['moves']
            }
            cleaned_data.append(cleaned_entry)

    return cleaned_data

# Testing function
if __name__ == "__main__":
    with open("moonboard_problems/masters/masters2019-40.json", "r", encoding="utf-8") as f:
        moonboard_data = json.load(f)
    cleaned_data = clean_moonboard_data(moonboard_data["data"])
    output = json.dumps(cleaned_data, indent=2)
    print('\n'.join(output.splitlines()[:20]))

    # Save cleaned data to a JSON file
    # with open("masters2019-40_cleaned.json", "w", encoding="utf-8") as out_f:
    #    json.dump({"data": cleaned_data}, out_f, indent=2, ensure_ascii=False)