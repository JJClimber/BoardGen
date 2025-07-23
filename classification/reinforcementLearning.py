from sklearn.linear_model import LinearRegression
from unicodedata import name
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier

import json 
import itertools


def moonboard_cleaning_2019(df):
    '''
    Cleans and processes moonboard data for the 2019 Masters for learning (After already cleaning).
    Parameters:
    df (pd.DataFrame): dataframe containing moonboard data 

    Returns:
    pd.DataFrame: Processed DataFrame ready for analysis.
    '''
    import numpy as np

    # Generate all possible move labels (A1-K18)
    letters = [chr(i) for i in range(ord('A'), ord('K')+1)]
    numbers = [str(i) for i in range(1, 19)]
    all_moves = [f"{l}{n}" for l, n in itertools.product(letters, numbers)]

    # Prepare the new DataFrame
    processed_rows = []
    for _, row in df.iterrows():
        new_row = {
            'name': row['name'],
            'grade': row['grade'],
            'isBenchmark': row['isBenchmark']
        }
        # Initialize all move columns to False
        for move in all_moves:
            new_row[move] = False

        # Count start/end moves
        start_count = 0
        end_count = 0
        for move in row.get('moves', []):
            desc = move.get('description')
            if desc in all_moves:
                new_row[desc] = True
            if move.get('isStart'):
                start_count += 1
            if move.get('isEnd'):
                end_count += 1
        new_row['twoHandStart'] = start_count == 2
        new_row['twoHandEnd'] = end_count == 2
        processed_rows.append(new_row)

    processed_df = pd.DataFrame(processed_rows)

    bool_cols = processed_df.columns[processed_df.columns.get_loc('isBenchmark'):]
    processed_df[bool_cols] = processed_df[bool_cols].astype(int)
    
    return processed_df

def KNearestNeighbors(df, n_neighbors):
    """
    Applies K-Nearest Neighbors algorithm to the data.

    Parameters:
    df (pd.DataFrame): DataFrame containing moonboard data.
    n_neighbors (int): Number of neighbors to use for KNN.

    Returns:
    pd.DataFrame: DataFrame with predictions added.
    """
    X = df.drop(columns=['name', 'grade'])
    y = df['grade']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = KNeighborsClassifier(n_neighbors=n_neighbors)
    model.fit(X_train, y_train)

    predictions = model.predict(X_test)
    accuracy = (predictions == y_test).mean()
    print(f"KNN Test Accuracy: {accuracy:.2f}")

    df['predicted_grade'] = model.predict(X)

    return df

# Helper function for linear regression as we need to map grades to numeric values
def grade_to_numeric(df_climbing, grade_map):
    """
    Maps the 'grade' column in the DataFrame to numeric values using the provided grade_map.
    Returns a new DataFrame with a 'grade_numeric' column.
    """
    df = df_climbing.copy()
    df['grade'] = df_climbing['grade'].map(grade_map)
    return df

def numeric_to_grade(nums, grade_map):
    inv_map = {v: k for k, v in grade_map.items()}
    return [inv_map.get(min(inv_map, key=lambda x: abs(x-n)), 'Unknown') for n in nums]

def linearRegression(df, grade_map):
    """
    Applies Linear Regression to the moonboard data to predict grades.

    Parameters:
    df (pd.DataFrame): DataFrame containing moonboard data.
    grade_map (dict): Mapping of grades to numeric values.

    Returns:
    pd.DataFrame: DataFrame with predictions added.
    """
    mapping_df = grade_to_numeric(df, grade_map)
    X = mapping_df.drop(columns=['name', 'grade'])
    Y = mapping_df['grade']

    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

    model = LinearRegression()
    model.fit(X_train, y_train)

    predictions = model.predict(X_test)
    pred_grades = numeric_to_grade(predictions, grade_map)
    true_grades = numeric_to_grade(y_test, grade_map)
    accuracy = sum([p == t for p, t in zip(pred_grades, true_grades)]) / len(y_test)
    print(f"Linear Regression Test Accuracy (rounded to nearest grade): {accuracy:.2f}")

    df['predicted_grade_regression'] = numeric_to_grade(model.predict(X), grade_map)

    return df


if __name__ == "__main__":
    with open(r"C:\Users\Jason\Documents\VSCode\BoardGen\moonboard_problems\cleaned_data\masters2019-40_cleaned.json", "r", encoding="utf-8") as f:
        data = json.load(f)
    moonboard_df = pd.DataFrame(data["data"])
    moonboard_df = moonboard_cleaning_2019(moonboard_df)

    # Drop grades 8A+, 8B, 8B+ as there are too few examples
    rare_grades = [ "8A+", "8B", "8B+"]
    moonboard_df = moonboard_df[~moonboard_df['grade'].isin(rare_grades)].reset_index(drop=True)

    # print(moonboard_df['grade'].value_counts())
    # print(moonboard_df.head(20))

    # KNN  
    '''
    result_df = KNearestNeighbors(moonboard_df, n_neighbors=10)
    cols_to_show = list(result_df.columns[:3]) + [result_df.columns[-1]]
    print(result_df[cols_to_show].head())
    '''

    grade_map = {
    "6A+": 0.5, "6B": 1, "6B+": 1.5, "6C": 2, "6C+": 2.5,
    "7A": 3, "7A+": 3.5, "7B": 4, "7B+": 4.5, "7C": 5,
    "7C+": 5.5, "8A": 6}

    '''
    # Linear Regression
    result_df = linearRegression(moonboard_df, grade_map)
    cols_to_show_reg = list(result_df.columns[:3]) + [result_df.columns[-1]]
    print(result_df[cols_to_show_reg].head())
    '''