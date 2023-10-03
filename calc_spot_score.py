# !pip install -q condacolab
# import condacolab
# condacolab.install()
# !conda install -c conda-forge spot

import spot
import pandas as pd
import os

def load_data(filename):
    # Construct the full path to the input text file
    input_file_path = os.path.join(input_folder, filename)

    # Create empty lists to store data from each section
    speech = []
    ltl = []
    predictions = []
    references = []

    # Read the text file line by line
    with open(input_file_path, 'r') as file:
        lines = file.readlines()

    # Initialize variables to None
    current_speech = None
    current_ltl = None
    current_predictions = None
    current_references = None

    # Iterate through the lines of the file
    for line in lines:
        line = line.strip()  # Remove leading/trailing whitespace
        if line.startswith("Speech:"):
            current_speech = line.replace("Speech:", "").strip()
        elif line.startswith("LTL:"):
            current_ltl = line.replace("LTL:", "").strip()
        elif line.startswith("Predictions:"):
            current_predictions = line.replace("Predictions:", "").strip()
        elif line.startswith("References:"):
            current_references = line.replace("References:", "").strip()
        elif not line:
            speech.append(current_speech)
            ltl.append(current_ltl)
            predictions.append(current_predictions)
            references.append(current_references)

    # Create a DataFrame from the lists
    df = pd.DataFrame({
        'Speech': speech,
        'Predictions': predictions,
        'References': references
    })
    return df


#Spot Score Calculation
def cal_spot(preds, refs):
    return spot.are_equivalent(preds, refs)

#Input Folder
input_folder = os.getcwd() + '/results/predictions/speechGesturesT5/test'  #Update it for the required input directory
input_file =  input_folder + '/test_data_epoch_100.txt'
df = load_data(input_file)
score = 0

for i in range(len(df)):
    preds = df.iloc[:,1][i]
    refs = df.iloc[:,2][i]
    refs = refs.replace("[", "").replace("]", "")

    ref_formula = spot.formula(refs)

    try:
        pred_formula = spot.formula(preds)
        spotval = spot.are_equivalent(ref_formula, pred_formula)
    except:
        # print("Bad LTL", preds)
        spotval = False

    if spotval:
        score += 1

print(score/len(df))

