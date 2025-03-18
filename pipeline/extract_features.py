import sys
import pandas as pd


# From Asher et al. (2009)
verb_categories = {
    "inform": ["inform", "notify", "explain"],
    "assert": ["assert", "claim", "insist"],
}


def extract_interview_verb_categories(data: pd.DataFrame):
    for category in verb_categories.keys():
        data[f"interview_{category}_ratio"] = 0

    interview: str = data["Transcript"]

    num_appraisal_verbs = 0

    for token in interview:
        for category, verbs in verb_categories.items():
            if token in verbs:
                data[f"interview_{category}_ratio"] += 1
                num_appraisal_verbs += 1

    if num_appraisal_verbs > 0:
        for category in verb_categories.keys():
            data[f"interview_{category}_ratio"] /= num_appraisal_verbs
    
    return data


ignore_columns = ["Name", "Role", "Transcript", "Resume", "Reason_for_decision", "Job_Description"]


def extract_all_features(data):
    data_with_features = extract_interview_verb_categories(data)

    data_with_features.drop(columns=ignore_columns, axis=1, inplace=True)

    return data_with_features


def main():
    input_file = sys.argv[1]
    output_file = sys.argv[2]

    data = pd.read_csv(input_file)

    data_with_features = extract_all_features(data)

    data_with_features.to_csv(output_file, index=False)


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python3 pipeline/extract_features.py <input_file> <output_file>")
    else:
        main()