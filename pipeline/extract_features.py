import sys
import pandas as pd
import spacy


nlp = spacy.load("en_core_web_sm")


# From Asher et al. (2009)
appraisal_categories = {
    "inform": ["inform", "notify", "explain"],
    "assert": ["assert", "claim", "insist"],
    "tell": ["say", "announce", "report"],
    "remark": ["comment", "observe", "remark"],
    "think": ["think", "reckon", "consider"],
    "guess": ["presume", "suspect", "wonder"],
    "blame": ["blame", "criticize", "condemn"],
    "praise": ["praise", "agree", "approve"],
    "appreciation": ["good", "shameful", "brilliant"],
    "recommend": ["advise", "argue for"],
    "suggest": ["suggest", "propose"],
    "hope": ["wish", "hope"],
    "anger_calmdown": ["irritation", "anger"],
    "astonishment": ["astound", "daze"],
    "love_fascinate": ["fascinate", "captivate"],
    "hate_disappoint": ["demoralize", "disgust"],
    "fear": ["fear", "frighten", "alarm"],
    "offense": ["hurt", "chock"],
    "sadness_joy": ["happy", "sad"],
    "bore_entertain": ["bore", "distraction"]
}


def extract_interview_appraisal_categories(interview: str):
    appraisal_word_ratios = pd.Series({f"interview_{category}_ratio": 0.0 for category in appraisal_categories.keys()})

    num_appraisal_words = 0

    doc = nlp(interview)

    for token in doc:
        for category, words in appraisal_categories.items():
            if token.lemma_ in words:
                appraisal_word_ratios[f"interview_{category}_ratio"] += 1
                num_appraisal_words += 1

    if num_appraisal_words > 0:
        for category in appraisal_categories.keys():
            appraisal_word_ratios[f"interview_{category}_ratio"] /= num_appraisal_words
    
    return appraisal_word_ratios


ignore_columns = ["Name", "Role", "Transcript", "Resume", "Reason_for_decision", "Job_Description"]


def extract_all_features(data: pd.DataFrame):
    data_with_features = data.copy()
    
    appraisal_word_ratios = data_with_features["Transcript"].apply(extract_interview_appraisal_categories)

    data_with_features = pd.concat([data_with_features, appraisal_word_ratios], axis=1)

    data_with_features.drop(columns=ignore_columns, axis=1, inplace=True)

    return data_with_features


def main(input_file: str, output_file: str):
    data = pd.read_csv(input_file)

    data_with_features = extract_all_features(data)

    data_with_features.to_csv(output_file, index=False)


if __name__ == "__main__":
    if len(sys.argv) == 5 and sys.argv[4] == "DEBUG":
        input_file = sys.argv[2]
        output_file = sys.argv[3]

        main(input_file, output_file)

    elif len(sys.argv) != 3:
        print("Usage: python3 pipeline/extract_features.py <input_file> <output_file>")
    
    else:
        input_file = sys.argv[1]
        output_file = sys.argv[2]

        main(input_file, output_file)