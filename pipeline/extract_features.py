import sys
import pandas as pd
import spacy
from spacy.tokens import Doc
from concurrent.futures import ThreadPoolExecutor
import re


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


# From Biber (2004)
attitudinal_categories = {
    "positive": ["amazed", "amazing", "amused", "funny", "glad", "good", 
                 "grateful", "great", "happy", "hopeful", "wonderful", 
                 "pleased", "preferable", "reassured", "relieved", "thankful", 
                 "extraordinary", "incredible", "acceptable", "advisable", 
                 "appropriate", "encouraged", "silly", "desirable", "fortunate", 
                 "interesting", "nice", "lucky", "satisfied", "sensible", 
                 "surprised", "surprising", "neat"],
    "neutral": ["adamant", "anomalous", "aware", "careful", "conceivable", 
                "critical", "crucial", "curious", "essential", "fitting", 
                "imperative", "incidental", "inconceivable", "indisputable", 
                "ironic", "natural", "necessary", "notable", "noteworthy", 
                "noticeable", "obligatory", "odd", "okay", "paradoxical", 
                "peculiar", "ridiculous", "strange", "sufficient", "typical", 
                "unaware", "understandable", "untypical", "unusual", "vital"],
    "negative": ["afraid", "alarmed", "angry", "annoyed", "annoying", "concerned", 
                 "depressed", "upset", "upsetting", "worried", "disappointed", 
                 "disappointing", "dissatisfied", "distressed", "disturbed", 
                 "dreadful", "embarrasing", "uncomfortable", "unfair", "unfortunate", 
                 "unhappy", "unlucky", "awful", "frightened", "frightening", "horrible", 
                 "hurt", "unacceptable", "unthinkable", "ashtonished", "ashtonishing", "sorry", 
                 "mad", "stupid", "irritated", "irritating", "sad", "shocked", "shocking", "tragic"],
}


def extract_interview_appraisal_categories_doc(doc: Doc, index: int, prefix: str, word_to_category: dict[str, str]):
    num_appraisal_words = 0
    word_counts = {word: 0 for word in word_to_category.keys()}

    for token in doc:
        if token.lemma_ in word_to_category:
            word_counts[token.lemma_] += 1
            num_appraisal_words += 1

    category_counts = {category: sum([word_counts[word] for word, cat in word_to_category.items() if cat == category]) 
                       for category in appraisal_categories.keys()}

    if len(doc) == 0:
        ratios = {f"interview_{prefix}_{category}_ratio": 0 for category in appraisal_categories.keys()}
    else:
        ratios = {f"interview_{prefix}_{category}_ratio": category_counts[category] / len(doc) for category in appraisal_categories.keys()}
    
    return index, ratios


def add_doc(text: str, index: int, docs: list[Doc]):
    docs[index] = nlp(text)


transcript_speech_pattern = re.compile(r"^(.*?):(.*)$")


def separate_interviewer_and_candidate_transcripts(transcript: str):
    lines = transcript.split("\n\n")
    lines = [line.strip() for line in lines if line.strip()]
    lines = [line.replace("\n", " ") for line in lines]

    interviewer_lines: list[str] = []
    candidate_lines: list[str] = []

    for i, line in enumerate(lines):
        match = transcript_speech_pattern.match(line)

        if match:
            groups = match.groups()

            if len(groups) != 2:
                continue

            speaker = groups[0]
            speech = groups[1]
            
            if speaker == "INTERVIEWER":
                interviewer_lines.append(speech)
            elif speaker == "CANDIDATE":
                candidate_lines.append(speech)
            else:
                print("Invalid transcript found at", i)

    interviewer_transcript = "\n".join(interviewer_lines)
    candidate_transcript = "\n".join(candidate_lines)

    return interviewer_transcript, candidate_transcript


def extract_interview_appraisal_categories(data: pd.DataFrame):
    interviews = data["Transcript"]

    appraisal_word_ratios = pd.DataFrame(0.0, index=interviews.index, 
                                         columns=[f"interview_{category}_ratio" for category in appraisal_categories.keys()])
    appraisal_word_ratios_interviewer = pd.DataFrame(0.0, index=interviews.index, 
                                         columns=[f"interview_interviewer_{category}_ratio" for category in appraisal_categories.keys()])
    appraisal_word_ratios_candidate = pd.DataFrame(0.0, index=interviews.index,
                                            columns=[f"interview_candidate_{category}_ratio" for category in appraisal_categories.keys()])

    interviewer_docs: list[Doc] = [None] * len(interviews)
    candidate_docs: list[Doc] = [None] * len(interviews)

    with nlp.select_pipes(enable=["tok2vec", "tagger", "attribute_ruler", "lemmatizer"]):
        interviewer_transcripts: list[str] = []
        candidate_transcripts: list[str] = []

        for _, row in data.iterrows():
            interview: str = row["Transcript"]
            candidate_name: str = row["Name"]

            interviewer_transcript, candidate_transcript = separate_interviewer_and_candidate_transcripts(interview)

            interviewer_transcripts.append(interviewer_transcript)
            candidate_transcripts.append(candidate_transcript)

        interviewer_docs = list(nlp.pipe(interviewer_transcripts, n_process=-1))
        candidate_docs = list(nlp.pipe(candidate_transcripts, n_process=-1))

    word_to_category_interviewer = {word: category for category, words in appraisal_categories.items() for word in words}
    word_to_category_candidate = {word: category for category, words in appraisal_categories.items() for word in words}

    with ThreadPoolExecutor(1000) as executor:
        futures_interviewer = {executor.submit(extract_interview_appraisal_categories_doc, doc, i, "interviewer", word_to_category_interviewer): 
                               i for i, doc in enumerate(interviewer_docs)}
        for future in futures_interviewer:
            index, ratios = future.result()
            for category, ratio in ratios.items():
                appraisal_word_ratios_interviewer.at[index, category] = ratio

        futures_candidate = {executor.submit(extract_interview_appraisal_categories_doc, doc, i, "candidate", word_to_category_candidate):
                                 i for i, doc in enumerate(candidate_docs)}
        for future in futures_candidate:
            index, ratios = future.result()
            for category, ratio in ratios.items():
                appraisal_word_ratios_candidate.at[index, category] = ratio

        appraisal_word_ratios = pd.concat([appraisal_word_ratios_interviewer, appraisal_word_ratios_candidate], axis=1)

    return appraisal_word_ratios


def extract_interview_attitudinal_adjectives_doc(doc: Doc, index: int, word_to_category: dict[str, str]):
    num_attitudinal_words = 0
    word_counts = {word: 0 for word in word_to_category.keys()}

    for token in doc:
        if token.lemma_ in word_to_category:
            word_counts[token.lemma_] += 1
            num_attitudinal_words += 1

    category_counts = {category: sum([word_counts[word] for word, cat in word_to_category.items() if cat == category]) 
                       for category in attitudinal_categories.keys()}

    ratios = {f"interview_attitudinal_{category}_ratio": category_counts[category] / len(doc) for category in attitudinal_categories.keys()}
    return index, ratios


def extract_interview_attitudinal_adjectives(interviews: pd.Series):
    attitudinal_word_ratios = pd.DataFrame(0.0, index=interviews.index, 
                                           columns=[f"interview_attitudinal_{category}_ratio" for category in attitudinal_categories.keys()])

    docs: list[Doc] = [None] * len(interviews)

    with nlp.select_pipes(enable=["tok2vec", "tagger", "attribute_ruler", "lemmatizer"]):
        docs = list(nlp.pipe(interviews, n_process=-1))

    word_to_category = {word: category for category, words in attitudinal_categories.items() for word in words}

    with ThreadPoolExecutor() as executor:
        futures = {executor.submit(extract_interview_attitudinal_adjectives_doc, doc, i, word_to_category): i for i, doc in enumerate(docs)}
        for future in futures:
            index, ratios = future.result()
            for category, ratio in ratios.items():
                attitudinal_word_ratios.at[index, category] = ratio

    return attitudinal_word_ratios


ignore_columns = ["Name", "Role", "Transcript", "Resume", "Reason_for_decision", "Job_Description"]


def extract_all_features(data: pd.DataFrame):
    data_with_features = data.copy()
    
    appraisal_word_ratios = extract_interview_appraisal_categories(data_with_features)
    attitudinal_word_ratios = extract_interview_attitudinal_adjectives(data_with_features["Transcript"])

    data_with_features = pd.concat([data_with_features, 
                                    appraisal_word_ratios, 
                                    attitudinal_word_ratios], axis=1)

    data_with_features.drop(columns=ignore_columns, axis=1, inplace=True)

    return data_with_features


def main(input_file: str, output_file: str):
    data = pd.read_csv(input_file)

    data_with_features = extract_all_features(data)

    data_with_features.to_csv(output_file, index=False)


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python3 pipeline/extract_features.py <input_file> <output_file>")
    
    else:
        input_file = sys.argv[1]
        output_file = sys.argv[2]

        main(input_file, output_file)