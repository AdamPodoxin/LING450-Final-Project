import sys
import re
import pandas as pd


def get_honorific(name: str):
    honorifics = ["Mr", "Mrs", "Ms", "Dr", "PhD"]

    for honorific in honorifics:
        if honorific.lower() in name.lower():
            return honorific
    
    return None


def get_name_with_honorifics(name: str, honorific: str):
    last_name = name.split(" ")[-1]
    return f"{honorific} {last_name}"


def get_name_line_pattern(name: str, start_only = True):
    pattern = rf"\**{name}\**:\**\s*\**(.*)"

    if start_only:
        pattern = rf"^{pattern}"
        
    return re.compile(pattern, re.IGNORECASE)


def find_interviewer_name(transcript: str):
    interviewer_name = "Interviewer"
    interviewer_name_pattern = get_name_line_pattern(interviewer_name, start_only=False)
    matches = interviewer_name_pattern.findall(transcript)

    # Name of interviewer is mentioned at the beginning,
    # and the name is used in each line instead of "Interviewer:"
    if len(matches) == 1:
        matches = re.search(interviewer_name_pattern, transcript)
        
        if matches:
            interviewer_name = matches.group(1).strip()

            if ',' in interviewer_name:
                interviewer_name = interviewer_name.split(",")[0].strip()
    
    interviewer_name = re.sub(r"[^a-zA-Z\s,\.]", "", interviewer_name)
    
    honorific = get_honorific(interviewer_name)
    if honorific:
        interviewer_name = get_name_with_honorifics(interviewer_name, honorific)
    else:
        interviewer_name_pattern = get_name_line_pattern(interviewer_name, start_only=False)
        matches = interviewer_name_pattern.findall(transcript)

        if matches:
            return interviewer_name

        # Try first name only
        first_name = interviewer_name.split(" ")[0]
        interviewer_name_pattern = get_name_line_pattern(first_name, start_only=False)
        matches = interviewer_name_pattern.findall(transcript)

        if matches:
            interviewer_name = first_name
            return interviewer_name

        interviewer_name = "hiring manager"

    return interviewer_name


def find_candidate_name(transcript: str, original_name: str):
    candidate_name = original_name

    if len(candidate_name.split(" ")) > 2:
        matches = re.findall(rf"\b{candidate_name}\b", transcript, re.IGNORECASE)

        if len(matches) <= 1:
            candidate_name = " ".join(candidate_name.split(" ")[:2])

    candidate_name_pattern = get_name_line_pattern(candidate_name, start_only=False)
    matches = candidate_name_pattern.findall(transcript)

    if matches:
        return candidate_name

    # Try first name only
    first_name = candidate_name.split(" ")[0]
    candidate_name_pattern = get_name_line_pattern(first_name, start_only=False)
    matches = candidate_name_pattern.findall(transcript)

    if matches:
        return first_name
    else:
        candidate_name = "candidate"

    return candidate_name


def clean_transcript(data: pd.Series):
    try:
        transcript: str = data["Transcript"]

        lines = transcript.split("\n\n")
        lines = [line.strip() for line in lines if line.strip()]
        lines = [line.replace("\n", " ") for line in lines]

        interviewer_name = find_interviewer_name(transcript).lower()
        candidate_name: str = find_candidate_name(transcript, data["Name"])
        candidate_name = candidate_name.lower()

        interviewer_lines: list[str] = []
        candidate_lines: list[str] = []

        interviewer_line_pattern = get_name_line_pattern(interviewer_name)
        candidate_line_pattern = get_name_line_pattern(candidate_name)

        for line in lines:
            if interviewer_line_pattern.match(line):
                interviewer_lines.append(interviewer_line_pattern.match(line).group(1).strip())
            elif candidate_line_pattern.match(line):
                candidate_lines.append(candidate_line_pattern.match(line).group(1).strip())

        combined_lines = []

        i = 0
        j = 0

        while i < len(interviewer_lines) and j < len(candidate_lines):
            combined_lines.append(f"INTERVIEWER: {interviewer_lines[i]}")
            combined_lines.append(f"CANDIDATE: {candidate_lines[j]}")
            i += 1
            j += 1
        
        while i < len(interviewer_lines):
            combined_lines.append(f"INTERVIEWER: {interviewer_lines[i]}")
            i += 1
        
        while j < len(candidate_lines):
            combined_lines.append(f"CANDIDATE: {candidate_lines[j]}")
            j += 1

        cleaned_transcript = "\n".join(combined_lines)

        return cleaned_transcript
    except Exception as e:
        print(f"Error cleaning transcript for {data["ID"]}: {e}")
        return data["Transcript"]


def main(input_file: str, output_file: str):
    data = pd.read_csv(input_file)

    for _, row in data.iterrows():
        cleaned_transcript = clean_transcript(row)
        data.at[row.name, "Transcript"] = cleaned_transcript
    
    data.to_csv(output_file, index=False)


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python3 pipeline/clean_data.py <input_file> <output_file>")
    
    else:
        input_file = sys.argv[1]
        output_file = sys.argv[2]

        main(input_file, output_file)