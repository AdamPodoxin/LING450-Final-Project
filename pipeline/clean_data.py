import sys
import re
import pandas as pd


honorifics_abbr = ["Mr", "Mrs", "Ms", "Dr", "PhD"]
honorifics_full = ["Mister", "Missus", "Miss", "Doctor"]


def get_honorific(name: str):
    for honorific in honorifics_abbr:
        if f"{honorific.lower()}." in name.lower():
            return honorific
    
    for honorific in honorifics_full:
        if honorific.lower() in name.lower():
            return honorific
    
    return None


def get_name_with_honorifics(name: str, honorific: str, use_full_name = False):
    if len(name.split(" ")) == 2:
        # Already in correct form
        return name

    last_name = name.split(" ")[2]

    name_to_use = " ".join(name.split(" ")[1:3]) if use_full_name else last_name

    if honorific in honorifics_full:
        return f"{honorific} {name_to_use}"
    else:
        return f"{honorific}. {name_to_use}"


def get_name_line_pattern(name: str, start_only = True):
    pattern = rf"\**\-*{name}\**\-*:\**\-*\s*\**\-*(.*)"

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
        interviewer_name_pattern = get_name_line_pattern(interviewer_name, start_only=False)
        matches = interviewer_name_pattern.findall(transcript)

        if len(matches) > 1:
            return interviewer_name
        
        interviewer_name_without_honorific = " ".join(interviewer_name.split(" ")[1:])
        interviewer_name_pattern = get_name_line_pattern(interviewer_name_without_honorific, start_only=False)
        matches = interviewer_name_pattern.findall(transcript)

        if len(matches) > 1:
            return interviewer_name_without_honorific

        interviewer_name = get_name_with_honorifics(interviewer_name, honorific)
        interviewer_name_pattern = get_name_line_pattern(interviewer_name, start_only=False)
        matches = interviewer_name_pattern.findall(transcript)
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
        interviewer_name_pattern = get_name_line_pattern(interviewer_name, start_only=False)
        matches = interviewer_name_pattern.findall(transcript)

        if matches:
            return interviewer_name
        
        interviewer_name = "HR manager"
        interviewer_name_pattern = get_name_line_pattern(interviewer_name, start_only=False)
        matches = interviewer_name_pattern.findall(transcript)

        if matches:
            return interviewer_name
    
        interviewer_name = "Company Representative"
        interviewer_name_pattern = get_name_line_pattern(interviewer_name, start_only=False)
        matches = interviewer_name_pattern.findall(transcript)

        if matches:
            return interviewer_name
        
        interviewer_name = "interviewer"

    return interviewer_name


def find_candidate_name(transcript: str, original_name: str):
    candidate_name = original_name
    candidate_name_pattern = get_name_line_pattern(candidate_name, start_only=False)
    matches = candidate_name_pattern.findall(transcript)

    if len(matches) > 1:
        return candidate_name

    # Name of candidate is mentioned at the beginning,
    # and the name is used in each line instead of "Candidate:"
    candidate_name_pattern = get_name_line_pattern("Candidate", start_only=False)
    matches = candidate_name_pattern.findall(transcript)

    if len(matches) == 1:
        matches = re.search(candidate_name_pattern, transcript)
        
        if matches:
            candidate_name = matches.group(1).strip()

            if ',' in candidate_name:
                candidate_name = candidate_name.split(",")[0].strip()
    
    candidate_name = re.sub(r"[^a-zA-Z\s,\.]", "", candidate_name)

    if len(candidate_name.split(" ")) > 2:
        # find honorific at the beginning of the name
        honorific = get_honorific(candidate_name)
        if honorific and candidate_name.lower().startswith(honorific.lower()):
            candidate_name = get_name_with_honorifics(candidate_name, honorific)
        else:
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
    
    candidate_name = "candidate"
    candidate_name_pattern = get_name_line_pattern(candidate_name, start_only=False)
    matches = candidate_name_pattern.findall(transcript)

    if matches:
        return candidate_name

    candidate_name = original_name

    return candidate_name


def is_valid_transcript(transcript: str):
    if transcript == "":
        return False
    elif transcript == " ":
        return False
    elif transcript is None:
        return False
    elif not isinstance(transcript, str):
        return False
    elif "CANDIDATE:" not in transcript or "INTERVIEWER:" not in transcript:
        return False
    
    return True


def clean_transcript(data: pd.Series):
    try:
        transcript: str = data["Transcript"]

        lines = transcript.split("\n\n")
        lines = [line.strip() for line in lines]
        lines = [line.replace("\n", "") for line in lines]

        lines_joined = "\n".join(lines)

        interviewer_name = find_interviewer_name(lines_joined).lower()

        candidate_name: str = find_candidate_name(lines_joined, data["Name"])
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
        return None


def main(input_file: str, output_file: str):
    data = pd.read_csv(input_file)
    original_num_rows = data.shape[0]

    for _, row in data.iterrows():
        cleaned_transcript = clean_transcript(row)

        if not is_valid_transcript(cleaned_transcript):
            cleaned_transcript = None
            print("Invalid transcript for", row["ID"])

        data.at[row.name, "Transcript"] = cleaned_transcript
    
    data = data.dropna()

    new_num_rows = data.shape[0]

    print("Removed", original_num_rows - new_num_rows, "rows")
    
    data.to_csv(output_file, index=False)


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python3 pipeline/clean_data.py <input_file> <output_file>")
    
    else:
        input_file = sys.argv[1]
        output_file = sys.argv[2]

        main(input_file, output_file)