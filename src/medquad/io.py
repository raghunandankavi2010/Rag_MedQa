import os
import subprocess
import xml.etree.ElementTree as ET
import pandas as pd

MEDQUAD_GIT = "https://github.com/abachaa/MedQuAD.git"


def clone_repo(url: str, dest: str) -> None:
    """Clone the repository to `dest` if it doesn't already exist or is empty."""
    if os.path.exists(dest) and any(os.scandir(dest)):
        return
    os.makedirs(os.path.dirname(dest) or ".", exist_ok=True)
    subprocess.run(["git", "clone", url, dest], check=True)


def parse_medquad_xml(file_path: str) -> list:
    results = []
    try:
        tree = ET.parse(file_path)
        root = tree.getroot()

        focus = root.findtext('Focus', default='')
        group = root.findtext('.//SemanticGroup', default='Unknown')

        syns = [s.text for s in root.findall('.//Synonym') if s.text]
        synonyms_str = ", ".join(syns)

        for qa in root.findall('.//QAPair'):
            question = qa.findtext('Question', default='')
            answer = qa.findtext('Answer', default='')
            qtype = qa.get('qtype', 'general')

            if answer and len(answer.strip()) > 0:
                results.append({
                    "focus": focus,
                    "group": group,
                    "synonyms": synonyms_str,
                    "qtype": qtype,
                    "question": question,
                    "answer": answer,
                    "source_file": os.path.basename(file_path)
                })
    except Exception:
        pass
    return results


def parse_folder(base_path: str) -> pd.DataFrame:
    """Walk `base_path` and parse all XML files into a DataFrame."""
    all_rows = []
    for root_dir, dirs, files in os.walk(base_path):
        for file in files:
            if file.lower().endswith('.xml'):
                full_path = os.path.join(root_dir, file)
                all_rows.extend(parse_medquad_xml(full_path))
    df = pd.DataFrame(all_rows)
    return df


def load_csv(path: str) -> pd.DataFrame:
    return pd.read_csv(path)
