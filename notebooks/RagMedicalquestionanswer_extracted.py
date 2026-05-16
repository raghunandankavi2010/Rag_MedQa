# --- cell start ---

#!/usr/bin/env python3
"""
Medical RAG Comparative Study - FULLY CORRECTED for Latest Libraries
Author: Raghunandan Kavi
Institution: Liverpool John Moores University
Dataset: MedQuAD (Medical Question Answering Dataset)

REQUIREMENTS (install via pip):
    pip install langchain langchain-openai langchain-chroma langchain-text-splitters
    pip install chromadb>=0.5.0 openai>=1.0.0
    pip install pandas numpy matplotlib seaborn scikit-learn nltk
    # Optional evaluation frameworks:
    pip install ragas>=0.2.0 deepeval
"""

import os
import sys
import json
import time
import zipfile
import requests
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import List, Dict, Tuple, Any
from dataclasses import dataclass
from collections import Counter, defaultdict

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
try:
    from dotenv import load_dotenv
    load_dotenv()  # Loads variables from .env file into environment
    DOTENV_AVAILABLE = True
except ImportError:
    DOTENV_AVAILABLE = False

# If .env loading failed or key still not set, show helpful message
if not DOTENV_AVAILABLE and ("OPENAI_API_KEY" not in os.environ or not os.environ["OPENAI_API_KEY"]):
    print("="*70)
    print("NOTE: python-dotenv not installed. To use .env file:")
    print("  pip install python-dotenv")
    print("="*70)
# =============================================================================
# API KEY VALIDATION (Fail fast with clear message)
# =============================================================================
if "OPENAI_API_KEY" not in os.environ or not os.environ["OPENAI_API_KEY"] or os.environ["OPENAI_API_KEY"] == 'OPENAI_API_KEY':
    raise EnvironmentError(
        "Please set your OPENAI_API_KEY environment variable before running.\n"
        "Example: export OPENAI_API_KEY='sk-...'"
    )

# =============================================================================
# Latest LangChain imports (v0.3+)
# =============================================================================
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import Chroma

from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_text_splitters import RecursiveCharacterTextSplitter, CharacterTextSplitter
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_classic.chains import create_retrieval_chain

# =============================================================================
# Evaluation frameworks - Updated for RAGAS v0.2+ and DeepEval latest
# =============================================================================
try:
    from ragas import evaluate
    from ragas.metrics import Faithfulness, AnswerRelevancy, ContextPrecision, ContextRecall
    RAGAS_AVAILABLE = True
except ImportError:
    RAGAS_AVAILABLE = False
    print("Warning: RAGAS not installed or incompatible version. Running with custom metrics only.")

try:
    from deepeval import evaluate as deepeval_evaluate
    from deepeval.test_case import LLMTestCase
    from deepeval.metrics import AnswerRelevancyMetric
    try:
        from deepeval.metrics import HallucinationMetric
    except ImportError:
        HallucinationMetric = None
    DEEPEVAL_AVAILABLE = True
except ImportError:
    DEEPEVAL_AVAILABLE = False
    print("Warning: DeepEval not installed.")

# =============================================================================
# Download required NLTK data (safe for all versions)
# =============================================================================
import nltk
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt', quiet=True)
    # punkt_tab is only present in NLTK 3.8.2+; silently ignore if missing
    try:
        nltk.data.find('tokenizers/punkt_tab')
    except LookupError:
        try:
            nltk.download('punkt_tab', quiet=True)
        except Exception:
            pass

# =============================================================================
# Configuration
# =============================================================================
@dataclass
class Config:
    DATA_DIR: str = "./medquad_data"
    VECTORDB_DIR: str = "./chroma_db"
    RESULTS_DIR: str = "./results"
    CHUNK_SIZE: int = 400
    CHUNK_OVERLAP: int = 50
    EMBEDDING_MODEL: str = "text-embedding-3-large"
    LLM_MODEL: str = "gpt-4o-mini"
    TEMPERATURE: float = 0.0
    TOP_K: int = 5
    EVAL_SAMPLE_SIZE: int = 100  # Reduced for API cost control; increase for full study
    
config = Config()

# Create directories
for dir_path in [config.DATA_DIR, config.VECTORDB_DIR, config.RESULTS_DIR]:
    os.makedirs(dir_path, exist_ok=True)

# =============================================================================
# Data Loader - FIXED download logic
# =============================================================================
class MedQuADLoader:
    """Handles downloading and parsing of MedQuAD dataset."""
    
    def __init__(self, data_dir: str):
        self.data_dir = Path(data_dir)
        self.csv_path = self.data_dir / "medquad.csv"
        
    def download(self) -> None:
        """Download MedQuAD from GitHub repository."""
        if self.csv_path.exists():
            print(f"Dataset already exists at {self.csv_path}")
            return
            
        print("Downloading MedQuAD dataset...")
        repo_url = "https://github.com/abachaa/MedQuAD.git"
        clone_dir = self.data_dir / "MedQuAD_repo"
        xml_dir = None
        
        # Try git clone first
        if not clone_dir.exists():
            try:
                import subprocess
                subprocess.run(
                    ["git", "clone", "--depth", "1", repo_url, str(clone_dir)], 
                    check=True, capture_output=True, text=True
                )
                print(f"Cloned repository to {clone_dir}")
                xml_dir = clone_dir
            except (subprocess.CalledProcessError, FileNotFoundError):
                print("Git not available or clone failed. Trying ZIP download...")
        
        # Fallback: Download ZIP archive if git failed
        if xml_dir is None:
            zip_url = "https://github.com/abachaa/MedQuAD/archive/refs/heads/master.zip"
            zip_path = self.data_dir / "medquad.zip"
            try:
                r = requests.get(zip_url, timeout=60)
                r.raise_for_status()
                with open(zip_path, "wb") as f:
                    f.write(r.content)
                
                with zipfile.ZipFile(zip_path, "r") as z:
                    z.extractall(self.data_dir)
                
                extracted = self.data_dir / "MedQuAD-master"
                if extracted.exists():
                    xml_dir = extracted
                else:
                    # Fallback: find any MedQuAD-* folder
                    candidates = list(self.data_dir.glob("MedQuAD-*"))
                    if candidates:
                        xml_dir = candidates[0]
                
                zip_path.unlink(missing_ok=True)
            except Exception as e:
                raise RuntimeError(f"Failed to download MedQuAD dataset: {e}")
        
        if xml_dir is None or not xml_dir.exists():
            raise RuntimeError(f"Could not locate MedQuAD data directory.")
            
        print(f"Parsing XML files from {xml_dir}...")
        
        data = []
        xml_files = list(xml_dir.rglob("*.xml"))
        if not xml_files:
            raise RuntimeError(f"No XML files found in {xml_dir}")
        
        for xml_file in xml_files:
            try:
                tree = ET.parse(xml_file)
                root = tree.getroot()
                
                # Extract QA pairs based on MedQuAD structure
                focus = root.findtext(".//Focus", default="")
                
                for qa in root.findall(".//QAPair"):
                    question = qa.findtext("Question", default="")
                    answer = qa.findtext("Answer", default="")
                    qtype = qa.get("qtype", "general")
                    
                    if question and answer:
                        data.append({
                            "focus": focus,
                            "question": question.strip(),
                            "answer": answer.strip(),
                            "qtype": qtype,
                            "source_file": xml_file.name
                        })
            except Exception as e:
                print(f"Error parsing {xml_file}: {e}")
                continue
        
        if not data:
            raise RuntimeError("No QA pairs extracted from XML files.")
        
        # Save to CSV
        df = pd.DataFrame(data)
        df.to_csv(self.csv_path, index=False)
        print(f"Saved {len(df)} QA pairs to {self.csv_path}")
        
        # Cleanup (robust removal to handle Windows permission errors)
        if clone_dir.exists():
            import shutil, stat, errno
            def _on_rm_error(func, path, exc_info):
                # exc_info is (type, value, traceback)
                exc = exc_info[1]
                try:
                    # Try to change permissions and retry
                    os.chmod(path, stat.S_IWRITE)
                except Exception:
                    pass
                try:
                    func(path)
                except Exception:
                    # If still failing for a file, try to remove as file
                    try:
                        if os.path.isdir(path):
                            os.rmdir(path)
                        else:
                            os.remove(path)
                    except Exception:
                        # give up for this path
                        pass
            try:
                shutil.rmtree(clone_dir, onerror=_on_rm_error)
            except Exception as e:
                print(f"Warning: Could not remove temporary directory {clone_dir}: {e}. You may need to delete it manually.")
            
    def load(self) -> pd.DataFrame:
        """Load and preprocess the dataset."""
        if not self.csv_path.exists():
            self.download()
            
        df = pd.read_csv(self.csv_path)
        
        # Clean data
        df = df.dropna(subset=["question", "answer"])
        df = df[df["question"].str.len() > 10]
        df = df[df["answer"].str.len() > 20]
        
        # Add metadata
        df["answer_len"] = df["answer"].apply(lambda x: len(str(x).split()))
        df["question_len"] = df["question"].apply(lambda x: len(str(x).split()))
        
        return df
# --- cell start ---

