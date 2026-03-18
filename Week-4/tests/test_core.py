import pytest
import sys
import os
import pandas as pd
import numpy as np

# Ensure we can import from 'scripts'
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from scripts.preprocess import clean_text
from scripts.prompt_engine import PromptEngine
from scripts.utils import calculate_md5
from sklearn.metrics import f1_score

# --- TEST 1: Data Preprocessing ---
def test_clean_text_removes_html():
    raw = "This is <br /> bad."
    cleaned = clean_text(raw)
    assert "<br" not in cleaned
    assert cleaned.strip() == "this is   bad" # Expect lowercasing + space replacement

def test_clean_text_removes_special_chars():
    raw = "Hello!!! 123"
    # Assuming baseline preprocess removes non-alpha
    cleaned = clean_text(raw)
    assert "!" not in cleaned
    assert "123" not in cleaned

# --- TEST 2: Metrics Calculation ---
def test_f1_score_calculation():
    y_true = [1, 0, 1, 1]
    y_pred = [1, 0, 0, 1]
    # Sklearn macro F1
    score = f1_score(y_true, y_pred, average='macro')
    assert 0.0 <= score <= 1.0
    assert score > 0.5 # Basic sanity check

# --- TEST 3: Prompt Engineering Logic ---
def test_prompt_generation_zero_shot():
    engine = PromptEngine()
    prompt = engine.generate_prompt("Movie was good", "Zero-Shot Basic")
    assert 'Review: "Movie was good"' in prompt
    assert 'Sentiment:' in prompt

def test_prompt_truncation():
    # Test if prompt engine handles long text without crashing
    long_text = "word " * 2000
    engine = PromptEngine()
    prompt = engine.generate_prompt(long_text, "Zero-Shot Basic")
    assert len(prompt) < 10000 # Should be reasonably truncated

# --- TEST 4: Utility Functions ---
def test_md5_checksum(tmp_path):
    # Create a temporary dummy file
    d = tmp_path / "test_data"
    d.mkdir()
    p = d / "test.txt"
    p.write_text("content")
    
    # Calculate checksum
    checksum = calculate_md5(str(p))
    # MD5 of "content" is known constant
    assert isinstance(checksum, str)
    assert len(checksum) == 32

# --- TEST 5: Config Structure (Integration-Lite) ---
def test_day3_config_integrity():
    # Verify that the config file is importable and has keys
    from scripts.day3_config import LORA_CONFIG
    assert "Base Model" in LORA_CONFIG
    assert "Train Data" in LORA_CONFIG