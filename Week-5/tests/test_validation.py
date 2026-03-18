# Unit Test
import pytest
import pandas as pd
import pandera as pa
from src.validation import ConstitutionSchema

# 1. Test Valid Schema
def test_schema_valid():
    data = {
        "type": ["Article", "Clause"],
        "id": ["Art_1", "Art_1.2"],
        "text": ["This is an article", "This is a clause"]
    }
    df = pd.DataFrame(data)
    validated_df = ConstitutionSchema.validate(df)
    assert isinstance(validated_df, pd.DataFrame)

# 2. Test Invalid Type (Fail Fast)
def test_schema_invalid_type():
    data = {
        "type": ["RandomObject"], # Invalid type
        "id": ["Art_1"],
        "text": ["Text"]
    }
    df = pd.DataFrame(data)
    with pytest.raises(pa.errors.SchemaError):
        ConstitutionSchema.validate(df)

# 3. Test Empty Text (Fail Fast)
def test_schema_empty_text():
    data = {
        "type": ["Article"],
        "id": ["Art_2"],
        "text": [""] # Empty text not allowed
    }
    df = pd.DataFrame(data)
    with pytest.raises(pa.errors.SchemaError):
        ConstitutionSchema.validate(df)

# 4. Test ID Regex (No spaces allowed in ID)
def test_schema_invalid_id():
    data = {
        "type": ["Article"],
        "id": ["Art 1"], # Space not allowed in ID regex
        "text": ["Text"]
    }
    df = pd.DataFrame(data)
    with pytest.raises(pa.errors.SchemaError):
        ConstitutionSchema.validate(df)

# 5. Test Regex Logic (Mocking extraction)
def test_regex_article_detection():
    import re
    re_article = re.compile(r'^(\d+)\.', re.IGNORECASE)
    line = "1. Name of Article"
    match = re_article.search(line)
    assert match is not None
    assert match.group(1) == "1"

# 6. Test Regex Clause Detection
def test_regex_clause_detection():
    import re
    re_clause = re.compile(r'^\((\d+)\)')
    line = "(1) This is a clause"
    match = re_clause.search(line)
    assert match is not None
    assert match.group(1) == "1"