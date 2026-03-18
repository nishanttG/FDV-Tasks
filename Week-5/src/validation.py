import pandera.pandas as pa
from pandera.typing import Series

class KGNodeSchema(pa.DataFrameModel):
    """
    Validates the flattened data structure from the parser.
    """
    # 1. 'type' must be one of these values
    type: Series[str] = pa.Field(isin=["Article", "Clause", "Part", "Schedule"])
    
    # 2. 'id' must be alphanumeric (regex match)
    id: Series[str] = pa.Field(str_matches=r"^[A-Za-z0-9_\-\.]+$") 
    
    # 3. FIX: Use 'str_length' dict instead of 'min_length'
    text: Series[str] = pa.Field(str_length={"min_value": 1}) 
    
    # 4. 'parent_id' is nullable
    parent_id: Series[str] = pa.Field(nullable=True, coerce=True) 

    class Config:
        strict = True # Fail if unknown columns appear
        coerce = True # Auto-convert types