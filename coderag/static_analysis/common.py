from typing import List, Optional
from pydantic import BaseModel

# Define a model for node information using Pydantic
class Element(BaseModel):
    type: str                     # "function" or "variable"
    key: str                      # Composed of file_location, class_hierarchy, and name
    name: str                     # For functions, full signature (function name + parameter list); for variables, variable name
    file_location: List[str]      # File location (only up to the file level), e.g. ["dirA", "a.py"]
    class_hierarchy: List[str]    # Nested class hierarchy, ordered from outer to inner, e.g. ["ClassA", "ClassB"]
    positions: List[str] = []     # List of occurrence positions, recording all line numbers, e.g. ["line_10", "line_45"]
    docstring: Optional[str] = None  # Optional field to store the function's docstring
    body: Optional[str] = None       # New field: represents the function body content
