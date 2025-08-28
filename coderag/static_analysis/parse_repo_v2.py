from pathlib import Path
from typing import List, Optional, Union
from pydantic import BaseModel
from tree_sitter import Language, Parser, Query, Node
from loguru import logger
import tree_sitter_python as tspython
import os
import chardet

def read_source_file(filepath: Path) -> bytes:
    raw = filepath.read_bytes()
    encoding = chardet.detect(raw)['encoding'] or 'utf-8'
    return raw.decode(encoding, errors='replace').encode('utf-8')


# 1. Pydantic models to represent code nodes
class FunctionNode(BaseModel):
    file_path: Path             # Relative path of the source file
    class_name: Optional[str]   # Name of the enclosing class; None if top-level function
    decorators: List[str]       # List of decorators
    name: str                   # Function name
    params: List[str]           # List of parameters
    docstring: Optional[str]    # Docstring of the function
    body: str                   # Source code of the function body


class VariableNode(BaseModel):
    file_path: Path             # Relative path of the source file
    class_name: str             # Name of the enclosing class
    is_static: bool             # Whether it is a static field
    name: str                   # Variable name
    annotation: Optional[str]   # Type annotation if any
    value: Optional[str]        # Initialization value if any


# 2. Helper function to render a node as Python source code string
def render_node(node: Union[FunctionNode, VariableNode], use_path=True) -> str:
    lines: List[str] = [f"# {node.file_path}"] if use_path else []

    if isinstance(node, FunctionNode):
        # Render function node
        if node.class_name:
            lines.append(f"class {node.class_name}:")
            indent = "    "
        else:
            indent = ""

        for dec in node.decorators:
            lines.append(f"{indent}@{dec}")

        params = ", ".join(node.params)
        lines.append(f"{indent}def {node.name}({params}):")

        if node.docstring:
            lines.append(f"{indent}    '''{node.docstring}'''")

        for line in node.body.splitlines():
            lines.append(f"{indent}    {line}")

    else:
        # Render variable node
        lines.append(f"class {node.class_name}:")
        indent = "    "
        if node.is_static:
            ann = f": {node.annotation}" if node.annotation else ""
            val = f" = {node.value}" if node.value else ""
            lines.append(f"{indent}{node.name}{ann}{val}")
        else:
            # Assume instance field initialized inside __init__
            lines.append(f"{indent}def __init__(self):")
            ann = f": {node.annotation}" if node.annotation else ""
            val = node.value or ""
            lines.append(f"{indent}    self.{node.name}{ann} = {val}")

    return "\n".join(lines)


# 3. Initialize Tree-sitter parser for Python
PY_LANGUAGE: Language = Language(tspython.language())
parser: Parser = Parser(PY_LANGUAGE)


# 4. Tree-sitter queries for functions and variables

FUNCTION_QUERY = Query(PY_LANGUAGE, r'''
(
  (function_definition) @func_def
)
''')

# Query for static fields (assignments at class top level)
VARIABLE_QUERY = Query(PY_LANGUAGE, r'''
(
  (class_definition
     body: (block (expression_statement
                    (assignment left: (identifier) @var_name
                                right: (_) @value
                    )
                 )+
              )
  )
)
''')

# Query for instance fields assigned inside __init__
INIT_QUERY = Query(PY_LANGUAGE, r'''
(
  (function_definition
     name: (identifier) @init_name
     body: (block
              (expression_statement
                 (assignment
                    left: (attribute
                             object: (identifier) @self_obj
                             attribute: (identifier) @var_name
                          )
                    right: (_) @value
                 )
              )+
          )
  )
)
''')


# 5. Utility functions to work with AST nodes

def get_node_text(node: Node, source: bytes) -> str:
    return source[node.start_byte:node.end_byte].decode('utf-8')


def find_enclosing_class(node: Node) -> Optional[str]:
    current = node
    while current:
        if current.type == 'class_definition':
            name_node = current.child_by_field_name('name')
            return name_node.text.decode('utf-8') if name_node else None
        current = current.parent
    return None


def extract_decorators(node: Node, source: bytes) -> List[str]:
    decorators = []
    parent = node.parent
    if parent and parent.type == 'decorated_definition':
        for child in parent.children:
            if child.type == 'decorator':
                decorators.append(get_node_text(child, source).lstrip('@'))
    return decorators


def extract_docstring(node: Node, source: bytes) -> Optional[str]:
    body_node = node.child_by_field_name('body')
    for child in body_node.children:
        if child.type == 'expression_statement' and child.children and child.children[0].type == 'string':
            return get_node_text(child.children[0], source).strip('"""')
    return None


def parse_parameters(param_node: Node, source: bytes) -> List[str]:
    text = get_node_text(param_node, source)
    return [param.strip() for param in text.strip('()').split(',') if param.strip()]


def extract_body_source(node: Node, source: bytes) -> str:
    body_node = node.child_by_field_name('body')
    return '\n'.join(
        get_node_text(child, source)
        for child in body_node.children
        if not (child.type == 'expression_statement' and child.children[0].type == 'string')
    )


# 6. Extraction functions for functions and variables from AST

def extract_functions(root: Node, source: bytes, file_path: str) -> List[FunctionNode]:
    functions: List[FunctionNode] = []
    # Query.matches returns list of tuples (int, dict[str, list[Node]])
    for _, captures in FUNCTION_QUERY.matches(root):
        for func_node in captures.get('func_def', []):
            functions.append(FunctionNode(
                file_path=Path(file_path),
                class_name=find_enclosing_class(func_node),
                decorators=extract_decorators(func_node, source),
                name=func_node.child_by_field_name('name').text.decode('utf-8'),
                params=parse_parameters(func_node.child_by_field_name('parameters'), source),
                docstring=extract_docstring(func_node, source),
                body=extract_body_source(func_node, source)
            ))
    return functions


def extract_variables(root: Node, source: bytes, file_path: str) -> List[VariableNode]:
    variables: List[VariableNode] = []
    # Static fields
    for _, captures in VARIABLE_QUERY.matches(root):
        var_names = captures.get('var_name', [])
        values = captures.get('value', [])
        for var_node, val_node in zip(var_names, values):
            name = var_node.text.decode('utf-8')
            assignment = var_node.parent
            class_name = find_enclosing_class(assignment) or ''
            value = get_node_text(val_node, source)
            variables.append(VariableNode(
                file_path=Path(file_path),
                class_name=class_name,
                is_static=True,
                name=name,
                annotation=None,
                value=value
            ))

    # Instance fields initialized inside __init__
    for _, captures in INIT_QUERY.matches(root):
        self_objs = captures.get('self_obj', [])
        var_names = captures.get('var_name', [])
        values = captures.get('value', [])
        for self_obj, var_node, val_node in zip(self_objs, var_names, values):
            if self_obj.text.decode('utf-8') == 'self':
                name = var_node.text.decode('utf-8')
                assignment = var_node.parent
                class_name = find_enclosing_class(assignment) or ''
                value = get_node_text(val_node, source)
                variables.append(VariableNode(
                    file_path=Path(file_path),
                    class_name=class_name,
                    is_static=False,
                    name=name,
                    annotation=None,
                    value=value
                ))
    return variables


# 7. Walk through repository and collect all nodes

def parse_repository(repo_path: Path) -> List[Union[FunctionNode, VariableNode]]:
    all_nodes: List[Union[FunctionNode, VariableNode]] = []
    for root, _, files in os.walk(repo_path):
        for fname in files:
            if fname.endswith('.py'):
                full_path = Path(root) / fname
                if not full_path.exists():
                    logger.warning(f"File [{full_path}] is broken or inaccessible.")
                    continue
                src = read_source_file(full_path)
                tree = parser.parse(src)
                all_nodes.extend(extract_functions(tree.root_node, src, str(full_path)))
                all_nodes.extend(extract_variables(tree.root_node, src, str(full_path)))
    return all_nodes
