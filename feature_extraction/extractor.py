"""
Module 2 -- AST Feature Extraction

Parses Python source into an Abstract Syntax Tree and, for every function
definition, computes five structural metrics:
    - loc                : lines of code in the function
    - max_nesting_depth   : deepest stack of if/for/while/try blocks
    - num_params          : parameter count
    - num_branches        : count of branching nodes (if/for/while/except/etc.)
    - naming_quality      : ratio of "good" (non-trivial) variable names

Usage as a library:
    from extractor import extract_functions
    features = extract_functions(source_code, file_path="foo.py")
"""

import ast
from dataclasses import dataclass, asdict

# Names shorter than this, or matching this throwaway set, count against
# naming_quality unless they're conventional loop/comprehension counters
# in a very short scope (we keep it simple and just penalize length).
TRIVIAL_NAME_MAX_LEN = 2
TRIVIAL_NAMES = {"tmp", "temp", "val", "data", "obj", "x1", "x2", "foo", "bar"}

BRANCH_NODES = (ast.If, ast.For, ast.While, ast.Try, ast.ExceptHandler, ast.With)
NESTING_NODES = (ast.If, ast.For, ast.While, ast.Try, ast.With)


@dataclass
class FunctionFeatures:
    file_path: str
    function_name: str
    lineno: int
    loc: int
    max_nesting_depth: int
    num_params: int
    num_branches: int
    naming_quality: float  # 0..1, higher = better names

    def to_dict(self):
        return asdict(self)


class FeatureExtractor(ast.NodeVisitor):
    """
    Walks the AST once per function definition. Nesting depth is tracked
    by incrementing a counter on entry to any nesting-capable block and
    decrementing on exit, taking the max seen.
    """

    def __init__(self, file_path: str):
        self.file_path = file_path
        self.results: list[FunctionFeatures] = []

    def visit_FunctionDef(self, node: ast.FunctionDef):
        self._process_function(node)
        self.generic_visit(node)

    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef):
        self._process_function(node)
        self.generic_visit(node)

    def _process_function(self, node):
        loc = self._count_loc(node)
        max_depth = self._max_nesting_depth(node)
        num_params = len(node.args.args) + len(node.args.kwonlyargs)
        num_branches = self._count_branches(node)
        naming_quality = self._naming_quality(node)

        self.results.append(FunctionFeatures(
            file_path=self.file_path,
            function_name=node.name,
            lineno=node.lineno,
            loc=loc,
            max_nesting_depth=max_depth,
            num_params=num_params,
            num_branches=num_branches,
            naming_quality=naming_quality,
        ))

    @staticmethod
    def _count_loc(node) -> int:
        end_line = getattr(node, "end_lineno", None)
        if end_line is None:
            return 1
        return end_line - node.lineno + 1

    @staticmethod
    def _max_nesting_depth(node) -> int:
        def walk(n, depth):
            deepest = depth
            for child in ast.iter_child_nodes(n):
                child_depth = depth + 1 if isinstance(child, NESTING_NODES) else depth
                deepest = max(deepest, walk(child, child_depth))
            return deepest
        return walk(node, 0)

    @staticmethod
    def _count_branches(node) -> int:
        count = 0
        for child in ast.walk(node):
            if isinstance(child, BRANCH_NODES):
                count += 1
        return count

    @staticmethod
    def _naming_quality(node) -> float:
        names = set()
        for child in ast.walk(node):
            if isinstance(child, ast.Name) and isinstance(child.ctx, (ast.Store,)):
                names.add(child.id)
            if isinstance(child, ast.arg):
                names.add(child.arg)

        if not names:
            return 1.0  # no locals to judge -- don't penalize

        good = sum(
            1 for n in names
            if len(n) > TRIVIAL_NAME_MAX_LEN and n.lower() not in TRIVIAL_NAMES
        )
        return round(good / len(names), 4)


def extract_functions(source_code: str, file_path: str = "<unknown>") -> list[dict]:
    """Parse source_code and return a list of per-function feature dicts.
    Silently skips files that fail to parse (e.g. Python 2 syntax, syntax errors)."""
    try:
        tree = ast.parse(source_code)
    except (SyntaxError, ValueError):
        return []

    extractor = FeatureExtractor(file_path)
    extractor.visit(tree)
    return [r.to_dict() for r in extractor.results]


if __name__ == "__main__":
    # Quick self-test
    sample = '''
def add(a, b):
    return a + b

def process(data, x1, temp):
    if data:
        for i in range(10):
            if i > 5:
                while True:
                    try:
                        pass
                    except Exception:
                        pass
    return temp
'''
    for f in extract_functions(sample, "sample.py"):
        print(f)
