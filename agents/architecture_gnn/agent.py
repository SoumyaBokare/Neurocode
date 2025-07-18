import ast
import os
import networkx as nx

class ArchitectureAgent:
    def analyze(self, file_tree: dict = None, root_dir: str = None) -> dict:
        # If file_tree is provided, use it; else, parse Python files in root_dir
        if file_tree:
            graph = nx.DiGraph()
            for module, imports in file_tree.items():
                for imp in imports:
                    graph.add_edge(module, imp)
            return {"adjacency_list": nx.to_dict_of_lists(graph)}
        elif root_dir:
            graph = nx.DiGraph()
            for dirpath, _, filenames in os.walk(root_dir):
                for fname in filenames:
                    if fname.endswith('.py'):
                        fpath = os.path.join(dirpath, fname)
                        with open(fpath, 'r', encoding='utf-8') as f:
                            tree = ast.parse(f.read(), filename=fpath)
                            for node in ast.walk(tree):
                                if isinstance(node, ast.Import):
                                    for n in node.names:
                                        graph.add_edge(fname, n.name)
                                elif isinstance(node, ast.ImportFrom):
                                    if node.module:
                                        graph.add_edge(fname, node.module)
            return {"adjacency_list": nx.to_dict_of_lists(graph)}
        else:
            return {"error": "Provide file_tree or root_dir"}
