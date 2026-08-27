"""argparse help 文字列中の未エスケープ % を全件検出する使い捨て診断 (2026-08-24)。"""
import ast

path = "scripts/visualize_advantage_overlay.py"
src = open(path, encoding="utf-8").read()
tree = ast.parse(src)


class V(ast.NodeVisitor):
    def visit_Call(self, node: ast.Call) -> None:
        if isinstance(node.func, ast.Attribute) and node.func.attr == "add_argument":
            for kw in node.keywords:
                if kw.arg == "help":
                    try:
                        val = ast.literal_eval(kw.value)
                    except Exception:
                        val = None
                    if isinstance(val, str):
                        try:
                            val % {}
                        except Exception as e:
                            print(f"BAD at line {node.lineno}: {e}")
                            print(repr(val[:300]))
        self.generic_visit(node)


V().visit(tree)
