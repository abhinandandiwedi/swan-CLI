import ast
from pathlib import Path

def generate_map(repo_path="~/-TIDDA-Core"):
    root = Path(repo_path).expanduser().resolve()
    skip_dirs = {"venv", ".venv", ".git", "__pycache__", "CMakeFiles", "colmap_hip_build", "legacy", "test_dataset"}
    
    lines = ["# TIDDA-Core AST Repository Map\n"]
    for py_file in sorted(root.rglob("*.py")):
        if any(part in skip_dirs for part in py_file.parts):
            continue
        rel = py_file.relative_to(root)
        try:
            tree = ast.parse(py_file.read_text(encoding="utf-8", errors="ignore"))
            classes = [n.name for n in tree.body if isinstance(n, ast.ClassDef)]
            funcs = [n.name for n in tree.body if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))]
            summary = []
            if classes:
                summary.append(f"Classes: {', '.join(classes)}")
            if funcs:
                summary.append(f"Funcs: {', '.join(funcs[:8])}")
            info = " | ".join(summary) if summary else "Script/Config"
            lines.append(f"- `{rel}` -> {info}")
        except Exception:
            lines.append(f"- `{rel}` -> (Unparsed script)")

    out_file = root / "docs" / "repo_map.md"
    out_file.parent.mkdir(parents=True, exist_ok=True)
    out_file.write_text("\n".join(lines), encoding="utf-8")
    print(f"✔ AST Repo Map generated: {out_file} ({len(lines)} files indexed)")

if __name__ == "__main__":
    generate_map()
