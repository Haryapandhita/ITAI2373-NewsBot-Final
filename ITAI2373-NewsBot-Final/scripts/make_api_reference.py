"""Generate ``docs/api_reference.md`` from the docstrings in ``src/`` (keeps docs in sync with code)."""

import ast
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MODULES = [
    "config/settings.py",
    "src/newsbot.py",
    "src/data_processing/data_loader.py", "src/data_processing/text_preprocessor.py",
    "src/data_processing/data_validator.py", "src/data_processing/feature_extractor.py",
    "src/analysis/classifier.py", "src/analysis/sentiment_analyzer.py",
    "src/analysis/ner_extractor.py", "src/analysis/topic_modeler.py",
    "src/language_models/summarizer.py", "src/language_models/embeddings.py", "src/language_models/generator.py",
    "src/multilingual/language_detector.py", "src/multilingual/translator.py",
    "src/multilingual/cross_lingual_analyzer.py",
    "src/conversation/intent_classifier.py", "src/conversation/query_processor.py",
    "src/conversation/response_generator.py",
    "src/utils/visualization.py", "src/utils/evaluation.py", "src/utils/export.py",
]


def signature(fn: ast.FunctionDef) -> str:
    args = []
    a = fn.args
    defaults = [None] * (len(a.args) - len(a.defaults)) + list(a.defaults)
    for arg, d in zip(a.args, defaults):
        if arg.arg in ("self", "cls"):
            continue
        s = arg.arg
        if d is not None:
            s += "=" + ast.unparse(d)
        args.append(s)
    if a.vararg:
        args.append("*" + a.vararg.arg)
    for arg, d in zip(a.kwonlyargs, a.kw_defaults):
        args.append(arg.arg + ("=" + ast.unparse(d) if d is not None else ""))
    ret = " -> " + ast.unparse(fn.returns) if fn.returns else ""
    return f"{fn.name}({', '.join(args)}){ret}"


def first_para(doc: str) -> str:
    if not doc:
        return "*(no docstring)*"
    return doc.strip().split("\n\n")[0].replace("\n", " ")


def main():
    out = ["# NewsBot 2.0 – API Reference", "",
           "*Generated from the source docstrings by `scripts/make_api_reference.py`. "
           "Private helpers (leading underscore) are omitted.*", ""]
    toc = []
    body = []
    for rel in MODULES:
        tree = ast.parse((ROOT / rel).read_text(encoding="utf-8"))
        mod = rel.replace("/", ".").removesuffix(".py")
        anchor = mod.replace(".", "")
        toc.append(f"- [`{mod}`](#{anchor})")
        body += [f"## `{mod}`", "", first_para(ast.get_docstring(tree) or ""), ""]
        for node in tree.body:
            if isinstance(node, ast.ClassDef) and not node.name.startswith("_"):
                body += [f"### class `{node.name}`", "", first_para(ast.get_docstring(node) or ""), ""]
                for item in node.body:
                    if isinstance(item, ast.FunctionDef) and (not item.name.startswith("_") or item.name == "__init__"):
                        if item.name == "__init__" and not item.args.args[1:]:
                            continue
                        body.append(f"- **`{signature(item)}`** – {first_para(ast.get_docstring(item) or '')}")
                body.append("")
            elif isinstance(node, ast.FunctionDef) and not node.name.startswith("_"):
                body += [f"### function `{signature(node)}`", "", first_para(ast.get_docstring(node) or ""), ""]
    (ROOT / "docs" / "api_reference.md").write_text("\n".join(out + ["## Modules", ""] + toc + [""] + body), encoding="utf-8")
    print("wrote docs/api_reference.md")


if __name__ == "__main__":
    main()
