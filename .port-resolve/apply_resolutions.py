#!/usr/bin/env python3
"""Apply per-hunk resolution decisions to a file with conflict markers.

Usage: apply_resolutions.py <json-file>
JSON: {"path": "relative/path", "decisions": ["@U", "@F", "literal text...", ...]}
Decision for hunk i (0-based, in file order):
  "@U"  -> take upstream side
  "@F"  -> take fork side
  "@UF" -> upstream side then fork side
  "@FU" -> fork side then upstream side
  other -> literal replacement text (used verbatim)
"""
import sys, json, io

def main():
    spec = json.load(open(sys.argv[1]))
    path = spec["path"]
    txt = open(path, encoding="utf-8").read()
    lines = txt.split("\n")
    out = []
    i = 0
    hunk = 0
    decs = spec["decisions"]
    while i < len(lines):
        if lines[i].startswith("<<<<<<<"):
            start = i
            u = []
            t = []
            mode = "u"
            i += 1
            while i < len(lines) and not lines[i].startswith(">>>>>>>"):
                if lines[i].startswith("======="):
                    mode = "t"
                elif mode == "u":
                    u.append(lines[i])
                else:
                    t.append(lines[i])
                i += 1
            if i >= len(lines):
                sys.exit(f"ERROR: unterminated conflict in {path} at line {start+1}")
            i += 1  # skip >>>>>>> line
            d = decs[hunk]
            if d == "@U":
                out.extend(u)
            elif d == "@F":
                out.extend(t)
            elif d == "@UF":
                out.extend(u); out.extend(t)
            elif d == "@FU":
                out.extend(t); out.extend(u)
            else:
                # literal text: may contain \n escapes
                out.extend(d.split("\n"))
            hunk += 1
        else:
            out.append(lines[i])
            i += 1
    if hunk != len(decs):
        sys.exit(f"ERROR: {path}: applied {hunk} hunks but {len(decs)} decisions given")
    open(path, "w", encoding="utf-8").write("\n".join(out))
    print(f"{path}: {hunk} hunks resolved")

main()
