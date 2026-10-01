r"""Refuse to build or package a source whose control sequences were eaten.

A backslash escape written through a shell or a non-raw Python string turns
"\rsq{}" into a carriage return followed by "sq{}". It still compiles -- "sq{}"
is ordinary text -- so no LaTeX error or warning ever appears and the PDF
quietly prints "sq = 0.8759". Three forms have occurred in this project:

  * the carriage return promoted to a line break, leaving a line that starts
    with the tail of a control word ("sq{} = ...", "ef{sec:...}");
  * a lone carriage return left in the middle of a line;
  * other control characters (tab, backspace, form feed) from "\t", "\b", "\f".

build.ps1 runs this before compiling anything, and e5 before packaging.

    python lint_sources.py          exit status 1 if anything is found
"""
import glob
import os
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
PKG = os.path.join(os.path.dirname(os.path.dirname(HERE)), "Revision_R2")


def main():
    files = sorted(glob.glob(os.path.join(PKG, "*.tex")) +
                   glob.glob(os.path.join(PKG, "generated", "*.tex")))
    # the marked copy is generated from manuscript.tex and checked through it
    files = [f for f in files if not f.endswith("manuscript_marked.tex")]
    vocabulary = set()
    raw = {}
    for f in files:
        raw[f] = open(f, encoding="utf-8", newline="").read()
        vocabulary |= set(re.findall(r"\\([a-zA-Z]+)", raw[f]))

    problems = []
    for f, text in raw.items():
        name = os.path.relpath(f, PKG)
        for m in re.finditer(r"\r(?!\n)", text):
            line = text.count("\n", 0, m.start()) + 1
            problems.append((name, line, "lone carriage return", text[m.start() + 1:m.start() + 40]))
        for m in re.finditer(r"[\x00-\x08\x0b\x0c\x0e-\x1f]", text):
            line = text.count("\n", 0, m.start()) + 1
            problems.append((name, line, "control character 0x%02x" % ord(m.group(0)),
                             text[m.start() + 1:m.start() + 40]))
        for n, line in enumerate(text.replace("\r\n", "\n").split("\n"), 1):
            m = re.match(r"[a-z][a-zA-Z]*", line)
            if m and any(c + m.group(0) in vocabulary for c in "rtbfvan0"):
                problems.append((name, n, "line starts with the tail of a control word", line[:40]))

    if problems:
        print("lint: %d destroyed control sequence(s) -- not building:" % len(problems))
        for name, n, what, ctx in problems:
            print("  %s:%d  %s   ...%s" % (name, n, what, ctx.replace("\n", " ")))
        return 1
    print("lint: %d source files clean" % len(files))
    return 0


if __name__ == "__main__":
    sys.exit(main())
