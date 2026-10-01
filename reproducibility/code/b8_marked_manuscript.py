"""Build manuscript_marked.tex as a real comparison against the submitted paper.

The baseline is the file the editor received: main.tex from the Overleaf package
returned with the decision letter. It uses the same sn-jnl class as the revision,
so the comparison is body against body rather than format against format.

Before diffing, the package is linted for control sequences destroyed by a Python
escape. A non-raw string literal turns "\\ref" into a carriage return followed by
"ef", which still compiles -- "ef{x}" is ordinary text -- so no build ever
complains and the PDF quietly reads "efx". That failure is silent, it has
happened in this project, and it is cheap to test for, so it is tested for here
and the run stops if anything is found.

Four adjustments are made to the two inputs, all so the comparison compiles and
reads as a comparison:

  * cross-references in the baseline are resolved to the numbers they carried in
    the submitted PDF, because latexdiff comments out a \\label inside a deleted
    float and any label the revision reuses would clash;
  * generated table bodies are pasted into the revision, so the comparison shows
    which rows changed rather than one opaque token;
  * \\revmark spans are unwrapped, so words inside them are compared as words;
  * floats and tabulars are compared as whole objects. Every table here was
    rebuilt from new numbers and the figures were replaced, so a word-level
    comparison inside them produces noise and, where the float order changed,
    captions that swallow whole paragraphs.

    python b8_marked_manuscript.py
"""

import os
import re
import shutil
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ANALYSIS = os.path.dirname(HERE)
REVISION = os.path.dirname(ANALYSIS)
ROOT = os.path.dirname(REVISION)
PKG = os.path.join(REVISION, "Revision_R1_new")
SUBMITTED_DIR = os.path.join(
    REVISION, "757-1 迟超 Scientific Reports 一审意见下发 (2)",
    "Chemical_language_models_for_antioxidant_activity__-1_")
SUBMITTED = os.path.join(SUBMITTED_DIR, "main.tex")
# A copy of the submitted source lives in the package too, so the marked-up copy
# can still be rebuilt if the decision folder is moved or cleared out.
if not os.path.exists(SUBMITTED):
    SUBMITTED_DIR = os.path.join(PKG, "baseline")
    SUBMITTED = os.path.join(SUBMITTED_DIR, "main.tex")
REVISED = os.path.join(PKG, "manuscript.tex")
MARKED = os.path.join(PKG, "manuscript_marked.tex")
FIGDIR = os.path.join(PKG, "figures_original")
WORK = os.path.join(PKG, ".build", "diff")

DOCS = ["manuscript.tex", "supporting_info.tex", "response.tex", "cover_letter.tex",
        "preamble.tex", "si_numbers.tex", "author_metadata.tex"]

HEADER = r"""
\section*{About this marked-up copy}

This is a machine-generated comparison against the manuscript submitted for
review. Text added in revision is \textcolor{blue}{\uwave{underlined in blue}};
text removed is \textcolor{red}{\sout{struck through in red}}.

Every figure and table of the original submission was replaced in this revision.
Each one is therefore shown where it stood, its image crossed out in red and its
caption marked \textcolor{red}{[\dots\ deleted in this revision]}, followed by
the figure or table that replaced it, whose caption is marked
\textcolor{blue}{[Added in this revision]}. Figures and tables are compared as
whole objects rather than word by word, since none of them survived in edited
form. Cross-references inside deleted text are printed as the numbers those items
carried in the original submission, because the items themselves are gone.
"""


# ------------------------------------------------------------------- the lint
def lint():
    """Refuse to build on a source whose control sequences were eaten."""
    vocabulary = set()
    texts = {}
    for name in DOCS:
        path = os.path.join(PKG, name)
        if not os.path.exists(path):
            continue
        texts[name] = open(path, encoding="utf-8").read()
        vocabulary |= set(re.findall(r"\\([a-zA-Z]+)", texts[name]))

    # These files are written one paragraph per line, so a line that starts
    # mid-sentence means a line break landed where the source had none.
    broken = []
    for name, text in texts.items():
        for n, line in enumerate(text.replace("\r\n", "\n").split("\n"), 1):
            m = re.match(r"[a-z][a-zA-Z]*", line)
            if not m:
                continue
            word = m.group(0)
            repairs = [c for c in "rtbfvan0" if c + word in vocabulary]
            if repairs:
                broken.append((name, n, "\\" + repairs[0] + word, line[:80]))
    if broken:
        print("destroyed control sequences -- fix these before building:")
        for name, n, macro, line in broken:
            print("  %s:%d  should be %s" % (name, n, macro))
            print("      %s" % line)
        raise SystemExit(1)
    print("lint: no destroyed control sequences in %d source files" % len(texts))


# --------------------------------------------------------------- the baseline
def prepare_baseline():
    old = open(SUBMITTED, encoding="utf-8").read()

    numbers, counters = {}, {"figure": 0, "table": 0}
    for m in re.finditer(r"\\begin\{(figure|table)\}(.*?)\\end\{\1\}", old, re.S):
        kind, body = m.group(1), m.group(2)
        counters[kind] += 1
        for lab in re.findall(r"\\label\{([^}]*)\}", body):
            numbers[lab] = "%s~%d of the original submission" % (
                kind.capitalize(), counters[kind])

    unresolved = set(re.findall(r"\\ref\{([^}]*)\}", old)) - set(numbers)
    if unresolved:
        raise SystemExit("no original number for %s" % ", ".join(sorted(unresolved)))

    for lab, shown in numbers.items():
        # "Figure~\ref{fig:dist}" would otherwise read "Figure Figure 1 of ..."
        old = re.sub(r"(Figure|Table)~\\ref\{%s\}" % re.escape(lab), shown, old)
        old = old.replace("\\ref{%s}" % lab, shown)
        old = old.replace("\\label{%s}" % lab, "")
    print("baseline: %d cross-references resolved to the original numbering" % len(numbers))

    os.makedirs(FIGDIR, exist_ok=True)
    kept = 0
    for name in re.findall(r"\\includegraphics\[[^\]]*\]\{([^}]*)\}", old):
        src = os.path.join(SUBMITTED_DIR, name)
        if os.path.exists(src):
            shutil.copy2(src, os.path.join(FIGDIR, name))
            kept += 1
    old = re.sub(r"(\\includegraphics\[[^\]]*\]\{)", r"\1figures_original/", old)
    print("baseline: %d original figures kept for the marked-up copy" % kept)
    return old


# --------------------------------------------------------------- the revision
def prepare_revision():
    new = open(REVISED, encoding="utf-8").read()

    tag = "\\revmark{"
    out, i, spans = [], 0, 0
    while True:
        j = new.find(tag, i)
        if j < 0:
            out.append(new[i:])
            break
        out.append(new[i:j])
        k, depth = j + len(tag), 1
        while depth and k < len(new):
            depth += {"{": 1, "}": -1}.get(new[k], 0)
            k += 1
        out.append(new[j + len(tag):k - 1])
        i, spans = k, spans + 1
    new = "".join(out)
    print("revision: %d revmark spans unwrapped" % spans)

    def paste(m):
        path = os.path.join(PKG, m.group(1).replace("/", os.sep) + ".tex")
        return open(path, encoding="utf-8").read().rstrip()

    new, pasted = re.subn(r"\\inputrows\{([^}]*)\}", paste, new)
    print("revision: %d generated table bodies pasted in" % pasted)
    return new


# ------------------------------------------------------------------ safe list
def safe_commands():
    """Macros latexdiff must keep whole instead of breaking the markup around."""
    names = {"pic", "rsq", "qsq", "dpph", "etal", "ie", "eg", "revmark",
             "cmidrule", "multicolumn", "multirow", "toprule", "midrule",
             "bottomrule", "addlinespace"}
    for rel in ("si_numbers.tex", "generated/results_macros.tex",
                "generated/crossrefs.tex", "manuscript.tex"):
        path = os.path.join(PKG, rel.replace("/", os.sep))
        if os.path.exists(path):
            names |= set(re.findall(r"\\(?:new|renew|provide)command\*?\s*\{?\\([a-zA-Z]+)",
                                    open(path, encoding="utf-8").read()))
    return sorted(names)


# ------------------------------------------------- floats deleted in revision
def _balanced(text, start):
    """End index of the brace group that starts at text[start] == '{'."""
    depth, i = 0, start
    while i < len(text):
        depth += {"{": 1, "}": -1}.get(text[i], 0)
        if depth == 0:
            return i
        i += 1
    raise ValueError("unbalanced group")


def _mark_float(source, kind, number):
    """Re-typeset a float of the original submission as visibly deleted."""
    source = re.sub(r"\\includegraphics(\[[^\]]*\])?\{",
                    lambda m: "\\DIFdelincludegraphics" + (m.group(1) or "") + "{", source)

    j = source.find("\\caption{")
    if j >= 0:
        k = _balanced(source, j + len("\\caption"))
        inner = source[j + len("\\caption{"):k]
        banner = ("\\textcolor{red}{[%s~%d of the original submission, deleted in this "
                  "revision]}\\ " % (kind.capitalize(), number))
        source = source[:j] + "\\caption{" + banner + "\\textcolor{red}{" + inner + "}}" \
            + source[k + 1:]

    # a tabular cannot be struck through, so the deleted one is set in red
    source = source.replace("\\begin{tabular}", "{\\color{red}\\begin{tabular}")
    source = source.replace("\\end{tabular}", "\\end{tabular}}")
    source = source.replace("\\begin{tablenotes}", "{\\color{red}\\begin{tablenotes}")
    source = source.replace("\\end{tablenotes}", "\\end{tablenotes}}")
    return source


def restore_deleted_floats(diff):
    """Put back the figures and tables latexdiff comments out.

    A float that the revision removed is emitted by latexdiff as a run of
    "%DIFDELCMD <" lines, so it vanishes from the marked-up copy altogether and
    the reader cannot see that it used to be there. Every float of the original
    submission was replaced in this revision, which is exactly the thing worth
    showing, so each one is uncommented and re-typeset with its image crossed
    out in red and its caption labelled and coloured.
    """
    marker = "%DIFDELCMD < "
    lines = diff.split("\n")
    out, counters, restored = [], {"figure": 0, "table": 0}, 0
    i = 0
    while i < len(lines):
        m = re.match(r"^(.*?)%DIFDELCMD < (.*)$", lines[i])
        opener = re.search(r"\\begin\{(figure|table)\}", m.group(2)) if m else None
        if not opener:
            out.append(lines[i])
            i += 1
            continue

        kind = opener.group(1)
        body, j, closed = [m.group(2)], i + 1, False
        while j < len(lines):
            if not lines[j].strip():
                body.append("")
                j += 1
                continue
            inner = re.match(r"^%DIFDELCMD < (.*)$", lines[j])
            if not inner:
                break
            body.append(inner.group(1))
            j += 1
            if "\\end{%s}" % kind in inner.group(1):
                closed = True
                break
        if not closed:
            out.append(lines[i])
            i += 1
            continue

        counters[kind] += 1
        source = "\n".join(body)
        source = source[:source.rindex("\\end{%s}" % kind) + len("\\end{%s}" % kind)]
        out.append(m.group(1))
        out.append(_mark_float(source, kind, counters[kind]))
        restored += 1
        i = j + 1

    print("restored %d deleted float(s) as visibly removed" % restored)
    return "\n".join(out)


def annotate_added_floats(diff):
    """Say so on floats the revision added.

    latexdiff frames an added image in blue, but an added table gets no mark at
    all, so a table built entirely from new numbers reads as though it had been
    carried over. Both kinds get a banner on the caption.
    """
    token = re.compile(r"\\DIFaddbegin(?!FL)|\\DIFaddend(?!FL)|\\begin\{(figure|table)\}")
    inserts, depth, marked = [], 0, 0
    for m in token.finditer(diff):
        if m.group(0) == "\\DIFaddbegin":
            depth += 1
        elif m.group(0) == "\\DIFaddend":
            depth = max(0, depth - 1)
        elif depth:
            j = diff.find("\\caption{", m.end())
            if j < 0:
                continue
            inserts.append(j + len("\\caption{"))
            marked += 1

    banner = "\\textcolor{blue}{[Added in this revision]}\\ "
    for j in sorted(inserts, reverse=True):
        diff = diff[:j] + banner + diff[j:]
    print("marked %d added float(s)" % marked)
    return diff


def main():
    lint()
    os.makedirs(WORK, exist_ok=True)
    old_path = os.path.join(WORK, "old.tex")
    new_path = os.path.join(WORK, "new.tex")
    safe_path = os.path.join(WORK, "safecmd.txt")
    open(old_path, "w", encoding="utf-8").write(prepare_baseline())
    open(new_path, "w", encoding="utf-8").write(prepare_revision())
    names = safe_commands()
    open(safe_path, "w", encoding="utf-8").write("\n".join(names) + "\n")
    print("latexdiff: %d safe commands" % len(names))

    argv = ["latexdiff", "--encoding=utf8", "--type=UNDERLINE", "--floattype=FLOATSAFE",
            "--append-safecmd=" + safe_path,
            # sn-jnl takes the abstract as a command argument, which latexdiff
            # would otherwise pass through unmarked and silently show only the
            # revised wording
            "--append-textcmd=abstract",
            "--exclude-textcmd=section,subsection,subsubsection,paragraph",
            "--config", "PICTUREENV=picture|DIFnomarkup|figure|table|tabular",
            old_path, new_path]
    r = subprocess.run(argv, capture_output=True, text=True, encoding="utf-8")
    if r.returncode != 0 or not r.stdout:
        sys.stderr.write(r.stderr)
        raise SystemExit("latexdiff failed (rc=%s)" % r.returncode)
    if r.stderr.strip():
        print("latexdiff said:", r.stderr.strip())

    diff = r.stdout
    if "\\maketitle" not in diff:
        raise SystemExit("no \\maketitle in the comparison; refusing to write it")
    diff = annotate_added_floats(diff)
    diff = restore_deleted_floats(diff)
    diff = diff.replace("\\maketitle", "\\maketitle\n" + HEADER, 1)
    open(MARKED, "w", encoding="utf-8", newline="\n").write(diff)

    added = diff.count("\\DIFaddbegin") + diff.count("\\DIFaddbeginFL")
    deleted = diff.count("\\DIFdelbegin") + diff.count("\\DIFdelbeginFL")
    print("wrote %s\n  %d added passages, %d deleted passages"
          % (os.path.relpath(MARKED, ROOT), added, deleted))


if __name__ == "__main__":
    main()
