"""tldr-pages -> command catalog, plus the gate's test queries. Shared by the gate notebook and the app."""
import glob
import os
import re

PLATFORMS = ("common", "linux")


def parse_page(path):
    """-> list of {tool, desc, cmd} from one tldr page."""
    tool, out, desc = None, [], None
    for line in open(path, encoding="utf-8"):
        line = line.rstrip("\n")
        if line.startswith("# "):
            tool = line[2:].strip()
        elif line.startswith("- "):
            desc = line[2:].strip().rstrip(":")
        elif line.startswith("`") and line.endswith("`") and desc:
            out.append({"tool": tool, "desc": desc, "cmd": line[1:-1]})
            desc = None
    return out


def load_catalog(tldr_root):
    seen, cat = set(), []
    for plat in PLATFORMS:
        for p in sorted(glob.glob(os.path.join(tldr_root, "pages", plat, "*.md"))):
            for e in parse_page(p):
                text = f"{e['desc']}: {e['cmd']}"   # what the action head embeds
                if text not in seen:
                    seen.add(text)
                    cat.append({**e, "platform": plat, "text": text})
    return cat


# ---------------------------------------------------------------------------------------------
# Test queries. Written in everyday developer phrasing (NOT tldr's wording), before any model ran.
# gold: regex on the command string; a hit anywhere in top-k counts.
# pair: polarity pairs share an id; the two halves ask for opposite things with near-identical topic.
QUERIES = [
    # plain lookups
    ("disk_free",   "how much disk space is left on my machine",            r"^df\b", None),
    ("port_owner",  "find out which process is using port 8080",            r"^(lsof -i|netstat\b|fuser .*/tcp|ss .*(src|dst|listening|:\{\{|port))", None),
    ("untar",       "unzip a .tar.gz file",                                 r"^tar \S*x", None),
    ("count_lines", "count the number of lines in a file",                  r"^wc .*\[-l\|--lines\]", None),
    ("grep_rec",    "search for a word in every file under this folder",    r"^(grep .*(-r|recursive)|rg\b|ag\b|ack\b)", None),
    ("chmod_x",     "make my script runnable",                              r"^chmod .*\+x", None),
    ("branch_mv",   "rename the current git branch",                        r"^git branch .*(-m|move)", None),
    ("blame",       "see who last edited each line of a file",              r"^git blame(\s|$)", None),
    ("download",    "download a file from a URL",                           r"^(wget\b|curl .*(-O|-o|remote-name|output))", None),
    ("folder_size", "how big is this folder",                               r"^(du|dust|ncdu|gdu)\b", None),
    ("docker_ps",   "list the docker containers that are running",          r"^docker (compose )?(\{\{\[)?(ps|container ls)", None),
    ("venv",        "create a python virtual environment",                  r"(-m venv|^virtualenv\b|^uv venv|^python.*venv)", None),
    # polarity pairs: same topic, opposite intent
    ("undo_keep",   "undo my last commit but keep my changes",              r"^git reset (HEAD~|.*--soft|.*--mixed)", "P1"),
    ("undo_trash",  "undo my last commit and throw the changes away completely", r"^git reset .*--hard", "P1"),
    ("stage_all",   "stage all my changes in git",                          r"^git add .*(-A|all|\{\{\.\}\}|\s\.$)", "P2"),
    ("unstage_all", "unstage everything I added in git",                    r"^git (reset$|restore .*staged)", "P2"),
    ("br_del_loc",  "delete a git branch on my machine",                    r"^git branch .*(-d|-D|delete)", "P3"),
    ("br_del_rem",  "delete a git branch on the remote",                    r"^git push .*(delete|:)", "P3"),
    ("scp_up",      "copy a file from my laptop to a remote server",        r"^scp .*local.* .*remote", "P4"),
    ("scp_down",    "copy a file from a remote server to my laptop",        r"^scp .*remote\S*:\S+ .*local", "P4"),
]


def is_gold(cmd, pattern):
    return re.search(pattern, cmd) is not None
