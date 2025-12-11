#!/bin/bash

set -eou pipefail

sync_f="$1"
force_update="${2:-UNSET}"

case "$sync_f" in
*.ipynb)
    in_f_ext="ipynb"
    pair_f_ext="py"
    ;;
*.py)
    in_f_ext="py"
    pair_f_ext="ipynb"
    ;;
*)
    exit 1
    ;;
esac

sync_basename="$(basename -s ".${in_f_ext}" "$sync_f")"
sync_dirname="$(dirname "$sync_f")"
paired_f="${sync_dirname}/${sync_basename}.${pair_f_ext}"

if [ "$in_f_ext" == "ipynb" ] && [ ! -s "$paired_f" ]; then
    jupytext --set-formats ipynb,py:percent "$sync_f"
    jupytext --to py:percent --from ipynb --opt comment_magics=true "$sync_f"
    echo "Creating .ipynb file"
elif [ "$in_f_ext" == "py" ] && [ ! -s "$paired_f" ]; then
    jupytext --set-formats ipynb,py:percent "$sync_f"
    jupytext --to ipynb --from py:percent --opt comment_magics=true "$sync_f"
    echo "Creating .py file"
fi

ipynb_f="${sync_basename}.ipynb"
py_f="${sync_basename}.py"

# If the user wants to force an update on the notebook or the script, then change the
# mtime of the not-to-be-modified file, and run the sync as usual.
if [ ! "$force_update" == "UNSET" ]; then
    case "$force_update" in

    "-fupy" | "--force-update-py" | '-fuscript' | '--force-update-script')
        echo "Forcing update to python script ${py_f}"
        touch "$ipynb_f"
        ;;
    "-fuipynb" | "--force-update-ipynb" | "-funb" | "--force-update-nb")
        echo "Forcing update to jupyter notebook ${ipynb_f}"
        touch "$py_f"
        ;;
    *)
        echo "ERROR: Invalid argument ${force_update}"
        exit 1
        ;;
    esac
fi

# bat has a nicer output formatting for diffs, so use that if available.
if command -v -- "bat" >/dev/null 2>&1; then
    cat_cmd="bat --language diff --pager never"
else
    cat_cmd="cat"
fi

jupytext --sync --show-changes "$sync_f" | $cat_cmd

# To do the initial export:
# jupytext --to ipynb --from py:percent \
#     --opt comment_magics=true \
#     inr.py
# jupytext --set-formats ipynb,py:percent inr.py
