# -*- coding: utf-8 -*-
# IPython magic command to capture and print cell output at the same time.
# Taken directly from <https://stackoverflow.com/a/70870733>.

from IPython import get_ipython
from IPython.core import magic_arguments
from IPython.core.magic import register_cell_magic
from IPython.utils.capture import capture_output


@magic_arguments.magic_arguments()
@magic_arguments.argument(
    "output",
    type=str,
    default="",
    nargs="?",
    help="""The name of the variable in which to store output.
    This is a utils.io.CapturedIO object with stdout/err attributes
    for the text of the captured output.
    CapturedOutput also has a show() method for displaying the output,
    and __call__ as well, so you can use that to quickly display the
    output.
    If unspecified, captured output is discarded.
    """,
)
@magic_arguments.argument(
    "--no-stderr", action="store_true", help="""Don't capture stderr."""
)
@magic_arguments.argument(
    "--no-stdout", action="store_true", help="""Don't capture stdout."""
)
@magic_arguments.argument(
    "--no-display",
    action="store_true",
    help="""Don't capture IPython's rich display.""",
)
@register_cell_magic
def tee(line, cell):
    args = magic_arguments.parse_argstring(tee, line)
    out = not args.no_stdout
    err = not args.no_stderr
    disp = not args.no_display
    with capture_output(out, err, disp) as io:
        get_ipython().run_cell(cell)
    if args.output:
        get_ipython().user_ns[args.output] = io

    io()
