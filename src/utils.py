from __future__ import print_function
from sys import getsizeof, stderr
from itertools import chain
from collections import deque
try:
    from reprlib import repr
except ImportError:
    pass

import subprocess
from pathlib import Path

def total_size(o, handlers={}, verbose=False):
    """ Returns the approximate memory footprint an object and all of its contents.

    Automatically finds the contents of the following builtin containers and
    their subclasses:  tuple, list, deque, dict, set and frozenset.
    To search other containers, add handlers to iterate over their contents:

        handlers = {SomeContainerClass: iter,
                    OtherContainerClass: OtherContainerClass.get_elements}

    """
    dict_handler = lambda d: chain.from_iterable(d.items())
    all_handlers = {tuple: iter,
                    list: iter,
                    deque: iter,
                    dict: dict_handler,
                    set: iter,
                    frozenset: iter,
                   }
    all_handlers.update(handlers)     # user handlers take precedence
    seen = set()                      # track which object id's have already been seen
    default_size = getsizeof(0)       # estimate sizeof object without __sizeof__

    def sizeof(o):
        if id(o) in seen:       # do not double count the same object
            return 0
        seen.add(id(o))
        s = getsizeof(o, default_size)

        if verbose:
            print(s, type(o), repr(o), file=stderr)

        for typ, handler in all_handlers.items():
            if isinstance(o, typ):
                s += sum(map(sizeof, handler(o)))
                break
        return s

    return sizeof(o)


def create_video(paths_to_images: list[Path], outdir: Path, fps:int = 30):
    list_file = "images.txt"

    with open(list_file, "w") as f:
        for img in paths_to_images:
            f.write(f"file '{img.as_posix()}'\n")
            f.write(f"duration {1/fps:.6f}\n")
    output = outdir / "output.mp4"
    cmd = [
        "ffmpeg", "-y",
        "-f", "concat", "-safe", "0",
        "-i", list_file,
        "-r", str(fps),
        "-pix_fmt", "yuv420p",
        output.as_posix()
    ]

    subprocess.run(cmd, check=True)
    Path(list_file).unlink()
    print(f"Created video {output}")
    

def display_bytesize(n_bytes):
    units = ["B", "KB", "MB", "GB"]
    unit = 0
    out = float(n_bytes)
    while out >= 1024 and unit < len(units) - 1:
        out /= 1024
        unit += 1
    return f"{round(out, 1)} {units[unit]}"

