import sys
import pathlib

# path
DIR_SRC = pathlib.Path(str(sys.modules["__main__"].__file__)).parent
PATH_ICON = DIR_SRC / "static/favicon.ico"
