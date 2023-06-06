import copy
import logging
import os
import sys
import warnings

import colorama
colorama.init()

from .utils import jsonable

DEFAULT_COLOR = {
    "DEBUG": colorama.Fore.BLUE,
    "INFO": colorama.Fore.CYAN,
    "WARNING": colorama.Fore.YELLOW,
    "ERROR": colorama.Fore.RED,
    "CRITICAL": colorama.Fore.MAGENTA,
}
LOGFORMAT = "%(asctime)s|%(levelchar)s|%(name)s(%(filename)s:%(lineno)d)|%(message)s"
TIMEFORMAT = "%Y-%m-%d %H:%M:%S"

# ANSI colour code constants, in case of can't use colorama
ANSI_RESET = "\033[0m"
ANSI_COLOR = "\033[1;%dm"
ANSI_BOLD = "\033[1m"
(BLACK, RED, GREEN, YELLOW, BLUE, MAGENTA, CYAN, WHITE) = [ANSI_COLOR % x for x in range(30, 38)]

_HANDLERS = {}

def reset_handlers(root=None):
    if not root:
        root = ""
    logger = logging.getLogger(root)
    logger.setLevel("WARNING")
    logger.handlers = []
    if root in _HANDLERS:
        _HANDLERS[root].clear()

def list_handlers(root=None):
    if not root:
        root = ""
    return copy.deepcopy(_HANDLERS.get(root, []))

def add_handlers(handlers: list[dict], root=None) -> None:
    """Add log handlers to the logger. This function can be called multiple
    times to append handlers to the logger. Each handler spec should carry a
    `name`, which corresponds to a `name_handler()` function defined in this
    submodule.

    Example:
        root = logging.getLogger(__file__)
        handlers = [
            {"name":"console", "level":"DEBUG"},
            {"name":"null", "level":"CRITICAL"},
            {"name":"file", "level":"INFO", "filename":"/tmp/log.txt"},
        ]
        add_handlers(handlers, root)

    Args:
        handlers: A list of dicts where each dict describes one handler, or a
                  single dict for one handler. Each dict should carry a `name`
                  and optionally a `level`. Some specific handler may take
                  other keys, e.g., a file handler would need a filename.
    """
    if not root:
        root = ""
    # validate input
    def validate_spec(handler_dict):
        "Verify one handler dict. Raise exceptions on error"
        # handler spec should be JSON-able, with the name match a handler factory
        # defined here, and if specified a level, its name should be known
        if not jsonable(handler_dict):
            warnings.warn("Non JSON-able spec found: %r" % handler_dict, UserWarning)
        if "name" not in handler_dict:
            raise ValueError("No `name` in handler spec: %r" % handler_dict)
        name = handler_dict["name"]
        if not isinstance(name, str):
            raise ValueError("`name` should be str in handler spec: %r" % handler_dict)
        funcname = name + "_handler"
        if funcname not in globals():
            raise ValueError("Handler %s (factory function %s) does not exist in %s" % (name, funcname, __name__))
        func = globals()[funcname]
        if type(func).__name__ != "function":
            raise ValueError("Identifier %s (for handler %s) is not a function in  %s" % (funcname, name,  __name__))
        if "level" in handler_dict:
            levelname = handler_dict["level"]
            if isinstance(levelname,str) and levelname not in logging._nameToLevel:
                raise ValueError("Level %s is not defined in logging: %r" % (levelname, handler_dict))
        # if anything else is specified in the spec, it should match the
        # signautre of the factory function
        all_args = func.__code__.co_varnames[:func.__code__.co_argcount]
        req_args = set(all_args[:-len(func.__defaults__)])
        opt_args = set(all_args[len(req_args):])
        all_keys = set(handler_dict.keys())
        missing_keys = req_args - all_keys
        unknown_keys = all_keys - opt_args - req_args - {"name"}
        if missing_keys:
            raise ValueError("Keys %s missing in spec: %r" % (list(missing_keys), handler_dict))
        if unknown_keys:
            raise ValueError("Keys %s unrecognized in spec: %r" % (list(unknown_keys), handler_dict))
    if isinstance(handlers, dict):
        handlers = [handlers]
    for spec in handlers:
        validate_spec(spec)
    # create handlers according to specs and add to the logger
    handlers = copy.deepcopy(handlers)
    logger = logging.getLogger(root)
    for spec in handlers:
        func = globals()[spec["name"] + "_handler"]
        kwargs = dict(spec)
        del kwargs["name"]
        handler = func(**kwargs)
        level = logging._nameToLevel[spec["level"]] if isinstance(spec["level"],str) else spec["level"]
        logger.addHandler(handler)
        if level is not None and level < logger.level:
            logger.setLevel(level)
    # keep record of what added
    if root not in _HANDLERS:
        _HANDLERS[root] = []
    _HANDLERS[root].extend(handlers)
    return logger


def null_handler(level=None):
    """Null logging handler, the logging level is irrelevant"""
    return logging.NullHandler()


class ColorFormatter(logging.Formatter):
    """Logging formatter with color and supports `levelchar` as first character
    of `levelname` in the logging format string"""
    def set_color_scheme(self, scheme=DEFAULT_COLOR):
        "Set color scheme for this formatter. If not set, no color will be applied"
        self.color_scheme = scheme
        self.color_reset = colorama.Style.RESET_ALL

    def format(self, record):
        """Logging formatter that knows `levelchar` in format template and may
        apply ANSI color"""
        record.levelchar = record.levelname[0]
        msg = logging.Formatter.format(self, record)
        if hasattr(self, "color_scheme") and record.levelname in self.color_scheme:
            msg = self.color_scheme[record.levelname] + msg + self.color_reset
        return msg

def console_handler(level=None, stdout=False, color=True, logformat=LOGFORMAT, timeformat=TIMEFORMAT):
    """ANSI color console logging handler: Log message will be in color

    Args:
        stdout: If True, log messages to sys.stdout instead of stderr
        color: If False, log message will not use ANSI color
    """
    if stdout:
        handler = logging.StreamHandler(sys.stdout)
    else:
        handler = logging.StreamHandler()
    if level:
        handler.setLevel(level)
    formatter = ColorFormatter(logformat, timeformat)
    if color:
        formatter.set_color_scheme()
    handler.setFormatter(formatter)
    return handler

def file_handler(filename, level=None, logformat=LOGFORMAT, timeformat=TIMEFORMAT):
    handler = logging.FileHandler(filename, encoding="utf8")
    if level:
        handler.setLevel(level)
    handler.setFormatter(ColorFormatter(logformat, timeformat))
    return handler

def stream_handler(fileobj, level=None, logformat=LOGFORMAT, timeformat=TIMEFORMAT):
    """Write log to a file-like object, such as StringIO
    """
    handler = logging.StreamHandler(fileobj)
    if level:
        handler.setLevel(level)
    handler.setFormatter(ColorFormatter(logformat, timeformat))
    return handler

# old function, prefer add_handlers() instead
def get_logger(root=None, level=logging.DEBUG, filename=None, filelevel=None,
               stream=None, console=True, reset=False):
    """Configurating the logging facilities to log possibly to both console and
    file. You should only run this once at the root level and subsequently use
    logging.getLogger() to get a subordinate logger.

    Args:
        root (str): root logger to set, default is the global root
        level (int): the logging level to use for console, default at debug. This also
                accepts string name of the logging level
        filename (str): if provided, log will be appended to the file
        filelevel (int): the logging level to use for file, default is same as console's level
        stream (file-like object): if provided, write log to a stream object,
               such as StringIO buffer
        console (bool): whether to print log to console
        reset (bool): delete all existing log handler of the root logger if set to True
    Returns:
        a python logger object
    """
    from .utils import supports_color

    # reset string-type logging level into integers
    if isinstance(level, str):
        level = _loglevelcode(level)
    if level is None:
        level = logging.INFO
    assert isinstance(level, int)

    if isinstance(filelevel, str):
        filelevel = _loglevelcode(filelevel)
    if filelevel is None:
        filelevel = level
    assert isinstance(filelevel, int)

    if reset:
        reset_handlers(root)

    # prepare handlers
    handlers = []
    if console:
        handlers.append({"name":"console", "level":level, "color":supports_color()})
    if filename:
        handlers.append({"name":"file", "level":filelevel, "filename":filename})
    if stream:
        handlers.append({"name":"stream", "level":filelevel, "fileobj":stream})
    logger = add_handlers(handlers)
    return logger
