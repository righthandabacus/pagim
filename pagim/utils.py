#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Application-agnostic utility functions
"""

import io
import json
import logging
import mimetypes
import os
import os.path
import smtplib
import sys


def exception_hook(hooktype="trace"):
    """Replace system's exception hook

    Args:
        hooktype (str): The type of exception hook to register, any of "trace",
            "debug", "local" to mean exception log with traceback and exit only,
            launch post-mortem debugger, and print local variable value of each
            frame, respectively.
    """
    assert hooktype in ["trace", "debug", "local"]
    if hooktype == "trace":
        # reset system default
        sys.excepthook = sys.__excepthook__
    elif hooktype == "debug":
        def debughook(etype, value, tb):
            "Launch post-mortem debugger"
            import traceback, pdb
            traceback.print_exception(etype, value, tb)
            print() # make a new line before launching post-mortem
            pdb.pm() # post-mortem debugger
        sys.excepthook = debughook
    elif hooktype == "local":
        def dumphook(etype, value, tb):
            "Dump local variables at each frame of traceback"
            print_tb_with_local()
            sys.__excepthook__(etype, value, tb)
        sys.excepthook = dumphook

def print_tb_with_local():
    """Print stack trace with local variables. This does not need to be in
    exception. Print is using the system's print() function to stderr.
    """
    import traceback
    tb = sys.exc_info()[2]
    stack = []
    while tb:
        stack.append(tb.tb_frame)
        tb = tb.tb_next()
    traceback.print_exc()
    print("Locals by frame, innermost last", file=sys.stderr)
    for frame in stack:
        print("Frame {0} in {1} at line {2}".format(
            frame.f_code.co_name,
            frame.f_code.co_filename,
            frame.f_lineno), file=sys.stderr)
        for key, value in frame.f_locals.items():
            print("\t%20s = " % key, file=sys.stderr)
            try:
                if '__repr__' in dir(value):
                    print(value.__repr__(), file=sys.stderr)
                elif '__str__' in dir(value):
                    print(value.__str__(), file=sys.stderr)
                else:
                    print(value, file=sys.stderr)
            except:
                print("<CANNOT PRINT VALUE>", file=sys.stderr)

def supports_color():
    """Tells if the console (both stdout and stderr) supports ANSI color using a
    heuristic checking. Code adapted from Django. In Windows platform, a program
    called "ANSICON" is needed.

    Returns:
        True if we think the terminal supports ANSI color code. False otherwise.
    """
    if os.environ.get('TERM') == 'ANSI':
        return True # env var overridden result

    plat = sys.platform
    cygwin = "cygwin" in os.environ.get("HOME", "") # win python running in cygwin terminal
    supported = (plat != 'Pocket PC') and (plat != 'win32' or 'ANSICON' in os.environ)

    # isatty is not always implemented, #6223. Not checking stdout as this is
    # mostly for logging use only.
    is_a_tty = hasattr(sys.stderr, 'isatty') and sys.stderr.isatty()
    return cygwin or (supported and is_a_tty)

def _loglevelcode(levelstring):
    """Convert a logging level string into the integer code.
    Python 3.2+ supports level strings but this way is more flexible

    Args:
        levelstring (str): anything with prefix "all", "not", "debug", "info",
                           "warn", "err", "except", "crit", or "fatal"
    Returns:
        int of the corresponding logging level according to the logging module
    """
    levelmap = {
        "all":   logging.NOTSET,
        "not":   logging.NOTSET,
        "debug": logging.DEBUG,
        "info":  logging.INFO,
        "warn":  logging.WARN,
        "err":   logging.ERROR,
        "except":logging.ERROR,
        "crit":  logging.CRITICAL,
        "fatal": logging.FATAL,
    }
    for prefix, levelcode in levelmap.items():
        if levelstring.lower().startswith(prefix):
            return levelcode
    return None # not recognized

def readxml(filename, retain_namespace=False):
    """Read XML from a file and optionally remove the namespace (default)

    Args:
        filename (str): XML file to read
        retain_namespace (bool): If set to True, the namespace info will be retained. Default False

    Returns:
        lxml etree DOM of the XML
    """
    from lxml import etree
    dom = etree.parse(os.path.expanduser(filename))
    if not retain_namespace:
        # XSLT from https://stackoverflow.com/questions/4255277
        xslt = """
            <xsl:stylesheet version="1.0" xmlns:xsl="http://www.w3.org/1999/XSL/Transform">
                <xsl:output method="xml" indent="no"/>

                <xsl:template match="/|comment()|processing-instruction()">
                    <xsl:copy>
                      <xsl:apply-templates/>
                    </xsl:copy>
                </xsl:template>

                <xsl:template match="*">
                    <xsl:element name="{local-name()}">
                      <xsl:apply-templates select="@*|node()"/>
                    </xsl:element>
                </xsl:template>

                <xsl:template match="@*">
                    <xsl:attribute name="{local-name()}">
                      <xsl:value-of select="."/>
                    </xsl:attribute>
                </xsl:template>
            </xsl:stylesheet>
        """
        xslt_doc = etree.parse(io.BytesIO(bytes(xslt, "utf-8")))
        transform = etree.XSLT(xslt_doc)
        dom = transform(dom)
    return dom

def readkeyval(filename):
    """Load a text file of key=val lines and return a dictionary. Comments can
    be started with # char and run up to the end of line. Supposed to be used as
    a config file

    Args:
        filename (str): Filename, assumed accessible by open()
    Returns:
        Dictionary of the corresponding key=val context
    """
    COMMENT_CHAR, OPTION_CHAR = '#', '='
    options = {}
    with open(os.path.expanduser(filename), 'r') as fp:
        for line in fp:
            # remove comments from a line
            if COMMENT_CHAR in line:
                line, _ = line.split(COMMENT_CHAR, 1)
            # parse key=value
            if OPTION_CHAR in line:
                option, value = line.split(OPTION_CHAR, 1)
                options[option.strip()] = value.strip()
    return options

def readyaml(filename):
    """Load a YAML file and return a dictionary. Supposed to be used as a config
    file.

    Args:
        filename (str): Filename, assumed accessible and in YAML format

    Returns:
        Dictionary of the YAML file content

    Raises:
        IOError if cannot read filename, AssertionError if the YAML read is not a dictionary
    """
    import yaml
    data = yaml.load(open(os.path.expanduser(filename)))
    assert isinstance(data, dict)
    return data

def email(sender, recipient, subject, body, smtphost, smtpport=25,
          priority=3, cc=None, attachment=None, asbyte=False):
    """Send email

    Args:
        sender (str): email of sender
        recipient (str): email of recipient
        subject (str): subject line
        body (st): email body
        smtpHost (str): The SMTP host name
        smtpPort (int): SMTP port number, default 25
        priority (int): Priority of email, default 3
        cc (str): email of CC recipients
        attachment (list): Paths to files to attach in the email
        asbyte (bool): If set to True, no email will be sent but the formatted
                       email is returned as string
    Returns:
        The email message formatted for SMTP if asbyte is True, otherwise nothing
    """
    # TODO add bcc support
    if isinstance(priority, str):
        if priority.lower().startswith("high"):
            priority = 1
        elif priority.lower().startswith("low"):
            priority = 5
    assert not attachment or isinstance(attachment, list)
    assert isinstance(priority, int) and 1 <= priority <= 5
    assert isinstance(smtpport, int)

    from email.message import EmailMessage

    msg = EmailMessage()
    msg["From"] = sender
    msg["To"] = recipient
    msg["Subject"] = subject
    msg.set_content(body)
    if priority != 3:
        msg["X-Priority"] = priority # <3 means high and >3 means low
    if cc:
        msg['Cc'] = cc
    for path in attachment or []:
        ctype, encoding = mimetypes.guess_type(path)
        if ctype is None or encoding is not None:
            ctype = 'application/octet-stream'
        maintype, subtype = ctype.split('/', 1)
        filename = os.path.split(path)[-1]
        with open(path, "rb") as fp:
            msg.add_attachment(fp.read(), maintype=maintype, subtype=subtype, filename=filename)
    if not asbyte:
        smtp = smtplib.SMTP(smtphost, smtpport)
        smtp.send_message(msg)
        smtp.quit()
    else: # for debug, return message as string
        from email.policy import SMTP
        return msg.as_bytes(policy=SMTP)

def curl(url, data=None, method='get'):
    """Download one file from a web URL. The request is stateless, just like
    what the cURL tool would do.

    Args:
        url (str): URL to request
        data (dict): data to pass on into the request, such as the post body
        method (str): one of the six HTTP methods, "get", "put", "post",
                      "delete", "head", "options"
    Returns:
        requests.response object. We can get the content in binary or text using
        response.content or response.text respectively
    """
    import requests
    assert method in ["get", "put", "post", "delete", "head", "options"]
    requestfunction = getattr(requests, method)
    params = {}
    if data:
        params["data"] = data
    return requestfunction(url, **params)

def sieve(iterable, indicator):
    """Split an iterable into two lists by a boolean indicator function. Unlike
    `partition()` in iters.py, this does not clone the iterable twice. Instead,
    it run the iterable once and return two lists.

    Args:
        iterable: iterable of finite items. This function will scan it until the
                  end
        indicator: an executable function that takes an argument and returns a
                   boolean
    Returns:
        A tuple (positive, negative), which each items are from the iterable and
        tested with the indicator function
    """
    positive = []
    negative = []
    for item in iterable:
        (positive if indicator(item) else negative).append(item)
    return positive, negative

def flatten(sequence, types=None, checker=lambda x:hasattr(x,'__iter__')):
    """Flatten a sequence. By default, a sequence will be flattened until no element has __iter__
    attribute.

    Args:
        types: a data type or a tuple of data types. If provided, only elements of these types will
               be flattened
        checker: a function. If types is not provided, this checker will tell if an element should
                 be flattened

    Returns:
        This is a generator that recursively yields all elements in the input sequence
    """
    for x in sequence:
        if (types and isinstance(x, types)) or (not types and checker(x)):
            for z in flatten(x):
                yield z
        else:
            yield x

# strip down an input dict to keep only some specified keys
subdict = lambda _dict, _keys: {k:v for k,v in _dict.items() if k in _keys}

def jsonable(data):
    """Tell if the input data is a JSON-able type, which contains only the following recursively:
    str, int, float, bool, None, list, dict"""
    basictypes = (str, int, float, bool)
    if isinstance(data, dict):
        return all(jsonable(k) for k in data.keys()) and all(jsonable(v) for v in data.values())
    if isinstance(data, (tuple, list, set, frozenset)):
        return all(jsonable(x) for x in data)
    return data is None or isinstance(data, basictypes)

def attr2dict(obj):
    """Convert the object's attributes into a dict recursively, until everything is in a jsonable
    type"""
    if obj is None or isinstance(obj, (bytes, str, int, float, bool)):
        # basic type, return intact
        return obj
    elif isinstance(obj, dict):
        ret = {k:attr2dict(v) for k,v in obj.items()}
    elif isinstance(obj, (tuple, list, set, frozenset)):
        ret = [attr2dict(x) for x in obj]
    else:
        ret = {}
        for attr in dir(obj):
            if not attr.startswith("_"):
                ret[attr] = attr2dict(getattr(obj, attr))
    return ret

def jsondumps(obj, **kwargs):
    """JSON serialize an object with better default than json.dumps()"""
    kwargs["ensure_ascii"] = kwargs.get("ensure_ascii", False)
    kwargs["default"] = kwargs.get("default", attr2dict)
    return json.dumps(obj, **kwargs)

# vim:set ts=4 sw=4 sts=4 tw=100 fdm=indent et:
