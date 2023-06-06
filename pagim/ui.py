"""Gradio User interface"""

import os
import shutil
from typing import Iterable

import cv2
import gradio as gr
import uvicorn
from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles

from .logger import add_handlers
from .utils import exception_hook
from .filesys import project_root, walkdir_with_extension
from .imgfunc import opdict


exception_hook("debug")
_ = add_handlers([
    {"name": "console", "level": "DEBUG"}
])  # initialize root logger


IMAGEDIR = os.path.join(project_root(), "images")
IMAGEEXT = [".jpg", ".jpeg", ".jpe", ".png", ".bmp", ".dlib", ".webp", ".pbm",
            ".pgm", ".ppm", ".pxm", ".pnm", ".tiff", ".tif",]

# Global states
state: dict = dict(filename=None, orig=None, prev=None, this=None, history=None)


def listdir() -> Iterable[str]:
    """Read the image dir and yield out all image files"""
    for filename in walkdir_with_extension(IMAGEDIR, IMAGEEXT, 0):
        yield os.path.basename(filename)


def getfilename(filename):
    """Return a usable filename based on the tentative filename"""
    base, ext = os.path.splitext(filename)
    counter = None
    while True:
        if counter is None:
            testfilename = base + ext
        else:
            testfilename = f"{base}.{counter}{ext}"
        path = os.path.join(IMAGEDIR, testfilename)
        if not os.path.isfile(path):
            return testfilename
        else:
            counter = 1 if counter is None else counter+1


def load_image(filename: str) -> None:
    """Read the image file from disk and reset states"""
    img = cv2.imread(os.path.join(IMAGEDIR, filename), cv2.IMREAD_COLOR)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    state.update({
        "filename": filename,
        "orig": img,
        "prev": None,
        "this": None,
        "history": ["img = cv2.imread(%s, cv2.COLOR_BGR2RGB)" % repr(state["filename"]),
                    "img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)"],
    })


def gr_fileupload(files: list):
    """Move all uploaded file to local dir"""
    for file in files:
        localpath = file.name
        realfilename = getfilename(os.path.basename(localpath))
        targetpath = os.path.join(IMAGEDIR, realfilename)
        shutil.move(localpath, targetpath)
    return gr.Dropdown.update(choices=list(listdir()))


def gr_select(filename: str):
    """On select of filename, reset the screen"""
    load_image(filename)
    return state["orig"], None, None, ""


def gr_restart():
    """On restart button, reset all history"""
    if state["orig"] is None:
        return
    state.update({
        "prev": None,
        "this": None,
        "history": ["img = cv2.imread(%s, cv2.COLOR_BGR2RGB)" % repr(state["filename"]),
                    "img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)"]
    })
    return None, None, "\n".join(state["history"])


def gr_action(operation):
    """On action button, update the image"""
    if state["orig"] is None or operation not in opdict:
        return
    if state["this"] is None:
        state["prev"] = state["orig"]
    else:
        state["prev"] = state["this"]
    function = opdict[operation]
    newimg, command = function(state["prev"])
    if isinstance(command, str):
        state["history"].append(command)
    else:
        assert isinstance(command, list)
        state["history"].extend(command)
    state["this"] = newimg
    return state["prev"], state["this"], "\n".join(state["history"])



def create_gradio(projectname="Test out image operations"):
    """Create gradio app to be hooked to FastAPI"""
    with gr.Blocks(analytics_enabled=False, title=projectname, css=None) as demo:
        # inputs
        gr.HTML(r'<div><h1 style="position:relative;"><img src="static/logo.png" style="float:right;" />%s</h1></div>' % projectname)
        with gr.Row():
            imagefiles = list(listdir())
            filenames = gr.Dropdown(choices=imagefiles, label="Images from disk")
            ops = gr.Dropdown(choices=list(opdict.keys()), label="Operations")
            with gr.Column():
                restart = gr.Checkbox(label="Restart")
                run = gr.Button("Action")
        # output
        with gr.Row():
            img1 = gr.Image(label="Original", show_label=True)
            img2 = gr.Image(label="Previous", show_label=True)
            img3 = gr.Image(label="Current", show_label=True)
        oplog = gr.Textbox(label="History of operations", interactive=False)
        fileupload = gr.Files(file_types=IMAGEEXT,
                              label="Upload new images")
        # action hook
        fileupload.upload(gr_fileupload,
                          inputs=[fileupload],
                          outputs=[filenames],
                          api_name="upload_file")
        filenames.input(gr_select,
                        inputs=[filenames],
                        outputs=[img1, img2, img3, oplog],
                        api_name="select_file")
        restart.input(gr_restart,
                      inputs=[],
                      outputs=[img2, img3, oplog],
                      api_name="restart")
        run.click(gr_action,
                  inputs=[ops],
                  outputs=[img2, img3, oplog],
                  api_name="apply")
    return demo


app = FastAPI()

# for serving Synechron logo
static_dir = "static"
os.makedirs(static_dir, exist_ok=True)
app.mount("/static", StaticFiles(directory=static_dir), name="static")

# let Gradio hook up itself to FastAPI
app = gr.mount_gradio_app(app, create_gradio(), path="/")

# serve the app
if __name__ == "__main__":
    # create_gradio().launch() -- only if no FastAPI needed, e.g., no static file
    # uvicorn.run(app, host="0.0.0.0", port=7860)  -- cannot do reload=True
    uvicorn.run("pagim.ui:app", host="0.0.0.0", port=7860, reload=True)


# ----
#  create actions without user interaction
#    1. gr.RegisterEvent(fn=function, event="my_event_name", inputs=[x], outputs=[y])
#    2. gr.TriggerEvent("my_event_name")
