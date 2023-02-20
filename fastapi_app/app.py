import os
import io
import shutil

from fastapi import FastAPI, UploadFile, Form
from fastapi.templating import Jinja2Templates
from fastapi.responses import HTMLResponse
from fastapi.requests import Request

from interfacemaster.interface_generator import core, get_disorientation
from interfacemaster.symmetric_tilt import get_csl_twisted_graphenes

import numpy as np
from numpy.linalg import det
import pandas as pd
import plotly.express as px

UPLOAD_DIR = './uploads'
ALLOWED_EXTENSIONS = set(['cif', ])
CIF_DIR = '../test_files/cif_files'
TEMPLATE_DIR = 'templates'
templates = Jinja2Templates(directory=TEMPLATE_DIR)

app = FastAPI()


@app.get("/")
def read_root():
    return {"Hello": "World"}


@app.get("/items/{item_id}")
def read_item(item_id):
    """パスパラメータ"""
    return {"item_id": item_id}


fake_items_db = [{"item_name": "Foo"}, {
    "item_name": "Bar"}, {"item_name": "Baz"}]


@app.get("/items/")
def read_query_item(skip: int = 0, limit: int = 10):
    """クエリパラメータ"""
    return fake_items_db[skip: skip + limit]


@app.get("/twisted_graphene", response_class=HTMLResponse)
def get_product(request: Request):
    return templates.TemplateResponse(
        "twisted_graphene.html",
        {
            "request": request,
        }
    )


def get_upload_file(
    upload_file: UploadFile | None = None,
    default_file: str | None = None,
    verbose: bool = True
):
    if upload_file.filename == "":
        if default_file is None:
            raise RuntimeError(
                "No file uploaded and no default file specified!")
        filename = os.path.join(
            CIF_DIR,
            default_file)
        if verbose:
            print(f"use default file {filename}")
    else:
        if not upload_file.filename.endswith(".cif"):
            raise RuntimeError("the file is not CIF!")
        filename = os.path.join(
            UPLOAD_DIR,
            upload_file.filename
        )
        with open(filename, 'w+b') as buffer:
            shutil.copyfileobj(upload_file.file, buffer)
        if verbose:
            print(f"use upload file {filename}")
    return filename


@app.post("/twisted_graphene", response_class=HTMLResponse)
async def show_product(
    upload_file: UploadFile | None = None,
    lim: str = Form(...),
    maxsigma: str = Form(...),
    request: Request = {}
):
    filename = get_upload_file(
        upload_file=upload_file,
        default_file='C_mp-990448_conventional_standard.cif'
    )
    sigmas, thetas, A_cnid, anum = get_csl_twisted_graphenes(
        lim=int(lim),
        filename=filename,
        maxsigma=int(maxsigma),
        verbose=False
    )
    df = pd.DataFrame(np.column_stack(
        (sigmas, thetas / np.pi * 180, A_cnid, anum)),
        index=np.arange(len(sigmas)) + 1,
        columns=['sigma', 'thetas', 'CNID area', 'atom number'])
    df['sigma'] = df['sigma'].astype('int')
    df['atom number'] = df['atom number'].astype('int')
    with io.StringIO() as io_object:
        fig = px.scatter(
            x=df["thetas"],
            y=df["sigma"],
            title="twisted angle vs Sigma value"
        )
        fig.write_html(io_object, full_html=False)
        result = f"""
        {io_object.getvalue()}
        "
        {df.to_html().replace(
            'class="dataframe"',
            'class="table table-hover table-striped table-bordered"'
            )}
        """
    print("finished")
    return templates.TemplateResponse(
        "result.html",
        {
            "result": result,
            "request": request
        }
    )


def show_matrix(m, format=".0f"):
    return "\\begin{bmatrix}" \
        + "\\\\".join(['&'.join([f'{value:{format}}' for value in row]) for row in m]) \
        + "\\end{bmatrix}"



@app.post("/sample_stgb", response_class=HTMLResponse)
async def post_sample_stgb(
    upload_file_1: UploadFile | None = None,
    upload_file_2: UploadFile | None = None,
    v1: str = Form(...),
    hkl1: str = Form(...),
    v2: str = Form(...),
    hkl2: str = Form(...),
    du: str = Form(...),
    S: str = Form(...),
    maxsigma1: str = Form(...),
    maxsigma2: str = Form(...),
    dd: str = Form(...),
    request: Request = {}
):
    filename_1 = get_upload_file(
        upload_file=upload_file_1,
        default_file='C_mp-990448_conventional_standard.cif'
    )
    if upload_file_2.filename == "":
        filename_2 = filename_1
    else:
        filename_2 = get_upload_file(
            upload_file=upload_file_2,
        )

    try:
        my_interface = core(
            file_1=filename_1, file_2=filename_2
        )
        # get the rotation mtx
        v1 = list(map(int, v1.split(",")))
        hkl1 = list(map(int, hkl1.split(",")))
        v2 = list(map(int, v2.split(",")))
        hkl2 = list(map(int, hkl2.split(",")))
        R = get_disorientation(
            L1=my_interface.conv_lattice_1,
            L2=my_interface.conv_lattice_2,
            v1=v1, hkl1=hkl1,
            v2=v2, hkl2=hkl2
        )
        my_interface.parse_limit(
            du=float(du),
            S=float(S),
            sgm1=int(maxsigma1),
            sgm2=int(maxsigma2),
            dd=float(dd)
        )
        my_interface.search_fixed(R)
        sigma1 = int(abs(np.round(det(my_interface.U1))))
        sigma2 = int(abs(np.round(det(my_interface.U2))))
        volume = int(abs(np.round(det(my_interface.CSL))))
        result = (
            f"\\(volume = {volume}\\)"
            f"\\(CSL = {show_matrix(my_interface.CSL, format='.4g')}\\)"
            f"\\(\\Sigma_1 = {sigma1}, "
            f"U_1 = {show_matrix(my_interface.U1)}\\)"
            f"\\(\\Sigma_2 = {sigma2}, "
            f"U_2 = {show_matrix(my_interface.U2)}\\)"
            f"\\(D = {show_matrix(my_interface.D, format='.4e')}\\)"
            f"\\(\\det(D) = {det(my_interface.D)}\\)"
        )
        return templates.TemplateResponse(
            "sample_stgb.html",
            {
                "request": request,
                "result": result
            }
        )
    except Exception as ex:
        return templates.TemplateResponse(
            "sample_stgb.html",
            {
                "request": request,
                "result": ex
            }
        )


@app.get("/sample_stgb", response_class=HTMLResponse)
def get_sample_stgb(request: Request):
    return templates.TemplateResponse(
        "sample_stgb.html",
        {
            "request": request,
        }
    )


@app.get("/show_model", response_class=HTMLResponse)
def get_show_model(request: Request):
    return templates.TemplateResponse(
        "show_model.html",
        {
            "request": request,
        }
    )

@app.post("/show_model", response_class=HTMLResponse)
async def post_show_model(
    upload_file: UploadFile | None = None,
    request: Request = {}
):
    filename = get_upload_file(
        upload_file=upload_file,
        default_file='C_mp-990448_conventional_standard.cif'
    )
    import ase.io
    from ase.io import x3d
    atoms = ase.io.read(filename)
    from ase.build import make_supercell
    multiplier = np.identity(3) * 2
    atoms = make_supercell(atoms, multiplier)
    with io.StringIO() as io_object:
        x3d.write_html(io_object, atoms)
        result = io_object.getvalue()
    print("finished")
    return templates.TemplateResponse(
        "show_model.html",
        {
            "result": result,
            "request": request
        }
    )
