"""
Flask app
"""
import os
from flask import Flask, flash, request, redirect, render_template
from werkzeug.utils import secure_filename
import numpy as np
from numpy.linalg import det
import pandas as pd
from flask import send_from_directory
from interfacemaster.interface_generator import core, get_disorientation
from interfacemaster.symmetric_tilt import get_csl_twisted_graphenes
import plotly.express as px
import io

# 画像のアップロード先のディレクトリ
UPLOAD_FOLDER = './uploads'
# アップロードされる拡張子の制限
ALLOWED_EXTENSIONS = set(['cif', ])

app = Flask(__name__)
app.secret_key = 'hogehoge'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER


@app.route("/")
def main():
    return render_template("index.html")


@app.route("/plotly")
def plotly():
    with io.StringIO() as io_object:
        x = np.linspace(-2 * np.pi, 2 * np.pi, 256)
        fig = px.scatter(
            x=x,
            y=np.sin(x),
            title="Sine carve"
        )
        fig.write_html(io_object, full_html=False)
        return render_template("result.html", result=io_object.getvalue())


@app.route("/twisted_graphene", methods=['GET', 'POST'])
def twisted_graphene():
    if request.method == 'POST':
        if 'file' not in request.files:
            print('No file')
            file = None
        file = request.files['file']
        if file.filename == '':
            print('No file')
            file = None
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
        if file is None:
            filename = os.path.join(
                '../test_files/cif_files',
                'C_mp-990448_conventional_standard.cif')
        else:
            filename = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        sigmas, thetas, A_cnid, anum = get_csl_twisted_graphenes(
            lim=10,
            filename=filename,
            maxsigma=int(request.form.get('maxsigma')),
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
            result = io_object.getvalue()
        return render_template('result.html', result=result + df.to_html())
    return render_template("twisted_graphene.html")


@app.route("/sample_stgb", methods=['GET', 'POST'])
def sample_stgb():
    if request.method == 'POST':
        try:
            if 'file' not in request.files:
                flash('No file')
                return redirect(request.url)
            file = request.files['file']
            if file.filename == '':
                flash('No file')
                return redirect(request.url)
            if file and allowed_file(file.filename):
                filename = secure_filename(file.filename)
                file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
            filename = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            my_interface = core(
                file_1=filename, file_2=filename
            )
            # get the rotation mtx
            v1 = list(map(int, request.form.get('v1').split(",")))
            hkl1 = list(map(int, request.form.get('hkl1').split(",")))
            v2 = list(map(int, request.form.get('v2').split(",")))
            hkl2 = list(map(int, request.form.get('hkl2').split(",")))
            R = get_disorientation(
                L1=my_interface.conv_lattice_1,
                L2=my_interface.conv_lattice_2,
                v1=v1, hkl1=hkl1,
                v2=v2, hkl2=hkl2
            )
            my_interface.parse_limit(
                du=float(request.form.get('du')),
                S=float(request.form.get('S')),
                sgm1=int(request.form.get('maxsigma1')),
                sgm2=int(request.form.get('maxsigma2')),
                dd=float(request.form.get('dd')))
            my_interface.search_fixed(R)
            sigma1 = int(abs(np.round(det(my_interface.U1))))
            sigma2 = int(abs(np.round(det(my_interface.U2))))
            result = (
                f"CSL:<br>{'<br>'.join([f'{v}' for v in my_interface.CSL])}"
                f"<br>U1: {my_interface.U1}, Sigma1: {sigma1}"
                f"<br>U2: {my_interface.U2}, Sigma1: {sigma2}"
                f"<br>D: {my_interface.D}, det(D): {det(my_interface.D)}"
            )
            return render_template('result.html', result=result)
        except Exception as ex:
            return render_template('result.html', result=ex)
    return render_template("sample_stgb.html")


def allowed_file(filename):
    # .があるかどうかのチェックと、拡張子の確認
    # OKなら１、だめなら0
    return ('.' in filename
            and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS)


@app.route('/uploads/<filename>')
# ファイルを表示する
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)
