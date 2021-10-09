import os

from flask import Flask, request, render_template, send_file, redirect, session
from werkzeug.utils import secure_filename
from dataframe_processor import process_df

import pandas as pd

#from secret_codes import SECRET_KEY, ACCESS_CODE

INPUT_FOLDER = os.environ.get("CLF_INPUT_FOLDER", default="input_files")
OUTPUT_FOLDER = os.environ.get("CLF_OUTPUT_FOLDER", default="output_files")
LOGIN_REQUIRED = int(os.environ.get("CLF_LOGIN_REQUIRED", default=False))
INDICATE_HANDLERS = int(os.environ.get("CLF_INDICATE_HANDLERS", default=True))
SECRET_KEY = os.environ.get("CLF_SECRET_KEY", default="secretkey")
ACCESS_CODE = os.environ.get("CLF_ACCESS_CODE", default="accesscode")

for folder in (INPUT_FOLDER, OUTPUT_FOLDER):
    if not os.path.exists(folder):
        os.mkdir(folder)

app = Flask(__name__, template_folder="templates")
app.config["UPLOAD_FOLDER"] = INPUT_FOLDER
app.secret_key = SECRET_KEY

## Access code check
@app.route('/login', methods=["GET","POST"])
def check_access():
    if request.method == 'POST':
        if request.form.get("access_code") == ACCESS_CODE:
            session["access_code"] = request.form["access_code"]
            return redirect('/')
    return render_template("access.html")


## GUI for Excel I/O
@app.route('/', methods=["GET","POST"])
def index():
    if LOGIN_REQUIRED:
        if session.get("access_code") != ACCESS_CODE:
            return redirect('/login')
    if request.method == 'POST':
        if 'dataFile' in request.files:
            file = request.files['dataFile']
            if file:
                filename, extension = file.filename[:file.filename.rfind('.')], file.filename[file.filename.rfind('.'):].lower()
                if extension in ('.xlsx','.csv'):
                    inp_filepath = os.path.join(INPUT_FOLDER, secure_filename(file.filename))
                    out_filepath = os.path.join(OUTPUT_FOLDER, secure_filename(file.filename))
                    file.save(inp_filepath)

                    if extension == '.xlsx':
                        df = pd.read_excel(inp_filepath, engine="openpyxl")
                        df = process_df(df, INDICATE_HANDLERS)
                        df.to_excel(out_filepath, engine="openpyxl")
                    elif extension == '.csv':
                        df = pd.read_csv(inp_filepath)
                        df = process_df(df, INDICATE_HANDLERS)
                        df.to_csv(out_filepath)

                    attachment_name = filename+'_processed'+extension
                    return send_file(out_filepath, as_attachment=True,attachment_filename=attachment_name)
                
    return render_template('index.html')
