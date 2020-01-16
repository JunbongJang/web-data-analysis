# Author: Junbong Jang
# Date: 11/30/2018
# app/views.py

from flask import render_template, flash, request, redirect, url_for
from werkzeug.utils import secure_filename
import os

from analysis.data_preprocessor import Data_Preprocessor
from app import app, google_spreadsheet
from analysis import multiple_regression, apa_formater, csv_parser


def allowed_file(filename):
    ALLOWED_EXTENSIONS = set(['txt', 'csv'])
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


# http://flask.pocoo.org/docs/1.0/patterns/fileuploads/
@app.route('/', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        print('Post')
        # check if the post request has the file part
        if 'file' not in request.files:
            flash('No file part')
            return redirect(request.url)
        file = request.files['file']
        # if user does not select file, browser also
        # submit an empty part without filename
        if file.filename == '':
            flash('No selected file')
            return redirect(request.url)
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
            return redirect(url_for('display_upload',
                                    filename=filename))  # build a URL to a specific function
    elif request.method == 'GET':
        print('Get')
    return render_template("index.html", view_state='home')


@app.route('/display/<filename>', methods=['GET', 'POST'])
def display_upload(filename):
    spreadsheet_instance = google_spreadsheet.Google_Spreadsheet()
    service = spreadsheet_instance.user_authentication()
    spreadsheet_url = spreadsheet_instance.create_spreadsheet(service, filename)

    list_of_values = csv_parser.read_uploaded_csv(filename)
    spreadsheet_instance.update_spreadsheet(service, list_of_values)
    return render_template('upload_display.html',
                           view_state='data',
                           spreadsheet_url=spreadsheet_url,
                           filename=filename,
                           columns=list_of_values[0])


@app.route('/result/<filename>', methods=['GET', 'POST'])
def result(filename):
    print('result ~~~~~~')
    # https://stackoverflow.com/questions/25065900/request-args-getkey-gives-null-flask
    parsed_independent_var = request.form.get('independent_var').split(", ")
    parsed_dependent_var = [request.form.get('dependent_var')]
    print(parsed_independent_var)
    print(parsed_dependent_var)
    print(filename)

    data_preprocessor = Data_Preprocessor(filename = "app/uploads/"+filename,
                                          missing_data='drop',
                                          x_columns=parsed_independent_var,
                                          y_columns=parsed_dependent_var,
                                          categorical_columns=['a01'])

    regression_model = multiple_regression.Multiple_Regression(data_preprocessor)
    model_stat_dict, coefficients_dict = regression_model.calc_multiple_regression_stat()
    print(model_stat_dict)
    print(coefficients_dict)
    regression_model.draw_correlation_matrix(filename)
    with app.app_context():
        return render_template('result.html',
                               view_state='result',
                               filename=filename,
                               descriptive_stat=regression_model.calc_descriptive_dict(),
                               corr_2d=apa_formater.corr_matrix_apa(regression_model),
                               model_stat_dict=model_stat_dict,
                               coefficients_dict=coefficients_dict)
