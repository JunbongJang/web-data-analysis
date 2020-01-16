# Author: Junbong Jang
# Date: 11/30/2018
# app/__init__.py

from flask import Flask

# Initialize the app
app = Flask(__name__,
            static_url_path='',
            static_folder='static',
            template_folder="templates",
            instance_relative_config=True)

# Load the views
from app import views

# Load the config file
app.config.from_object('config')
app.config['UPLOAD_FOLDER'] = 'app/uploads/'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB
app.secret_key = 'super secret key'