# from flask import Flask, render_template, request, redirect, url_for
# import os

# app = Flask(__name__)
# UPLOAD_FOLDER = 'uploads'
# app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# if not os.path.exists(UPLOAD_FOLDER):
#     os.makedirs(UPLOAD_FOLDER)

# @app.route('/', methods=['GET', 'POST'])
# def home():
#     if request.method == 'POST':
#         files = request.files.getlist('resume')
#         for file in files:
#             if file.filename != '':
#                 file.save(os.path.join(app.config['UPLOAD_FOLDER'], file.filename))
#         return redirect(url_for('home'))
#     return render_template('index.html')

# if __name__ == '__main__':
#     app.run(debug=True)


