from main import extract_text_pymupdf,match_resume
from flask import Flask,request,redirect,render_template
from werkzeug.utils import secure_filename
import os

app=Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads/'



@app.route('/')
def home():
    return render_template('/home.html')



@app.route('/upload', methods=["GET", "POST"])
def upload():
    if request.method == "POST":
        if 'resume' not in request.files:
            return "No file uploaded", 400
        
        file = request.files['resume']
        
        if file.filename == '':
            return "No selected file", 400
        
        if file and file.filename.endswith(('.pdf', '.doc', '.docx')):
            filename = secure_filename(file.filename)
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(file_path)
            text = extract_text_pymupdf(file_path)
            data = match_resume(text)
            result=True if data>0.4 else False
            if result:
                return render_template('feedback.html',data=result)
            else:
                return render_template('upload.html',data="no")
        else:
            return "Invalid file format. Only PDF and DOCX allowed.", 400

    return render_template('upload.html')

@app.route('/about')
def about():
    return render_template('/about.html')


@app.route('/feedback')
def feedback():
    return render_template('/feedback.html')


if __name__=="__main__":
    app.run(debug=True,port=4000)