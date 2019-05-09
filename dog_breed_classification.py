
from CNNmodels import dog_detector as model
from flask import Flask, url_for, request, render_template, flash, redirect
from werkzeug import secure_filename
import os

# python dog_breed_classification.py
# Running on http://127.0.0.1:5000/

IMG_FOLDER = 'static/img'

m = model.DogDetector()

app = Flask(__name__)
app.config['IMG_FOLDER'] = IMG_FOLDER

def _get_img_path(image):
    return os.path.join(app.config['IMG_FOLDER'], image)

def _get_ref_img_path(ref):
    if len(ref.split()) > 1:
        ref = ref.replace(' ', '_')

    return os.path.join('img/ref', ref + '.jpg')

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        if 'image' not in request.files:
            flash('No file part')
            return redirect(request.url)
            
        file = request.files['image']
        if file.filename == '':
            flash('No selected file')
            return redirect(request.url)
        # save file in temp folder
        # redirect to result page
        filename = secure_filename(file.filename)
        file.save(_get_img_path(filename))
        return redirect(url_for('result', image=filename))
        
    return render_template('index.html')
    
@app.route('/delete/<image>') 
def delete(image):
    # call to delete selected 
    if request.method == 'POST':
        to_delete = _get_img_path(image)
    
@app.route('/result/<image>')
def result(image):
    # check if file exists in img folder
    # process file
    # render result template
    code, predictions = m.image_prediction(_get_img_path(image))

    if code == 1:
        info = 'human_face'
    elif code == 2:
        info = 'neither'
    elif code == 3:
        info = 'dog'

    return render_template('result.html', info=info, predictions=predictions, 
    img_file = os.path.join('img', image))

if __name__ == '__main__':
    app.run()
