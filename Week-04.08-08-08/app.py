from flask import Flask, render_template, request, jsonify
import cv2
import pytesseract
import re
import os

import platform
if platform.system() == 'Windows':
    pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'
elif platform.system() == 'Linux':
    pytesseract.pytesseract.tesseract_cmd = 'tesseract'

app = Flask(__name__, template_folder="./templates")

UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}

if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

    
def process_image(image_path):
    image = cv2.imread(image_path)
    if image is None:
        return None
    

    # Bölgeler: (x1, y1, x2, y2)
    regions = {
        'tc_no': (0.057, 0.245, 0.35, 0.36),
        #'name': (0.296, 0.54, 0.56, 0.61),
        #'surname': (0.296, 0.42, 0.56, 0.5),
        #'birth_date': (0.296, 0.65, 0.56, 0.725),
        'others': (0.296, 0.35, 0.56, 0.725)
    }

    h, w = image.shape[:2]

    # Ön işleme
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.medianBlur(gray, 3)

    results = {}
    for label, (x1_r, y1_r, x2_r, y2_r) in regions.items():
        x1 = int(x1_r*w)
        y1 = int(y1_r*h)
        x2 = int(x2_r*w)
        y2 = int(y2_r*h)

        region_img = blurred[y1:y2, x1:x2]

        # Tesseracta bölgenin resmini gönder
        data = pytesseract.image_to_data(region_img, lang='tur', output_type=pytesseract.Output.DICT)

        if label == 'tc_no':
            match = None
            for i, word in enumerate(data['text']):
                # 11 rakamdan oluşan sayıları ara
                if re.fullmatch(r'\d{11}', word):
                    match = word
                    (x, y, w_box, h_box) = (data['left'][i], data['top'][i], data['width'][i], data['height'][i])
                    #cv2.rectangle(image, (x1 + x, y1 + y), (x1 + x + w_box, y1 + y + h_box), (0, 255, 0), 2)
                    break
            results['TC_NO'] = match if match else "Bulunamadı"

        elif label == 'others':
            uppercase_indices = [i for i, w in enumerate(data['text']) if w.isupper()]
            if uppercase_indices:
                surname = data['text'][uppercase_indices[0]]
                name = ' '.join([data['text'][i] for i in uppercase_indices[1:]]) if len(uppercase_indices) > 1 else ''
                results['SURNAME'] = surname
                results['NAME'] = name

                for i in uppercase_indices:
                    (x, y, w_box, h_box) = (data['left'][i], data['top'][i], data['width'][i], data['height'][i])
                    cv2.rectangle(image, (x1 + x, y1 + y), (x1 + x + w_box, y1 + y + h_box), (0, 255, 0), 2)
            else:
                results['SURNAME'] = "Bulunamadı"
                results['NAME'] = "Bulunamadı"

            dob_index = None
            for i, word in enumerate(data['text']):
                if re.fullmatch(r'\d{2}[./]\d{2}[./]\d{4}', word):
                    dob_index = i
                    (x, y, w_box, h_box) = (data['left'][i], data['top'][i], data['width'][i], data['height'][i])
                    cv2.rectangle(image, (x1 + x, y1 + y), (x1 + x + w_box, y1 + y + h_box), (0, 255, 0), 2)
                    break
            results['BIRTHDATE'] = data['text'][dob_index] if dob_index is not None else "Bulunamadı"

    return results

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return jsonify({'error': 'Dosya seçilmedi'})
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'Dosya seçilmedi'})
    
    if file and allowed_file(file.filename):
        filename = file.filename
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        
        try:
            info = process_image(filepath)
            
            if info is None:
                return jsonify({
                    'success': False,
                    'message': 'Görsel işlenemedi'
                })

            try:
                os.remove(filepath)
            except:
                pass

            return jsonify({
                'success': True,
                'info': {
                    'tc_no': info.get('TC_NO'),
                    'name': info.get('NAME'),
                    'surname': info.get('SURNAME'),
                    'birthdate': info.get('BIRTHDATE')
                }
            })
        except Exception as e:
            try:
                os.remove(filepath)
            except:
                pass
            
            return jsonify({
                'success': False,
                'message': f'İşleme hatası: {str(e)}'
            })
    
    return jsonify({'error': 'Geçersiz dosya formatı'})

if __name__ == "__main__":
    app.run(debug=True, host='0.0.0.0', port=5000)

