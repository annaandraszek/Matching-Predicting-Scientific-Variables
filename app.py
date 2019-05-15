from flask import Flask, request, redirect, url_for, render_template
app = Flask(__name__)
from main import app_user_input

@app.route('/result/<variable>')
def result(variable):
    return 'Predicting %s' % variable

@app.route('/predict', methods=['POST', 'GET'])
def predict():
    if request.method == 'POST':
        inpt = request.form['var']
        #return render_template('predict.html')
        #return redirect(url_for('result', variable=inpt))
        result = str(app_user_input(s=str(inpt)))
        return render_template('predict.html', prediction=result)
    else:
        #inpt = request.args.get('var')
        return render_template('predict.html')
        #return redirect(url_for('result', variable=inpt))

@app.route('/')
def home():
    return 'Hi! Go to /predict to enter a scientific variable'


if __name__ == '__main__':
    app.run(debug=True)