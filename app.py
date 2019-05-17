from flask import Flask, request, redirect, url_for, render_template
app = Flask(__name__)
from main import app_user_input, app_load_models

app_load_models()


@app.route('/result/<variable>')
def result(variable):
    return 'Predicting %s' % variable

@app.route('/')
@app.route('/predict', methods=['POST', 'GET'])
def predict():
    if request.method == 'POST':
        inpt = request.form['var']
        #return render_template('indexindex.html')
        #return redirect(url_for('result', variable=inpt))
        result = app_user_input(str(inpt))
        print(result, len(result))
        if isinstance(result, str):
            return render_template('index.html', no_prediction=result)
        if len(result) <= 1:
            result = [result]
        return render_template('index.html', prediction=result)
    else:
        #inpt = request.args.get('var')
        return render_template('index.html')
        #return redirect(url_for('result', variable=inpt))

#def home():
#    return 'Hi! Go to /predict to enter a scientific variable'


if __name__ == '__main__':
    app.run(debug=True)
