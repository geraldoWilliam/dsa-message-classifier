from flask import Flask, render_template, request, jsonify
from tools import classify

app = Flask(__name__)

@app.route('/')
def home_route():
    return render_template('index.html')

@app.route('/recognizer')
def recognize_route():
    message = request.args.get('message')
    classifier = request.args.get('classifier')
    balanced_data = request.args.get('balanced_data') == 'true'
    selected_features = request.args.get('selected_features') == 'true'
    return jsonify(classify(message, classifier, balanced_data, selected_features))

if __name__ == '__main__':
    app.run(debug=True)
