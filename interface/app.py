from flask import Flask, render_template, request, jsonify
from tools import classify

app = Flask(__name__)

@app.route('/')
def home_route():
    return render_template('index.html')

@app.route('/recognizer')
def recognize_route():
    message = request.args.get('message')
    return jsonify(classify(message))

if __name__ == '__main__':
    app.run(debug=True)
