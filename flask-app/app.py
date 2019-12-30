from flask import Flask, request, render_template, jsonify

app = Flask(__name__)

# important!!! it is a decorator!
@app.route('/', methods=['GET'])
def index():
	return '<h1>hello world...again<h1>'


if __name__ == '__main__':
	app.run()