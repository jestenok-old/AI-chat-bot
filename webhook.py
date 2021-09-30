from flask import Flask, request
from chat import get_response


app = Flask(__name__)


@app.route('/aichatbot', methods=['GET'])
def chat():
    return get_response(request.args.get('text'))


if __name__ == '__main__':
    print("webhook start")
    app.run('jestenok.ru', port=3210)
