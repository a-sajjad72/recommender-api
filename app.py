from flask import Flask
from routes import api

app = Flask(__name__)
app.json.sort_keys = False
app.register_blueprint(api)

if __name__ == "__main__":
    app.run(host="localhost", port=3000, debug=True)
