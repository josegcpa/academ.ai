import os
from flask import Flask, render_template, Response

app = Flask(__name__, template_folder="static")

SERVE_PORT = int(os.environ.get("SERVE_PORT", 1234))
API_HOST = os.environ.get("API_HOST", "localhost")
API_PORT = int(os.environ.get("API_PORT", 8000))


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/static/script.js")
def serve_script():
    js_content = render_template(
        "script.js", API_HOST=API_HOST, API_PORT=API_PORT
    )
    return Response(js_content, mimetype="application/javascript")


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("academ_ai.app:app", host="0.0.0.0", port=SERVE_PORT)
