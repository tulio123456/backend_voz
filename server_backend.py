
from flask import Flask, request, jsonify
from flask_cors import CORS
import threading
import tempfile
import os
import time
import whisper

app = Flask(__name__)
CORS(app)

MODEL_NAME = "small"
PORT = int(os.environ.get("PORT", 5000))

model = whisper.load_model(MODEL_NAME)

latest_command = {}
latest_command_lock = threading.Lock()

def async_transcribe_and_store(player_name, audio_path):
    try:
        result = model.transcribe(audio_path, language="pt", fp16=False)
        text = result.get("text", "").strip()
        normalized = text.lower()

        follow_keywords = ["segue", "siga", "vem", "segue-me", "segue me", "vai comigo"]
        stop_keywords = ["pare", "para", "fica", "fica aí", "fica ai", "pára"]

        command = None
        for k in follow_keywords:
            if k in normalized:
                command = "follow"
                break
        for k in stop_keywords:
            if k in normalized:
                command = "stop"
                break

        if command:
            with latest_command_lock:
                latest_command[player_name] = {"cmd": command, "raw": text, "ts": time.time()}
    except Exception as e:
        print("err:", e)
    finally:
        try: os.remove(audio_path)
        except: pass

@app.route("/upload_audio", methods=["POST"])
def upload_audio():
    if 'file' not in request.files or 'player' not in request.form:
        return "missing", 400

    f = request.files['file']
    player = request.form['player']

    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".wav")
    f.save(tmp.name)
    tmp.close()

    t = threading.Thread(target=async_transcribe_and_store, args=(player, tmp.name))
    t.daemon = True
    t.start()

    return jsonify({"status": "ok"}), 200

@app.route("/get_command", methods=["GET"])
def get_command():
    player = request.args.get("player")
    if not player:
        return jsonify({"found": False}), 400

    with latest_command_lock:
        entry = latest_command.pop(player, None)

    if not entry:
        return jsonify({"found": False}), 200

    return jsonify({"found": True, "cmd": entry["cmd"], "raw": entry["raw"], "ts": entry["ts"]}), 200

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=PORT)
