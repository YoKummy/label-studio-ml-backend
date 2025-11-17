from flask import Flask, request, jsonify
import paramiko
import subprocess
app = Flask(__name__)

ALLOWED_COMMANDS = {
    "nvidia-smi": "nvidia-smi",
    "l": "ls",
    "whoami": "whoami",
    "ipconfig": "ipconfig",
    "tree": "tree"
    # you can add your training commands here
    # "train": "python3 /path/to/train.py --batch 16 --data dataset.yaml"
}

@app.route('/run/<command_name>', methods=['GET'])
def run_command(command_name):
    command = ALLOWED_COMMANDS.get(command_name)
    if not command:
        return jsonify({"output": "", "error": f"Command '{command_name}' not allowed."}), 403

    try:
        result = subprocess.run(command, shell=True, capture_output=True, text=True)
        output_lines = result.stdout.splitlines()
        error_lines = result.stderr.splitlines()
        return jsonify({"output": output_lines, "error": error_lines})

    except Exception as e:
        return jsonify({"output": "", "error": str(e)})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=4090)
