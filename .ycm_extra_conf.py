
import subprocess


def Settings(**kwargs):

    completed = subprocess.run(
        ["poetry", "env", "info", "-p", "-n"], capture_output=True)
    python_path = completed.stdout.decode().strip() + "/bin/python"

    return {
        'interpreter_path': python_path
    }
