"""Training utilities."""

import os
import shutil
import json
import numpy as np
import tensorflow as tf


def prepare_directories(what, env_name, resuming=False, args=None):
    """Prepare the directories where weights and logs are saved.

    :param what: what is trained, usually 'agent' or 'features'.
    :param env_name: the actual paths are a composition of
        'what' and 'env_name'.
    :param resuming: if True, the directories are not deleted.
    :param args: argsparse.Namespace of arguments. If given, this is saved
        to 'args' file inside the log directory.
    :return: two paths, respectively for models and logs.
    """

    # Choose diretories
    models_path = os.path.join("models", what, env_name)
    logs_path = os.path.join("logs", what, env_name)
    dirs = (models_path, logs_path)

    # Delete old ones
    if not resuming:
        if any(os.path.exists(d) for d in dirs):
            print(
                "Old logs and models will be deleted. Continue (Y/n)? ",
                end="")
            c = input()
            if c not in ("y", "Y", ""):
                quit()

        # New
        for d in dirs:
            if os.path.exists(d):
                shutil.rmtree(d)
            os.makedirs(d)

    # Logs alwas use new directories (using increasing numbers)
    i = 0
    while os.path.exists(os.path.join(logs_path, str(i))):
        i += 1
    log_path = os.path.join(logs_path, str(i))
    os.mkdir(log_path)

    # Save arguments
    if args is not None:
        ArgumentSaver.save(os.path.join(log_path, "args.json"), args)

    return (models_path, log_path)


class ArgumentSaver:
    """Saves and loads command line arguments."""

    @staticmethod
    def save(path, args):
        """Saves the arguments to path.

        :param path: destination file path
        :param args: argparse.Namespace
        """

        with open(path, "w") as f:
            json.dump(vars(args), f, indent=4)

    @staticmethod
    def load(path, args):
        """Loads arguments from file.

        :param path: source file path
        :param args: destination argparse.Namespace
        """

        with open(path) as f:
            data = json.load(f)
        args.__dict__.update(data)
