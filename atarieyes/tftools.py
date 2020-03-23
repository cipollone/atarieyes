"""TensorFlow, TensorForce and training utilities."""

import os
import shutil
import numpy as np
import tensorflow as tf


def prepare_directories(what, env_name, resuming=False):
    """Prepare the directories where weights and logs are saved.

    :param what: what is trained, usually 'agent' or 'features'.
    :param env_name: the actual paths are a composition of
        'what' and 'env_name'.
    :param resuming: if True, the directories are not deleted.
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

    return (models_path, log_path)


class TensorBoardLogger:
    """Visualize data on TensorBoard."""

    def __init__(self, model, path):
        """Initialize.

        :param model: tensorflow model that should be saved.
        :param path: directory where logs should be saved.
        """

        self.path = path
        self.model = model
        self.summary_writer = tf.summary.create_file_writer(path)

    def save_graph(self, input_shape):
        """Save the graph of the model.

        :param input_shape: the shape of the input tensor of the model
            (without batch).
        """

        # Forward pass
        @tf.function
        def tracing_model_ops(inputs):
            return self.model(inputs)

        inputs = np.zeros((1, *input_shape), dtype=np.uint8)

        # Now trace
        tf.summary.trace_on(graph=True)
        tracing_model_ops(inputs)
        with self.summary_writer.as_default():
            tf.summary.trace_export(self.model.name, step=0)

    def save_scalars(self, step, metrics):
        """Save scalars.

        :param step: the step number
        :param metrics: a dict of (name: value)
        """

        with self.summary_writer.as_default():
            for name, value in metrics.items():
                tf.summary.scalar(name, value, step=step)

    def save_images(self, step, images):
        """Save images in TensorBoard.

        :param step: the step number
        :param images: a dict of batches of images. Only the first
            of each group is displayed.
        """

        with self.summary_writer.as_default():
            for name, batch in images.items():
                image = batch[0]
                image = tf.expand_dims(image, axis=0)
                tf.summary.image(name, image, step)


class CheckpointSaver:
    """Save weights and restore."""

    def __init__(self, model, path):
        """Initialize.

        :param model: the Keras model that will be saves.
        :param path: directory where checkpoints should be saved.
        """

        self.save_path = os.path.join(path, model.name)
        self.counters_path = os.path.join(path, "counters.txt")
        self.model = model
        self.score = float("-inf")

    def save(self, step, score=None):
        """Save.

        :param step: current training step.
        :param score: if provided, only save if score is higher than last
            saved score. If None, always save.
        :return: True when the model is saved.
        """

        # Return if not best
        if score is not None:
            if score < self.score:
                return False
            else:
                self.score = score

        # Save
        self.model.save_weights(
            self.save_path, overwrite=True, save_format="tf")
        with open(self.counters_path, "w") as f:
            f.write("step: " + str(step))
        return True

    def load(self, input_shape):
        """Restore weights from the previous checkpoint.

        NOTE: optimizer state is not restored.

        :param input_shape: shape of the input tensors (without batch).
            This is used to initialize the optimizer before restoring.
        :return: the step of the saved weights
        """

        # Restore
        self.model.load_weights(self.save_path)

        # Step
        with open(self.counters_path) as f:
            counters = f.read()
        step = int(counters.split(":")[1])

        return step
