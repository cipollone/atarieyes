"""Python utilities."""

from abc import ABCMeta
import signal
import os
import json
import shutil
import argparse


class ABCMeta2(ABCMeta):
    """This metaclass can be used just like ABCMeta.

    It adds the possibility to declare abstract instance attributes.
    These must be assigned to instances inside the __init__ method.
    How to use:

        class C(metaclass=ABCMeta2):
            attr = AbstractAttribute()
            ...
    Note: methods of this class are not inherited by other classes' instances.
    """

    def __init__(Class, classname, supers, classdict):
        """Save abstract attributes."""

        abstract = []
        for attr in dir(Class):
            if isinstance(getattr(Class, attr), AbstractAttribute):
                abstract.append(attr)
        Class.__abstract_attributes = abstract

    def __call__(Class, *args, **kwargs):
        """Intercept instance creation."""

        # Create instance
        instance = ABCMeta.__call__(Class, *args, **kwargs)

        # Check abstract
        not_defined = []
        for attr in Class.__abstract_attributes:
            if attr not in instance.__dict__:
                not_defined.append(attr)
        if not_defined:
            raise TypeError(
                Class.__name__ + ".__init__ did not define these abstract "
                "attributes:\n" + str(not_defined))

        return instance


class AbstractAttribute:
    """Define an abstract attribute. See description in ABCMeta2."""


class ABC2(metaclass=ABCMeta2):
    """Abstract class through inheritance.

    Use this class just like abc.ABC.
    """


class QuitWithResources:
    """Close the resources when ctrl-c is pressed."""

    __deleters = {}
    __initialized = False

    def __init__(self):
        """Don't instantiate."""

        raise TypeError("Don't instantiate this class")

    @staticmethod
    def close():
        """Close all and quit."""

        for name, deleter in QuitWithResources.__deleters.items():
            deleter()
        quit()

    @staticmethod
    def add(name, deleter):
        """Declare a new resource to be closed.

        :param name: any identifier for this resource.
        :param deleter: callable to be used when closing.
        """

        if not QuitWithResources.__initialized:
            signal.signal(
                signal.SIGINT, lambda sig, frame: QuitWithResources.close())
            QuitWithResources.__initialized = True

        if name in QuitWithResources.__deleters:
            raise ValueError("This name is already used")

        QuitWithResources.__deleters[name] = deleter

    @staticmethod
    def remove(name):
        """Removes a resource.

        :param name: identifier of a resource.
        """

        if name not in QuitWithResources.__deleters:
            raise ValueError(str(name) + " is not a resource")

        QuitWithResources.__deleters.pop(name)


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
    def load(path, args=None):
        """Loads arguments from file.

        :param path: source file path
        :param args: destination argparse.Namespace. If None, create one.
        :return: args or the new namespace
        """

        if args is None:
            args = argparse.Namespace()

        with open(path) as f:
            data = json.load(f)
        args.__dict__.update(data)

        return args


class Namespace(argparse.Namespace):
    """Simple Namespace class."""

    def __init__(self, *args, **kwargs):
        """Initialize.

        :param args: only one positional argument:
            namespace: argparse.Namespace to copy (optional)
        :param kwargs: attributes to set
        """

        if args:
            if len(args) > 1 or not isinstance(args[0], argparse.Namespace):
                raise TypeError(
                    "Only one positional argument is allowed: r Namespace.")
            namespace = args[0]
            self.__dict__.update(namespace.__dict__)

        self.__dict__.update(kwargs)

    def __iter__(self):
        """Iterate over (key, value) tuples; we can create a dict with this."""

        for pair in self.__dict__.items():
            assert pair[0] in self
            yield pair


def prepare_directories(
    what, env_name, resuming=False, args=None, no_create=False
):
    """Prepare the directories where weights and logs are saved.

    :param what: what is trained, usually 'agent' or 'features'.
    :param env_name: the actual paths are a composition of
        'what' and 'env_name'.
    :param resuming: if True, the directories are not deleted.
    :param args: argsparse.Namespace of arguments. If given, this is saved
        to 'args' file inside the log directory.
    :param no_create: do not touch the files; just return the current paths
        (implies resuming).
    :return: two paths, respectively for models and logs.
    """

    if no_create:
        resuming = True

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

    # Return current
    if no_create:
        last_log_path = os.path.join(logs_path, str(i-1))
        if i == 0 or not os.path.exists(last_log_path):
            raise RuntimeError("Dirs should be created first")
        return (models_path, last_log_path)

    # New log
    os.mkdir(log_path)

    # Save arguments
    if args is not None:
        ArgumentSaver.save(os.path.join(log_path, "args.json"), args)

    return (models_path, log_path)
