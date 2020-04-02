"""Python utilities."""

from abc import ABCMeta
import signal


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
                "class __init__ did not define these abstract attributes:\n" +
                str(not_defined))

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
