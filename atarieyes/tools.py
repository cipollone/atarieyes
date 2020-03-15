"""Generic utilities."""

from abc import ABCMeta


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
