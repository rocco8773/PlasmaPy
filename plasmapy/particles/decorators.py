"""
Module used to define the framework needed for the `particle_input` decorator.
The decorator takes string and/or integer representations of particles
as arguments and passes through the corresponding instance of the
`~plasmapy.particles.Particle` class.
"""
__all__ = [
    "particle_input",
    "ParticleLike",
]

import functools
import inspect
import numbers

from typing import Any, Callable, Dict, List, Mapping, Optional, Set, Tuple, Union

from plasmapy.particles.exceptions import (
    AtomicError,
    ChargeError,
    InvalidElementError,
    InvalidIonError,
    InvalidIsotopeError,
    InvalidParticleError,
)
from plasmapy.particles.particle_class import AbstractParticle, Particle
from plasmapy.utils.decorators import preserve_signature


ParticleLike = Union[str, int, numbers.Integral, Particle, AbstractParticle]


def _particle_errmsg(
    argname: str,
    argval: str,
    Z: int = None,
    mass_numb: int = None,
    funcname: str = None,
) -> str:
    """
    Return a string with an appropriate error message for an
    `~plasmapy.utils.InvalidParticleError`.
    """
    errmsg = f"In {funcname}, {argname} = {repr(argval)} "
    if mass_numb is not None or Z is not None:
        errmsg += "with "
    if mass_numb is not None:
        errmsg += f"mass_numb = {repr(mass_numb)} "
    if mass_numb is not None and Z is not None:
        errmsg += "and "
    if Z is not None:
        errmsg += f"integer charge Z = {repr(Z)} "
    errmsg += "does not correspond to a valid particle."
    return errmsg


def _category_errmsg(particle, require, exclude, any_of, funcname) -> str:
    """
    Return an appropriate error message for when a particle does not
    meet the required categorical specifications.
    """
    category_errmsg = (
        f"The particle {particle} does not meet the required "
        f"classification criteria to be a valid input to {funcname}. "
    )

    errmsg_table = [
        (require, "must belong to all"),
        (any_of, "must belong to any"),
        (exclude, "cannot belong to any"),
    ]

    for condition, phrase in errmsg_table:
        if condition:
            category_errmsg += (
                f"The particle {phrase} of the following categories: " f"{condition}. "
            )

    return category_errmsg


def _search_annotations_for_particle(func: Callable) -> Dict[str, bool]:
    """
    Search through a function's annotations to find which arguments are typed
    with `ParticleLike` or a subclass of
    `~plasmapy.particles.particle_class.AbstractParticle`.

    Parameters
    ----------
    func:
        The function who's annotations will be examined.

    Returns
    -------
    Dict[str, bool]
        A dictionary where the keys are the names of the function parameters and
        the associated values are `True` if the parameter was annotated with
        `ParticleLike` or a subclass of
        `~plasmapy.particles.particle_class.AbstractParticle`.

    """
    params = inspect.signature(func).parameters  # type: Mapping[str, inspect.Parameter]

    def measure_depth(annotations, depth=0):
        pass

    def examine_args(args: tuple) -> bool:
        """
        Examine an annotations arguments for `Particlelike` typing.  In order

        Parameters
        ----------
        args: tuple
            A tuple of the annotation arguments.

        Returns
        -------
        bool
            `True` if (1) at least one argument is `ParticleLike` or a subclass
            of `~plasmapy.particles.particle_class.AbstractParticle` and (2)
            all arguments are a subclass of one the types making up
            `ParticleLike` or an `Ellipsis` or `NoneType`.

        """
        status_list = []
        for arg in args:
            # check annotation argument at macro-level
            if arg is ParticleLike:
                isparlike = True
            elif inspect.isclass(arg) and issubclass(arg, AbstractParticle):
                isparlike = True
            elif arg is Ellipsis:
                status_list.append(True)
                continue
            else:
                isparlike = False

            if isparlike:
                status_list.append(True)
                continue

            # examine annotation argument against those that make up ParticleLike
            valids = list((*ParticleLike.__args__, type(None)))
            valids.remove(AbstractParticle)
            valids.remove(Particle)
            for valid in valids:
                try:
                    isparlike = issubclass(arg, valid)
                except TypeError:
                    isparlike = False

                if isparlike:
                    break

            status_list.append(isparlike)

        return len(status_list) > 0 and all(status_list)

    found = {}
    for name, param in params.items():
        annotation = param.annotation

        # if annotated with list or tuple, then replace with Tuple or List
        # respectively
        if type(annotation) is tuple:
            annotation = Tuple[annotation]
        elif type(annotation) is list:
            annotation = List[annotation]

        # Let's look for a Particle annotation
        if annotation is ParticleLike:
            status = True
        elif inspect.isclass(annotation) and issubclass(annotation, AbstractParticle):
            status = True
        elif hasattr(annotation, "__origin__"):
            origin = annotation.__origin__
            if origin == Union or origin is list or origin is tuple:
                status = examine_args(annotation.__args__)
            else:
                status = False
        else:
            status = False

        found[name] = status

    return found


def particle_input(
    func: Callable = None,
    require: Union[str, Set, List, Tuple] = None,
    any_of: Union[str, Set, List, Tuple] = None,
    exclude: Union[str, Set, List, Tuple] = None,
    none_shall_pass: bool = False,
) -> Any:
    """
    Convert arguments to methods and functions to
    `~plasmapy.particles.Particle` objects.

    Take positional and keyword arguments that are annotated with
    `~plasmapy.particles.Particle`, and pass through the
    `~plasmapy.particles.Particle` object corresponding to those arguments
    to the decorated function or method.

    Optionally, raise an exception if the particle does not satisfy the
    specified categorical criteria.

    Parameters
    ----------
    func : `callable`
        The function or method to be decorated.

    require : `str`, `set`, `list`, or `tuple`, optional
        Categories that a particle must be in.  If a particle is not in
        all of these categories, then an `~plasmapy.utils.AtomicError`
        will be raised.

    any_of : `str`, `set`, `list`, or `tuple`, optional
        Categories that a particle may be in.  If a particle is not in
        any of these categories, then an `~plasmapy.utils.AtomicError`
        will be raised.

    exclude : `str`, `set`, `list`, or `tuple`, optional
        Categories that a particle cannot be in.  If a particle is in
        any of these categories, then an `~plasmapy.utils.AtomicError`
        will be raised.

    none_shall_pass : `bool`, optional
        If set to `True`, then the decorated argument may be set to
        `None` without raising an exception.  In such cases, this
        decorator will pass `None` through to the decorated function or
        method.  If set to `False` and the annotated argument is given
        a value of `None`, then this decorator will raise a `TypeError`.

    Notes
    -----
    If the annotated argument is named `element`, `isotope`, or `ion`,
    then the decorator will raise an `~plasmapy.utils.InvalidElementError`,
    `~plasmapy.utils.InvalidIsotopeError`, or `~plasmapy.utils.InvalidIonError`
    if the particle does not correspond to an element, isotope, or ion,
    respectively.

    If exactly one argument is annotated with `~plasmapy.particles.Particle`,
    then the keywords ``Z`` and ``mass_numb`` may be used to specify the
    integer charge and/or mass number of an ion or isotope.  However,
    the decorated function must allow ``Z`` and/or ``mass_numb`` as keywords
    in order to enable this functionality.

    Raises
    ------
    `TypeError`
        If the annotated argument is not a `str`, `int`, `tuple`, `list`
        or `~plasmapy.particles.Particle`; or if ``Z`` or ``mass_numb`` is
        not an `int`.

    `ValueError`
        If the number of input elements in a collection do not match the
        number of expected elements.

    `~plasmapy/utils/InvalidParticleError`
        If the annotated argument does not correspond to a valid
        particle.

    `~plasmapy/utils/InvalidElementError`
        If an annotated argument is named ``element``, and the input
        does not correspond to an element, isotope, or ion.

    `~plasmapy/utils/InvalidIsotopeError`
        If an annotated argument is named ``isotope``, and the input
        does not correspond to an isotope or an ion of an isotope.

    `~plasmapy/utils/InvalidIonError`
        If an annotated argument is named ``ion``, and the input does
        not correspond to an ion.

    `~plasmapy/utils/ChargeError`
        If ``'charged'`` is in the ``require`` argument and the particle
        is not explicitly charged, or if ``any_of = {'charged',
        'uncharged'}`` and the particle does not have charge information
        associated with it.

    `~plasmapy/utils/AtomicError`
        If an annotated argument does not meet the criteria set by the
        categories in the ``require``, ``any_of``, and ``exclude``
        keywords; if more than one argument is annotated and ``Z`` or
        ``mass_numb`` are used as arguments; or if none of the arguments
        have been annotated with `~plasmapy.particles.Particle`.

    Examples
    --------
    The following simple decorated function returns the
    `~plasmapy.particles.Particle` object created from the function's
    sole argument:

    .. code-block:: python

        from plasmapy.particles import particle_input, Particle
        @particle_input
        def decorated_function(particle: Particle):
            return particle

    This decorator may also be used to accept arguments using tuple
    annotation containing specific number of elements or using list
    annotation which accepts any number of elements in an iterable.
    Returns a tuple of `~plasmapy.particles.Particle`:

    .. code-block:: python

        from plasmapy.particles import particle_input, Particle
        @particle_input
        def decorated_tuple_function(particles: (Particle, Particle)):
            return particles
        sample_particles = decorated_tuple_function(('He', 'Li'))

        @particle_input
        def decorated_list_function(particles: [Particle]):
            return particles
        sample_particles = decorated_list_function(('Al 3+', 'C'))
        sample_particles = decorated_list_function(['He', 'Ne', 'Ar'])

    This decorator may be used for methods in instances of classes, as
    in the following example:

    .. code-block:: python

        from plasmapy.particles import particle_input, Particle
        class SampleClass:
            @particle_input
            def decorated_method(self, particle: Particle):
                return particle
        sample_instance = SampleClass()
        sample_instance.decorated_method('Fe')

    Some functions may intended to be used with only certain categories
    of particles.  The ``require``, ``any_of``, and ``exclude`` keyword
    arguments enable this functionality.

    .. code-block:: python

        from plasmapy.particles import particle_input, Particle
        @particle_input(
            require={'matter'},
            any_of={'charged', 'uncharged},
            exclude={'neutrino', 'antineutrino'},
        )
        def selective_function(particle: Particle):
            return particle

    """

    if exclude is None:
        exclude = set()
    if any_of is None:
        any_of = set()
    if require is None:
        require = set()

    def decorator(f: Callable):
        signature = inspect.signature(f)

        @preserve_signature
        @functools.wraps(f)
        def wrapper(*args, **kwargs):
            funcname = f.__name__

            # annotations = f.__annotations__
            argnames = list(signature.parameters)
            bound_args = signature.bind(*args, **kwargs)
            bound_args.apply_defaults()

            # search for parameters annotated as ParticleLike
            args_w_par = _search_annotations_for_particle(f)
            for name, isp in args_w_par.copy().items():
                if not isp:
                    del args_w_par[name]

            # some warnings
            if len(args_w_par) == 0:
                raise AtomicError(
                    f"None of the arguments or keywords to {funcname} have been "
                    f"annotated with Particle or ParticleLike, as required by "
                    f"the @particle_input decorator."
                )
            elif len(args_w_par) > 1 and ("Z" in argnames or "mass_numb" in argnames):
                raise AtomicError(
                    f"The arguments Z and mass_numb in {funcname} are not "
                    f"allowed when more than one argument or keyword is "
                    f"annotated with Particle in functions decorated "
                    f"with @particle_input."
                )
            else:
                # If the number of arguments and keywords annotated with
                # Particle is exactly one, then the Z and mass_numb keywords
                # can be used without potential for ambiguity.

                Z = bound_args.arguments.get("Z", None)
                mass_numb = bound_args.arguments.get("mass_numb", None)

            # let's convert
            for argname in args_w_par:
                arg = bound_args.arguments[argname]
                annotation = signature.parameters[argname].annotation

                if hasattr(annotation, "__args__"):
                    can_be_none = none_shall_pass or type(None) in annotation.__args__
                else:
                    can_be_none = none_shall_pass

                # get the expected type
                if hasattr(annotation, "__origin__") and annotation.__origin__ is list:
                    expected_type = list
                elif hasattr(annotation, "__origin__") \
                        and annotation.__origin__ is tuple:
                    expected_type = tuple
                else:
                    expected_type = ParticleLike.__args__

                # check we got the expected type
                if not isinstance(arg, expected_type):
                    raise TypeError(
                        f"Got type {type(arg)} for argument {argname}, expected "
                        f"type {expected_type}."
                    )

                # convert to a particle class (if necessary)
                if expected_type in (list, tuple):
                    new_arg = []

                    for val in arg:
                        if val is None and can_be_none:
                            pass
                        elif not isinstance(val, AbstractParticle):
                            val = get_particle(
                                argname=argname,
                                params=(val, Z, mass_numb),
                                already_particle=False,
                                funcname=funcname,
                            )
                        new_arg.append(val)
                    new_arg = expected_type(new_arg)
                else:
                    if arg is None and can_be_none:
                        new_arg = arg
                    elif not isinstance(arg, AbstractParticle):
                        new_arg = get_particle(
                            argname=argname,
                            params=(arg, Z, mass_numb),
                            already_particle=False,
                            funcname=funcname,
                        )
                    else:
                        new_arg = arg
                bound_args.arguments[argname] = new_arg

            return f(*bound_args.args, **bound_args.kwargs)

        return wrapper

    def get_particle(argname, params, already_particle, funcname):
        argval, Z, mass_numb = params

        # Convert the argument to a Particle object if it is not
        # already one.

        if not already_particle:

            if not isinstance(argval, (numbers.Integral, str, tuple, list)):
                raise TypeError(
                    f"The argument {argname} to {funcname} must be "
                    f"a string, an integer or a tuple or list of them "
                    f"corresponding to an atomic number, or a "
                    f"Particle object."
                )

            try:
                particle = Particle(argval, Z=Z, mass_numb=mass_numb)
            except InvalidParticleError as e:
                raise InvalidParticleError(
                    _particle_errmsg(argname, argval, Z, mass_numb, funcname)
                ) from e

        # We will need to do the same error checks whether or not the
        # argument is already an instance of the Particle class.

        if already_particle:
            particle = argval

        # If the name of the argument annotated with Particle in the
        # decorated function is element, isotope, or ion; then this
        # decorator should raise the appropriate exception when the
        # particle ends up not being an element, isotope, or ion.

        cat_table = [
            ("element", particle.element, InvalidElementError),
            ("isotope", particle.isotope, InvalidIsotopeError),
            ("ion", particle.ionic_symbol, InvalidIonError),
        ]

        for category_name, category_symbol, CategoryError in cat_table:
            if argname == category_name and not category_symbol:
                raise CategoryError(
                    f"The argument {argname} = {repr(argval)} to "
                    f"{funcname} does not correspond to a valid "
                    f"{argname}."
                )

        # Some functions require that particles be charged, or
        # at least that particles have charge information.

        _integer_charge = particle._attributes["integer charge"]

        must_be_charged = "charged" in require
        must_have_charge_info = set(any_of) == {"charged", "uncharged"}

        uncharged = _integer_charge == 0
        lacks_charge_info = _integer_charge is None

        if must_be_charged and (uncharged or must_have_charge_info):
            raise ChargeError(f"A charged particle is required for {funcname}.")

        if must_have_charge_info and lacks_charge_info:
            raise ChargeError(f"Charge information is required for {funcname}.")

        # Some functions require particles that belong to more complex
        # classification schemes.  Again, be sure to provide a
        # maximally useful error message.

        if not particle.is_category(require=require, exclude=exclude, any_of=any_of):
            raise AtomicError(
                _category_errmsg(particle, require, exclude, any_of, funcname)
            )

        return particle

    # The following code allows the decorator to be used either with or
    # without arguments.  This allows us to invoke the decorator either
    # as `@particle_input` or as `@particle_input()`, where the latter
    # call allows the decorator to have keyword arguments.

    if func is not None:
        return decorator(func)
    else:
        return decorator
