"""
This module contains the functionality used to define the :rst:dir:`automodsumm`
directive, its supporting configuration values, and the stub file writer.

.. rst:directive:: automodsumm

    The :rst:dir:`automodsumm` directive is a wrapper on Sphinx's
    :rst:dir:`autosummary` directive and, as such, all the options for
    :rst:dir:`autosummary` still work.  The difference, where :rst:dir:`autosummary`
    requires a list of all the objects to document, :rst:dir:`automodsumm`
    only requires the module name and then it will inspect the module to find
    all the objects to be documented according to the listed options.

    The module inspection will respect the ``__all__`` dunder if defined; otherwise,
    it will inspect all objects of the module.  The inspection will only gather
    direct sub-modules and ignore any 3rd party objects, unless listed in
    ``__all__``.

    The behavior of :rst:dir:`automodsumm` can additionally be set with the
    :ref:`configuration values described below <automodsumm-confvals>`.

    .. rst:directive:option:: toctree

        If you want the :rst:dir:`automodsumm` table to serve as a :rst:dir:`toctree`,
        then specify this option with a directory path ``DIRNAME`` with respect to
        the location of your `conf.py` file.

        .. code-block:: rst

            .. automodsumm:: plasmapy_sphinx.automodapi
                :toctree: DIRNAME

        This will signal `sphinx-autogen` to generate stub files for the objects in
        the table and place them in the directory named by ``DIRNAME``.  This behavior
        respects the configuration value :confval:`autosummary_generate`.
        Additionally, :rst:dir:`automodsumm` will not generate stub files for entry
        that falls into the **modules** group (see the
        :rst:dir:`automodsumm:groups` option below), unless
        :confval:`automod_generate_module_stub_files` is set ``True``.

    .. rst:directive:option:: groups

        When a module is inspected all the found objects are categorized into
        groups.  The first group collected is **modules**, followed by any custom
        group defined in :confval:`automod_custom_groups`, and, finally, the
        standard groups of **classes**, **exceptions**, **warnings**, **functions**,
        and **variables** (or all the rest).  By default, **all** groups will
        be included in the generated table.

        Using the `plasmapy_sphinx.automodsumm` module as an example, the
        :ref:`module's API <automodsumm-api>` shows it is made of classes
        and functions.  So the following yields,

        .. code-block:: rst

            .. automodsumm:: plasmapy_sphinx.automodsumm

        .. automodsumm:: plasmapy_sphinx.automodsumm

        However, if you only want to collect classes then one could do

        .. code-block:: rst

            .. automodsumm:: plasmapy_sphinx.automodsumm
               :groups: classes

        .. automodsumm:: plasmapy_sphinx.automodsumm
           :groups: classes

        If you want ot include multiple groups, then specify all groups as a
        comma separated list.

    .. rst:directive:option:: exclude-groups

        This option behaves just like :rst:dir:`automodsumm:groups` except
        you are selectively excluding groups for the generated table.  Using the
        same example as before, a table of just classes could be generated by
        doing

        .. code-block:: rst

            .. automodsumm:: plasmapy_sphinx.automodsumm
               :exclude-groups: functions

        .. automodsumm:: plasmapy_sphinx.automodsumm
           :exclude-groups: functions

    .. rst:directive:option:: skip

        This option allows you to skip (exclude) selected objects from the
        generated table.  The argument is given as a comma separated list of
        the object's short name.  Continuing with the example from above, lets
        skip `~plasmapy_sphinx.automodsumm.AutomodsummRenderer` and
        `~plasmapy_sphinx.automodsumm.GenDocsFromAutomodsumm` from the table.

        .. code-block:: rst

            .. automodsumm:: plasmapy_sphinx.automodsumm
               :groups: classes
               :skip: AutomodsummRenderer, GenDocsFromAutomodsumm

        .. automodsumm:: plasmapy_sphinx.automodsumm
           :groups: classes
           :skip: AutomodsummRenderer, GenDocsFromAutomodsumm

.. _automodsumm-confvals:

Sphinx Configuration Values
---------------------------

A configuration value is a variable that con be defined in `conf.py` to configure
the default behave of related `sphinx` directives.  The configuration values
below relate to the behavior of the :rst:dir:`automodsumm` directive.

.. confval:: automod_custom_groups

    A `sphinx` configuration value used to define custom groups which are used by
    :rst:dir:`automodapi` and :rst:dir:`automodsumm` when sorting the discovered
    objects of an inspected module.  An example custom group definition looks like

    .. code-block:: python

        automod_custom_group = {
            "aliases": {
                "title": "Aliases",
                "dunder": "__aliases__",
            }
        }

    where the top-level key (``"aliases"``) is the group name used in the
    :rst:dir:`automodsumm:groups` option, ``"title"`` defines the title
    text of the group heading, and ``"dunder"`` defines the dunder variable
    (like ``__all__``) in the module.  This dunder variable is then used to
    specify which module objects belong to the custom group.  Using
    `plasmapy.formulary.parameters` as an example, the **aliases** group can
    not be collected and displayed like

    .. code-block:: rst

        .. automodsumm:: plasmapy.formulary.parameters
           :groups: aliases

    .. automodsumm:: plasmapy.formulary.parameters
           :groups: aliases

.. confval:: automod_generate_module_stub_files

    (Default `False`)  By default :rst:dir:`automodsumm` will not generated stub files
    for the **modules** group, even when the `sphinx` configuration value
    `autosummary_generate
    <https://www.sphinx-doc.org/en/master/usage/extensions/autosummary.html?
    highlight=autosummary_generate#confval-autosummary_generate>`_
    is set `True`.  Setting this configure variable to `True` will cause stub
    files to be generated for the **modules** group.

.. confval:: autosummary_generate

    Same as the :rst:dir:`autosummary` configuration value `autosummary_generate
    <https://www.sphinx-doc.org/en/master/usage/extensions/autosummary.html?
    highlight=autosummary_generate#confval-autosummary_generate>`_.
"""
__all__ = [
    "Automodsumm",
    "AutomodsummOptions",
    "AutomodsummRenderer",
    "GenDocsFromAutomodsumm",
    "option_str_list",
    "setup",
]

import os
import re

from importlib import import_module
from jinja2 import TemplateNotFound
from sphinx.application import Sphinx
from sphinx.builders import Builder
from sphinx.ext.autodoc.mock import mock
from sphinx.ext.autosummary import (
    Autosummary,
    get_rst_suffix,
    import_by_name,
    import_ivar_by_name,
)
from sphinx.ext.autosummary.generate import (
    AutosummaryEntry,
    AutosummaryRenderer,
    generate_autosummary_content,
)
from sphinx.locale import __
from sphinx.util import logging
from sphinx.util.osutil import ensuredir
from typing import Any, Callable, Dict, List, Union

from .utils import (
    default_grouping_info,
    find_mod_objs,
    get_custom_grouping_info,
    templates_dir,
)

logger = logging.getLogger(__name__)


def option_str_list(argument):
    """
    An option validator for parsing a comma-separated option argument.  Similar to
    the validators found in `docutils.parsers.rst.directives`.
    """
    if argument is None:
        raise ValueError("argument required but none supplied")
    else:
        return [s.strip() for s in argument.split(",")]


class AutomodsummRenderer(AutosummaryRenderer):
    """
    A helper class for retrieving and rendering :rst:dir:`automodsumm` templates
    when writing stub files.

    Parameters
    ----------

    app : `sphinx.application.Sphinx`
        Instance of the `sphinx` application.

    template_dir : str
        Path to a specified template directory.
    """

    def __init__(self, app: Union[Builder, Sphinx], template_dir: str = None) -> None:

        asumm_path = templates_dir
        relpath = os.path.relpath(asumm_path, start=app.srcdir)
        app.config.templates_path.append(relpath)
        super().__init__(app, template_dir)

    def render(self, template_name: str, context: Dict) -> str:
        """
        Render a template file.  The render will first search for the template in
        the path specified by the sphinx configuration value :confval:`templates_path`,
        then the `~plasmapy_sphinx.templates_dir, and finally the
        :rst:dir:`autosummary` templates directory.  Upon finding the template,
        the values from the ``context`` dictionary will inserted into the
        template and returned.

        Parameters
        ----------
        template_name : str
            Name of the template file.

        context: dict
            Dictionary of values to be rendered (inserted) into the template.
        """
        if not template_name.endswith(".rst"):
            # if does not have '.rst' then objtype likely given for template_name
            template_name += ".rst"

        template = None
        for name in [template_name, "base.rst"]:
            for _path in ["", "automodapi/", "autosummary/"]:
                try:
                    template = self.env.get_template(_path + name)
                    return template.render(context)
                except TemplateNotFound:
                    pass

        if template is None:
            raise TemplateNotFound


class AutomodsummOptions:
    """
    Class for advanced conditioning and manipulation of option arguments for
    `plasmapy_sphinx.automodsum.Automodsumm`.
    """
    option_spec = {
        **Autosummary.option_spec,
        "groups": option_str_list,
        "exclude-groups": option_str_list,
        "skip": option_str_list,
    }
    """
    Mapping of option names to validator functions. (see 
    :attr:`docutils.parsers.rst.Directive.option_spec`)
    """

    _default_grouping_info = default_grouping_info.copy()

    logger = logger
    """
    Instance of the `~sphinx.util.logging.SphinxLoggerAdapter` for report during
    builds.
    """

    def __init__(
        self,
        app: Sphinx,
        modname: str,
        options: Dict[str, Any],
        docname: str = None,
        _warn: Callable = None,
    ):
        """
        Parameters
        ----------
        app : `~sphinx.application.Sphinx`
            Instance of the sphinx application.

        modname : `str`
            Name of the module given in the :rst:dir:`automodsumm` directive.  This
            is the module to be inspected and have it's objects grouped.

        options : Dict[str, Any]
            Dictionary of options given for the :rst:dir:`automodsumm` directive
            declaration.

        docname : str
            Name of the document/file where the :rst:dir:`automodsumm` direction
            was declared.

        _warn : Callable
            Instance of a `sphinx.util.logging.SphinxLoggerAdapter.warning` for
            reporting warning level messages during a build.
        """

        self._app = app
        self._modname = modname
        self._options = options.copy()
        self._docname = docname
        self._warn = _warn if _warn is not None else self.logger.warning

        self.toctree = {
            "original": None,
            "rel_to_doc": None,
            "abspath": None,
        }  # type: Dict[str, Union[str, None]]

        self.condition_options()

    @property
    def app(self) -> Sphinx:
        """Instance of the sphinx application."""
        return self._app

    @property
    def modname(self) -> str:
        """Name of the module given to :rst:dir:`automodsumm`."""
        return self._modname

    @property
    def options(self) -> Dict[str, Any]:
        """Copy of the options given to :rst:dir:`automodsumm`."""
        return self._options

    @property
    def docname(self) -> str:
        """Name of the document where :rst:dir:`automodsumm` was declared."""
        return self._docname

    @property
    def warn(self) -> Callable:
        """
        Instance of a `sphinx.util.logging.SphinxLoggerAdapter.warning` for
        reporting warning level messages during a build.
        """
        return self._warn

    @property
    def pkg_or_module(self) -> str:
        """
        Is module specified by :attr:`modname` a package or module (i.e. `.py` file).
        Return ``"pkg"`` for a package and ``"module"`` for a `.py` file.
        """
        mod = import_module(self.modname)
        if mod.__package__ == mod.__name__:
            return "pkg"
        else:
            return "module"

    def condition_options(self):
        """
        Method for doing any additional conditioning of option arguments.
        Called during class instantiation."""
        self.condition_toctree_option()
        self.condition_group_options()

    def condition_toctree_option(self):
        """
        Additional conditioning of the option argument ``toctree``. (See
        :rst:dir:`automodsumm:toctree` for additional details.)
        """
        if "toctree" not in self.options:
            return

        if self.docname is None:
            doc_path = self.app.confdir
        else:
            doc_path = os.path.dirname(os.path.join(self.app.srcdir, self.docname))

        self.toctree["original"] = self.options["toctree"]
        self.toctree["abspath"] = os.path.abspath(
            os.path.join(self.app.confdir, self.options["toctree"]),
        )
        self.toctree["rel_to_doc"] = os.path.relpath(
            self.toctree["abspath"], doc_path
        ).replace(os.sep, "/")

        self.options["toctree"] = self.toctree["rel_to_doc"]

    def condition_group_options(self):
        """
        Additional conditioning of the option arguments ``groups`` and
        ``exclude-groups``.  (See :rst:dir:`automodsumm:groups` and
        :rst:dir:`automodsumm:exclude-groups` for additional details.)
        """
        allowed_args = self.groupings | {"all"}
        do_groups = self.groupings.copy()  # defaulting to all groups

        # groups option
        if "groups" in self.options:
            opt_args = set(self.options["groups"])

            unknown_args = opt_args - allowed_args
            if len(unknown_args) > 0:
                self.warn(
                    f"Option 'groups' has unrecognized arguments "
                    f"{unknown_args}. Ignoring."
                )
                opt_args = opt_args - unknown_args

            if "all" not in opt_args:
                do_groups = opt_args

        # exclude groupings
        if "exclude-groups" in self.options:
            opt_args = set(self.options["exclude-groups"])
            del self.options["exclude-groups"]
        else:
            opt_args = set()

        unknown_args = opt_args - allowed_args
        if len(unknown_args) > 0:
            self.warn(
                f"Option 'exclude-groups' has unrecognized arguments "
                f"{unknown_args}. Ignoring."
            )
            opt_args = opt_args - unknown_args
        elif "all" in opt_args:
            self.warn(
                f"Arguments of 'groups' and 'exclude-groups' results in no content."
            )
            self.options["groups"] = []
            return

        do_groups = do_groups - opt_args
        self.options["groups"] = list(do_groups)

    @property
    def mod_objs(self) -> Dict[str, Dict[str, Any]]:
        """
        Dictionary of the grouped objects found in the module named by :attr:`modname`.

        See Also
        --------
        plasmapy_sphinx.utils.find_mod_objs
        """
        return find_mod_objs(self.modname, app=self.app)

    @property
    def groupings(self) -> set:
        """Set of all the grouping names."""
        return set(self.grouping_info)

    @property
    def default_grouping_info(self) -> Dict[str, Dict[str, str]]:
        """
        Dictionary of the default group information.

        See Also
        --------
        plasmapy_sphinx.utils.default_grouping_info
        """
        return self._default_grouping_info.copy()

    @property
    def custom_grouping_info(self) -> Dict[str, Dict[str, str]]:
        """
        Dictionary of the custom group info.

        See Also
        --------
        plasmapy_sphinx.utils.get_custom_grouping_info
        """
        return get_custom_grouping_info(self.app)

    @property
    def grouping_info(self) -> Dict[str, Dict[str, str]]:
        """
        The combined grouping info of :attr:`default_grouping_info` and
        :attr:`custom_grouping_info`
        """
        grouping_info = self.default_grouping_info
        grouping_info.update(self.custom_grouping_info)
        return grouping_info

    @property
    def mod_objs_option_filtered(self) -> Dict[str, Dict[str, Any]]:
        """
        A filtered version of :attr:`mod_objs` according to the specifications
        given in :attr:`options` (i.e. those given to :rst:dir:`automodsumm`).
        """
        try:
            mod_objs = self.mod_objs
        except ImportError:
            mod_objs = {}
            self.warn(f"Could not import module {self.modname}")

        do_groups = set(self.options["groups"])

        if len(do_groups) == 0:
            return {}

        # remove excluded groups
        for group in list(mod_objs):
            if group not in do_groups:
                del mod_objs[group]

        # objects to skip
        skip_names = set()
        if "skip" in self.options:
            skip_names = set(self.options["skip"])

        # filter out skipped objects
        for group in list(mod_objs.keys()):

            names = mod_objs[group]["names"]
            qualnames = mod_objs[group]["qualnames"]
            objs = mod_objs[group]["objs"]

            names_filtered = []
            qualnames_filtered = []
            objs_filtered = []

            for name, qualname, obj in zip(names, qualnames, objs):
                if not (name in skip_names or qualname in skip_names):
                    names_filtered.append(name)
                    qualnames_filtered.append(qualname)
                    objs_filtered.append(obj)

            if len(names) == 0:
                del mod_objs[group]
                continue

            mod_objs[group] = {
                "names": names_filtered,
                "qualnames": qualnames_filtered,
                "objs": objs_filtered,
            }
        return mod_objs

    def generate_obj_list(self, exclude_modules: bool = False) -> List[str]:
        """
        Take :attr:`mod_objs_option_filtered` and generated a list of the fully
        qualified objects names names.  The list is sorted based on the casefolded
        short names of the objects.

        Parameters
        ----------
        exclude_modules : bool
            (Default `False`) Set `True` to exclude the qualified names related to
            objects sorted in the **modules** group.
        """

        mod_objs = self.mod_objs_option_filtered

        if not bool(mod_objs):
            return []

        gather_groups = set(mod_objs.keys())
        if exclude_modules:
            gather_groups.discard("modules")

        names = []
        qualnames = []
        for group in gather_groups:
            names.extend(mod_objs[group]["names"])
            qualnames.extend(mod_objs[group]["qualnames"])

        content = [
            qualname for name, qualname in
            sorted(zip(names, qualnames), key=lambda x: str.casefold(x[0]))
        ]

        return content


class Automodsumm(Autosummary):
    required_arguments = 1
    optional_arguments = 0
    has_content = False
    option_spec = AutomodsummOptions.option_spec.copy()

    logger = logger

    def run(self):
        env = self.env
        modname = self.arguments[0]

        # for some reason, even though ``currentmodule`` is substituted in,
        # sphinx doesn't necessarily recognize this fact.  So we just force
        # it internally, and that seems to fix things
        env.temp_data["py:module"] = modname
        env.ref_context["py:module"] = modname

        nodelist = []

        # update toctree with relative path to file (not confdir)
        if "toctree" in self.options:
            self.options["toctree"] = self.option_processor().options["toctree"]

        # define additional content
        content = self.option_processor().generate_obj_list()
        for ii, modname in enumerate(content):
            if not modname.startswith("~"):
                content[ii] = "~" + modname
        self.content = content

        nodelist.extend(Autosummary.run(self))
        return nodelist

    def option_processor(self):
        processor = AutomodsummOptions(
            app=self.env.app,
            modname=self.arguments[0],
            options=self.options,
            docname=self.env.docname,
            _warn=self.warn,
        )
        return processor

    def get_items(self, names):
        try:
            self.bridge.genopt["imported-members"] = True
        except AttributeError:  # Sphinx < 4.0
            self.genopt["imported-members"] = True
        return Autosummary.get_items(self, names)


class GenDocsFromAutomodsumm:
    """Needed so stub file are automatically generated."""

    option_spec = AutomodsummOptions.option_spec.copy()

    _re = {
        "automodsumm": re.compile(r"^\n?(\s*)\.\.\s+automodsumm::\s*(\S+)\s*(?:\n|$)"),
        "automodapi": re.compile(r"^\n?(\s*)\.\.\s+automodapi::\s*(\S+)\s*(?:\n|$)"),
        "option": re.compile(r"^\n?(\s+):(\S*):\s*(\S.*|)\s*(?:\n|$)"),
        "currentmodule": re.compile(
            r"^\s*\.\.\s+(|\S+:)(current)?module::\s*([a-zA-Z0-9_.]+)\s*$"
        ),
    }

    app = None  # type: Sphinx
    logger = logger

    def __call__(self, app: Sphinx):
        """
        This routine is adapted from
        :func:`sphinx.ext.autosummary.process_generate_options` to scan through
        the source files, check for the `automodsumm` directive, and auto
        generate any associated stub files.
        """
        self.app = app
        genfiles = app.config.autosummary_generate

        if genfiles is True:
            env = app.builder.env
            genfiles = [
                env.doc2path(x, base=None)
                for x in env.found_docs
                if os.path.isfile(env.doc2path(x))
            ]
        elif genfiles is False:
            pass
        else:
            ext = list(app.config.source_suffix)
            genfiles = [
                genfile + (ext[0] if not genfile.endswith(tuple(ext)) else "")
                for genfile in genfiles
            ]

            for entry in genfiles[:]:
                if not os.path.isfile(os.path.join(app.srcdir, entry)):
                    self.logger.warning(
                        __(f"automodsumm_generate: file not found: {entry}")
                    )
                    genfiles.remove(entry)

        if not genfiles:
            return

        suffix = get_rst_suffix(app)
        if suffix is None:
            self.logger.warning(
                __(
                    "automodsumm generates .rst files internally. "
                    "But your source_suffix does not contain .rst. Skipped."
                )
            )
            return

        # from sphinx.ext.autosummary.generate import generate_autosummary_docs

        imported_members = app.config.autosummary_imported_members
        with mock(app.config.autosummary_mock_imports):
            self.generate_docs(
                genfiles,
                suffix=suffix,
                base_path=app.srcdir,
                imported_members=imported_members,
                overwrite=app.config.autosummary_generate_overwrite,
                encoding=app.config.source_encoding,
            )

    def find_mod_objs(self, modname: str):
        return find_mod_objs(modname, app=self.app)

    def generate_docs(
        self,
        source_filenames: List[str],
        output_dir: str = None,
        suffix: str = ".rst",
        base_path: str = None,
        imported_members: bool = False,
        overwrite: bool = True,
        encoding: str = "utf-8",
    ) -> None:
        """
        This code was adapted from
        :func:`sphinx.ext.autosummary.generate.generate_autosummary_docs`.
        """
        app = self.app

        _info = self.logger.info
        _warn = self.logger.warning

        showed_sources = list(sorted(source_filenames))
        _info(
            __(f"[automodsumm] generating stub files for {len(showed_sources)} sources")
        )

        if output_dir:
            _info(__(f"[automodsumm] writing to {output_dir}"))

        if base_path is not None:
            source_filenames = [
                os.path.join(base_path, filename) for filename in source_filenames
            ]

        template = AutomodsummRenderer(app)

        # read
        items = self.find_in_files(source_filenames)

        # keep track of new files
        new_files = []

        if app:
            filename_map = app.config.autosummary_filename_map
        else:
            filename_map = {}

        # write
        for entry in sorted(set(items), key=str):
            if entry.path is None:
                # The corresponding automodsumm:: directive did not have
                # a :toctree: option
                continue

            path = output_dir or os.path.abspath(entry.path)
            ensuredir(path)

            try:
                name, obj, parent, modname = import_by_name(entry.name)
                qualname = name.replace(modname + ".", "")
            except ImportError as e:
                try:
                    # try to import as an instance attribute
                    name, obj, parent, modname = import_ivar_by_name(entry.name)
                    qualname = name.replace(modname + ".", "")
                except ImportError:
                    _warn(__(f"[automodsumm] failed to import {entry.name}: {e}"))
                    continue

            context = {}
            if app:
                context.update(app.config.autosummary_context)

            content = generate_autosummary_content(
                name,
                obj,
                parent,
                template,
                entry.template,
                imported_members,
                app,
                entry.recursive,
                context,
                modname,
                qualname,
            )

            filename = os.path.join(path, filename_map.get(name, name) + suffix)
            if os.path.isfile(filename):
                with open(filename, encoding=encoding) as f:
                    old_content = f.read()

                if content == old_content:
                    continue
                elif overwrite:  # content has changed
                    with open(filename, "w", encoding=encoding) as f:
                        f.write(content)
                    new_files.append(filename)
            else:
                with open(filename, "w", encoding=encoding) as f:
                    f.write(content)
                new_files.append(filename)

        # descend recursively to new files
        if new_files:
            self.generate_docs(
                new_files,
                output_dir=output_dir,
                suffix=suffix,
                base_path=base_path,
                imported_members=imported_members,
                overwrite=overwrite,
            )

    def find_in_files(self, filenames: List[str]) -> List[AutosummaryEntry]:
        """
        Adapted from :func:`sphinx.ext.autosummary.generate.find_autosummary_in_files`.

        Find out what items are documented in `source/*.rst`.
        """
        documented = []  # type: List[AutosummaryEntry]
        for filename in filenames:
            with open(filename, encoding="utf-8", errors="ignore") as f:
                lines = f.read().splitlines()
                documented.extend(self.find_in_lines(lines, filename=filename))
        return documented

    def find_in_lines(
        self,
        lines: List[str],
        filename: str = None,
    ) -> List[AutosummaryEntry]:
        """
        Adapted from :func:`sphinx.ext.autosummary.generate.find_autosummary_in_lines`.

        Find out what items appear in automodsumm:: directives in the given lines.
        """

        from .automodapi import AutomodapiOptions

        documented = []  # type: List[AutosummaryEntry]

        current_module = None
        modname = ""

        options = {}  # type: Dict[str, Any]

        _option_cls = None

        in_automod_directive = False
        gather_objs = False

        last_line = False
        nlines = len(lines)

        for ii, line in enumerate(lines):
            if ii == nlines - 1:
                last_line = True

            # looking for option `   :option: option_args`
            if in_automod_directive:
                match = self._re["option"].search(line)
                if match is not None:
                    option_name = match.group(2)
                    option_args = match.group(3)
                    try:
                        option_args = _option_cls.option_spec[option_name](option_args)
                        options[option_name] = option_args
                    except (KeyError, TypeError):
                        pass
                else:
                    # done reading options
                    in_automod_directive = False
                    gather_objs = True

                if last_line:
                    # end of lines reached
                    in_automod_directive = False
                    gather_objs = True

                if in_automod_directive:
                    continue

            # looking for `.. automodsumm:: <modname>`
            match = self._re["automodsumm"].search(line)
            if match is not None:
                in_automod_directive = True
                # base_indent = match.group(1)
                modname = match.group(2)

                if current_module is None or modname == current_module:
                    pass
                elif not modname.startswith(f"{current_module}."):
                    modname = f"{current_module}.{modname}"
                _option_cls = AutomodsummOptions
                self.logger.info(f"[automodsumm] {modname}")

                if last_line:
                    # end of lines reached
                    in_automod_directive = False
                    gather_objs = True
                else:
                    continue

            # looking for `.. automodapi:: <modname>`
            match = self._re["automodapi"].search(line)
            if match is not None:
                in_automod_directive = True
                # base_indent = match.group(1)
                modname = match.group(2)

                if current_module is None or modname == current_module:
                    pass
                elif not modname.startswith(f"{current_module}."):
                    modname = f"{current_module}.{modname}"
                _option_cls = AutomodapiOptions

                if last_line:
                    # end of lines reached
                    in_automod_directive = False
                    gather_objs = True
                else:
                    continue

            # looking for `.. py:currentmodule:: <current_module>`
            match = self._re["currentmodule"].search(line)
            if match is not None:
                current_module = match.group(3)
                continue

            # gather objects and update documented list
            if gather_objs:
                process_options = _option_cls(
                    self.app,
                    modname,
                    options,
                    docname=filename,
                    _warn=self.logger.warning,
                )
                options = {
                    "toctree": process_options.toctree["abspath"],
                    "template": process_options.options.get("template", None),
                    "recursive": process_options.options.get("recursive", False),
                }

                exclude_modules = not self.app.config.automod_generate_module_stub_files
                obj_list = process_options.generate_obj_list(
                    exclude_modules=exclude_modules
                )

                for name in obj_list:
                    documented.append(
                        AutosummaryEntry(
                            name=name,
                            path=options["toctree"],
                            template=options["template"],
                            recursive=options["recursive"],
                        )
                    )

                self.logger.info(
                    f"[automodsumm stub file gen] collected {len(obj_list):4d} "
                    f"object(s) in '{modname}'"
                )

                # reset for next search
                options = {}
                gather_objs = False
                _option_cls = None

        return documented


def setup(app: Sphinx):

    app.setup_extension("sphinx.ext.autosummary")

    app.add_directive("automodsumm", Automodsumm)

    gendocs_from_automodsumm = GenDocsFromAutomodsumm()
    app.connect("builder-inited", gendocs_from_automodsumm)

    app.add_config_value("automod_custom_groups", dict(), True)
    app.add_config_value("automod_generate_module_stub_files", False, True)

    return {"parallel_read_safe": True, "parallel_write_safe": True}
