import collections
import itertools
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Type, Any
from rich.tree import Tree


from astromodels.utils.logging import setup_logger

from .cpickle_compatibility_layer import cPickle

log = setup_logger(__name__)


# This is necessary for pickle to be able to reconstruct a NewNode class (or derivate)
# during unpickling
class NewNodeUnpickler(object):
    def __call__(self, cls):

        instance = cls.__new__(cls)

        return instance


@dataclass(repr=False, unsafe_hash=True)
class NodeBase:
    _name: str
    _parent: Optional[Type["NodeBase"]] = field(repr=False, default=None)
    _children: Dict[str, Type["NodeBase"]] = field(
        default_factory=dict, repr=False, compare=False
    )
    _path: Optional[str] = field(repr=False, default="")

    # The next 3 methods are *really* necessary for anything to work

    def __reduce__(self):

        state = {}
        state["parent"] = self._get_parent()
        state["path"] = self._path
        state["children"] = self._get_children()
        state["child_names"] = [child.name for child in state["children"]]
        state["name"] = self._name
        state["__dict__"] = self.__dict__

        return NewNodeUnpickler(), (self.__class__,), state

    def __setstate__(self, state) -> None:

        self._children = {}

        # Set the name of this node

        self._name = state["name"]

        # Set the parent

        self._parent = state["parent"]

        # set the path

        self._path = state["path"]

        # Set the children
        # do this manually to avoid recursion
        # issues for children not yet built

        for child, name in zip(state["children"], state["child_names"]):

            self._children[name] = child

        # Restore everything else

        for k in state["__dict__"]:

            self.__dict__[k] = state["__dict__"][k]

    # This is necessary for copy.deepcopy to work
    def __deepcopy__(self, memodict={}):

        return cPickle.loads(cPickle.dumps(self))

    def _add_child(self, child: Type["NodeBase"]) -> None:

        if not isinstance(child, NodeBase):

            log.error(f"{child} is not of type Node")

            raise TypeError()

        log.debug_node(f"adding child {child._name}")

        if child._name not in self._children:

            # add the child to the dict

            self._children[child._name] = child

            # set the parent

            child._set_parent(self)

            # now go through and make sure
            # all the children know about the
            # new parent recursively

            child._update_child_path()

        else:

            log.error(f"A child with name {child._name} already exists")

            raise AttributeError()

    def _add_children(self, children: List[Type["NodeBase"]]) -> None:

        for child in children:
            self._add_child(child)

    def _remove_child(self, name: str, delete: bool = True) -> Optional["NodeBase"]:
        """
        return a child
        """

        # this kills the child

        if delete:

            del self._children[name]

            # return none

        # we want to keep the child
        # but orphan it

        else:

            child = self._children.pop(name)

            # now orphan the child

            child._orphan()

            # we want to get the
            # orphan back

            return child

            # return self._children.pop(name)

    def _orphan(self) -> None:
        """
        This will disconnect the current node from its parent
        and inform all the children about the change
        """

        # disconnect from parent

        self._parent = None

        # be nice to the kids and tell them

        self._update_child_path()

    def _set_parent(self, parent: Type["NodeBase"]) -> None:
        """
        set the parent and update path
        """

        self._parent = parent

        parent_path = self._parent._get_path()

        if parent_path == "__root__":

            self._path = f"{self._name}"

        else:

            self._path = f"{parent_path}.{self._name}"

        log.debug_node(f"path is now: {self._path}")

    def _get_child(self, name: str) -> "NodeBase":
        """
        return a child object
        """

        return self._children[name]

    def _has_child(self, name: str) -> bool:
        """
        is this child (name) in the tree
        """
        return name in self._children

    def _get_children(self) -> Tuple["NodeBase"]:
        """
        return a tuple of children
        """

        return tuple(self._children.values())

    def _get_child_from_path(self, path: str) -> "NodeBase":
        """
        get a child from a string path
        """
        nodes = path.split(".")
        _next = self
        for node in nodes:
            _next = _next._get_child(node)

        return _next

    def __getitem__(self, key) -> "NodeBase":
        return self._get_child_from_path(key)

    def _recursively_gather_node_type(self, node, node_type) -> Dict[str, "NodeBase"]:

        instances = collections.OrderedDict()

        for child in node._get_children():

            # log.debug(f"on child {child._name}")

            if isinstance(child, node_type):

                path = child._get_path()

                # log.debug(f"on child {path}")

                instances[path] = child

                for sub_child in child._get_children():

                    instances.update(
                        self._recursively_gather_node_type(sub_child, node_type)
                    )

            else:

                instances.update(self._recursively_gather_node_type(child, node_type))

        return instances

    def _get_parent(self) -> "NodeBase":
        return self._parent

    def _get_path(self) -> "str":
        """
        returns the str path of this node
        """
        if self._parent is not None:
            return self._path

        else:
            return self._name

    def _root(self, source_only: bool = False) -> "NodeBase":
        """
        returns the root of the node, will stop at the source
        if source_only is set to True
        """
        if self.is_root:
            return self

        else:

            current_node = self

            # recursively walk up the tree to
            # the root

            while True:

                parent = current_node._parent

                if source_only:

                    if parent.name == "__root__":

                        return current_node

                current_node = current_node._parent

                if current_node.is_root:

                    return current_node

    @property
    def path(self) -> str:
        return self._get_path()

    def _update_child_path(self) -> None:
        """

        Update the path of all children recursively.
        This is needed if the name is changed

        :returns:

        """
        # recursively update the path

        for name, child in self._children.items():

            child._path = f"{child._parent._get_path()}.{child._name}"

            if not child.is_leaf:

                child._update_child_path()

    def _change_name(self, name: str, clear_parent: bool = False) -> None:
        """
        change the name of this node. This will have to update the
        children about the change. if clear_parent is provided, then
        the parent is removed
        """
        self._name = name

        if (self._parent is not None) and (not clear_parent):

            self._set_parent(self._parent)

        # update all the children
        self._update_child_path()

    @property
    def is_root(self) -> bool:
        """
        is this the root of the tree
        """
        return self._parent is None

    @property
    def is_leaf(self) -> bool:
        """
        is this a a leaf of the tree
        """
        if len(self._children) == 0:
            return True

        else:

            return False

    @property
    def name(self) -> str:
        return self._name

    def __getattr__(self, name):

        if name in self._children:

            return self._children[name]

        else:

            # log.error(f"Accessing an element {name} of the node that does not exist")

            raise AttributeError(
                f"Accessing an element {name} of the node that does not exist"
            )

            # return super(NodeBase).__getattr__(name)

    def __setattr__(self, name, value):

        ### We cannot change a node
        ### but if the node has a value
        ### attribute, we want to call that

        if "_children" in self.__dict__:
            if name in self._children:

                if "_internal_value" in self._children[name].__dict__:

                    if not self._children[name].is_leaf:

                        log.warning(
                            f"Trying to set the value of a linked parameter ({name}) directly has no effect "
                        )

                        return

                    else:
                        # ok, this is likely  parameter

                        self._children[name].value = value

                else:

                    # this is going to be a node which
                    # we are not allowed to erase

                    # log.error(f"Accessing an element {name} of the node that does not exist")

                    raise AttributeError(
                        f"Accessing an element {name} of the node that does not exist"
                    )
            else:
                return super().__setattr__(name, value)
        else:

            return super().__setattr__(name, value)

    def plot_tree(self) -> Tree:
        """
        this plots the tree to the
        screen
        """

        try:

            out = self.to_dict_with_types()
            name = "model"

        except AttributeError:

            out = self.to_dict()
            name = self.name

        tree = Tree(
            name,
            guide_style="bold medium_orchid",
            style="bold medium_orchid",
            highlight=True,
        )

        _recurse_dict(out, tree)

        return tree


def _recurse_dict(d: Dict[str, Any], tree: Tree, branch_color: Optional[str] = None):

    for k, v in d.items():

        if isinstance(v, collections.OrderedDict):

            if branch_color is not None:

                color = branch_color

            else:

                color = "not bold not blink medium_spring_green"

            if "position" in k:

                k = f"🔭 {k}"
                color = "bold not blink medium_spring_green"

            if "(point source)" in k:

                k = k.replace("(point source)", "")

                color = "bold blink medium_orchid"

                k = f"✨ {k}"

            if "(extended source)" in k:

                k = k.replace("(extended source)", "")

                color = "bold blink medium_orchid"

                k = f"🌌 {k}"

            if "spectrum" in k:

                color = "bold not blink light_goldenrod1"

                k = f"🌈 {k}"

                branch_color = "not bold not blink light_goldenrod1"

            if "main" in k:

                branch_color = "not bold not blink turquoise2"

            branch = tree.add(k, guide_style="bold not blink grey74", style=color)

            _recurse_dict(v, branch, branch_color=branch_color)

        else:

            pass

    return
