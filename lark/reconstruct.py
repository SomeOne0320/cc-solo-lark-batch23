"""This is an experimental tool for reconstructing text from a shaped tree, based on a Lark grammar.
"""

from typing import Any, Dict, Callable, Iterable, Optional

from .lark import Lark
from .tree import Tree, ParseTree
from .visitors import Transformer_InPlace
from .lexer import Token, PatternStr, TerminalDef
from .grammar import Terminal, Symbol

from .exceptions import ReconstructionError, UnexpectedInput
from .tree_matcher import TreeMatcher, is_discarded_terminal, parse_rulename
from .utils import TextSlice, is_id_continue

def is_iter_empty(i):
    try:
        _ = next(i)
        return False
    except StopIteration:
        return True


def _trees_equal(a, b):
    """Structural equality that compares Token values instead of their identity.

    ``Token.__eq__`` inherits from ``str``, so it compares against the string the
    token was created with. A token mutated in place (``token.value = ...``) is
    therefore still equal to its original text, which would hide changed values
    when the reconstructed source is parsed back and compared.
    """
    if isinstance(a, Tree) or isinstance(b, Tree):
        if not (isinstance(a, Tree) and isinstance(b, Tree)):
            return False
        return (a.data == b.data and len(a.children) == len(b.children)
                and all(_trees_equal(x, y) for x, y in zip(a.children, b.children)))

    if isinstance(a, Token) or isinstance(b, Token):
        if not (isinstance(a, Token) and isinstance(b, Token)):
            return False
        return a.type == b.type and a.value == b.value

    return type(a) is type(b) and a == b


class WriteTokensTransformer(Transformer_InPlace):
    "Inserts discarded tokens into their correct place, according to the rules of grammar"

    tokens: Dict[str, TerminalDef]
    term_subs: Dict[str, Callable[[Symbol], str]]

    def __init__(self, tokens: Dict[str, TerminalDef], term_subs: Dict[str, Callable[[Symbol], str]]) -> None:
        self.tokens = tokens
        self.term_subs = term_subs

    def __default__(self, data, children, meta):
        if not getattr(meta, 'match_tree', False):
            return Tree(data, children)

        iter_args = iter(children)
        to_write = []
        for sym in meta.orig_expansion:
            if is_discarded_terminal(sym):
                try:
                    v = self.term_subs[sym.name](sym)
                except KeyError:
                    t = self.tokens[sym.name]
                    if not isinstance(t.pattern, PatternStr):
                        raise NotImplementedError("Reconstructing regexps not supported yet: %s" % t)

                    v = t.pattern.value
                to_write.append(v)
            else:
                x = next(iter_args)
                if isinstance(x, list):
                    to_write += x
                else:
                    if isinstance(x, Token):
                        assert Terminal(x.type) == sym, x
                    else:
                        name, _args = parse_rulename(sym.name)
                        assert x.data == name, (sym, x)
                        x.meta.orig_rulename = sym.name
                    to_write.append(x)

        assert is_iter_empty(iter_args)
        return to_write


class Reconstructor(TreeMatcher):
    """
    A Reconstructor that will, given a full parse Tree, generate source code.

    Note:
        The reconstructor cannot generate values from regexps. If you need to produce discarded
        regexes, such as newlines, use `term_subs` and provide default values for them.

    Parameters:
        parser: a Lark instance
        term_subs: a dictionary of [Terminal name as str] to [output text as str]

    Source-preserving reconstruction:
        Passing both ``source`` and ``original_tree`` makes the reconstructor edit
        the original text instead of generating a new one. Each token in ``tree``
        is matched against the corresponding token in ``original_tree``; only
        tokens whose value changed are replaced, using the positions recorded in
        ``original_tree``. All other characters, including whitespace and
        comments, are preserved byte for byte.

        The two trees must have the same shape (added or removed nodes, or
        changed rule/token types, are rejected), the source must match the
        original token positions, and the result is parsed again and compared to
        ``tree``.
    """

    write_tokens: WriteTokensTransformer

    def __init__(self, parser: Lark, term_subs: Optional[Dict[str, Callable[[Symbol], str]]]=None) -> None:
        TreeMatcher.__init__(self, parser)

        self.write_tokens = WriteTokensTransformer({t.name:t for t in self.tokens}, term_subs or {})

    def _reconstruct(self, tree, orig_rulename=None):
        unreduced_tree = self.match_tree(tree, orig_rulename or tree.data)

        res = self.write_tokens.transform(unreduced_tree)
        for item in res:
            if isinstance(item, Tree):
                child_rulename = getattr(item.meta, "orig_rulename", None)
                yield from self._reconstruct(item, child_rulename)
            else:
                yield item

    def reconstruct(self, tree: ParseTree, postproc: Optional[Callable[[Iterable[str]], Iterable[str]]]=None, insert_spaces: bool=True,
                    source: Any=None, original_tree: Optional[ParseTree]=None) -> Any:
        """Reconstruct source text from a parse tree.

        Parameters:
            tree: The target parse tree.
            postproc: Legacy standalone reconstruction: a callable to postprocess
                the generated items. Not used when ``source`` is given.
            insert_spaces: Legacy standalone reconstruction: whether to insert
                spaces between adjacent identifiers. Not used when ``source`` is given.
            source: The original source text ``original_tree`` was parsed from.
                When given, ``original_tree`` must be given as well. May be a
                ``str``, ``bytes``, or ``TextSlice`` matching the parse input.
            original_tree: A parse tree of ``source``, captured before ``tree``
                was modified. It provides the positions of the tokens to replace.

        Returns:
            The reconstructed text. With ``source`` and ``original_tree``, this is
            ``source`` with only the changed token values replaced.

        Raises:
            ReconstructionError: If the trees don't match structurally, a token
                has no source position, the source doesn't match the original
                tree, or the result doesn't parse back into ``tree``.
        """
        if source is None and original_tree is None:
            x = self._reconstruct(tree)
            if postproc:
                x = postproc(x)
            y = []
            prev_item = ''
            for item in x:
                if insert_spaces and prev_item and item and is_id_continue(prev_item[-1]) and is_id_continue(item[0]):
                    y.append(' ')
                y.append(item)
                prev_item = item
            return ''.join(y)

        if source is None or original_tree is None:
            raise ReconstructionError("Both 'source' and 'original_tree' must be provided for source-preserving reconstruction.")
        if postproc is not None:
            raise ReconstructionError("'postproc' cannot be used with source-preserving reconstruction.")
        if getattr(self.parser.options, 'transformer', None) is not None:
            raise ReconstructionError("Source-preserving reconstruction requires a parser without a transformer, "
                                      "because the result must be parsed back into a tree for verification.")

        return self._reconstruct_from_source(tree, source, original_tree)

    def _reconstruct_from_source(self, tree: ParseTree, source: Any, original_tree: ParseTree) -> Any:
        if not isinstance(tree, Tree) or not isinstance(original_tree, Tree):
            raise ReconstructionError("Source-preserving reconstruction requires parse trees at the root.")

        if isinstance(source, TextSlice):
            # Positions are absolute offsets into the full text, so work on that
            # text and only cut down to the slice at the very end.
            text = source.text
            offset, limit = source.start, source.end
        else:
            text = source
            offset, limit = 0, len(source)

        if isinstance(text, bytes):
            # Token values are bytes too when parsing with use_bytes=True. Guard
            # against a str source paired with bytes tokens (and vice versa).
            def as_source_type(s):
                return s.encode('utf-8') if isinstance(s, str) else s

            empty: Any = b''
            token_values = (t.value for t in original_tree.scan_values(lambda x: isinstance(x, Token)))
            if any(isinstance(v, str) for v in token_values):
                raise ReconstructionError("'source' is bytes, but the original tree contains str token values.")
        else:
            def as_source_type(s):
                return s

            empty = ''
            token_values = (t.value for t in original_tree.scan_values(lambda x: isinstance(x, Token)))
            if any(isinstance(v, bytes) for v in token_values):
                raise ReconstructionError("'source' is str, but the original tree contains bytes token values.")

        edits = []
        self._collect_source_edits(original_tree, tree, text, as_source_type, 'tree', edits)

        # Sort into source order and make sure the edits are applicable. The edits
        # are applied against the original offsets, so a changed length can never
        # shift a later edit.
        edits.sort(key=lambda e: (e[0], e[1]))
        unique_edits = []
        prev = None
        for edit in edits:
            start, end, value, path = edit
            if prev is not None and (start, end, value) == prev:
                # The same token object can appear more than once in a tree.
                continue
            if not (offset <= start <= end <= limit):
                raise ReconstructionError("Token at %s lies outside the given source range." % path)
            if prev is not None and start < prev[1]:
                raise ReconstructionError("Token positions overlap at %s; cannot reconstruct source." % path)
            unique_edits.append(edit)
            prev = (start, end, value)

        parts = []
        pos = offset
        for start, end, value, _path in unique_edits:
            parts.append(text[pos:start])
            parts.append(as_source_type(value))
            pos = end
        parts.append(text[pos:limit])
        new_source = empty.join(parts)

        self._verify_reconstruction(new_source, tree)
        return new_source

    def _collect_source_edits(self, original, target, source, as_source_type, path, edits):
        "Collect (start, end, new_value, path) for every token whose value changed."
        if isinstance(original, Tree):
            if not isinstance(target, Tree):
                raise ReconstructionError("Tree mismatch at %s: expected a tree node %r, got %r."
                                          % (path, original.data, target))
            if original.data != target.data:
                raise ReconstructionError("Rule mismatch at %s: original tree has %r, target tree has %r."
                                          % (path, original.data, target.data))
            if len(original.children) != len(target.children):
                raise ReconstructionError(
                    "Structure mismatch at %s (%r): original tree has %d children, target tree has %d. "
                    "Source-preserving reconstruction cannot add or remove nodes."
                    % (path, original.data, len(original.children), len(target.children)))

            for i, (original_child, target_child) in enumerate(zip(original.children, target.children)):
                self._collect_source_edits(original_child, target_child, source, as_source_type,
                                           '%s.children[%d]' % (path, i), edits)

            return

        if isinstance(original, Token):
            if not isinstance(target, Token):
                raise ReconstructionError("Token mismatch at %s: original tree has token %r, target tree has %r."
                                          % (path, original, target))
            if original.type != target.type:
                raise ReconstructionError("Token type mismatch at %s: original tree has %r (%r), target tree has %r (%r)."
                                          % (path, original.type, original, target.type, target))

            start = original.start_pos
            end = original.end_pos
            if start is None or end is None:
                raise ReconstructionError(
                    "Token %r at %s has no source position. Parse the original source with position tracking enabled."
                    % (original, path))
            if start < 0 or end < start or end > len(source):
                raise ReconstructionError("Token %r at %s has invalid source positions (%r, %r)."
                                          % (original, path, start, end))

            in_source = source[start:end]
            if in_source != as_source_type(original.value):
                raise ReconstructionError(
                    "Source doesn't match the original tree at %s: positions (%d, %d) contain %r, "
                    "but the original token is %r. If this token was generated by a postlexer "
                    "(e.g. _INDENT/_DEDENT), parse without keep_all_tokens so it isn't kept in the tree."
                    % (path, start, end, in_source, original.value))

            if target.value != original.value:
                edits.append((start, end, target.value, path))
            return

        # Anything else is a non-token leaf (e.g. a placeholder). They can't be
        # edited, so they must be equal.
        if type(original) is not type(target) or original != target:
            raise ReconstructionError("Value mismatch at %s: original tree has %r, target tree has %r."
                                      % (path, original, target))

    def _verify_reconstruction(self, new_source, target_tree):
        "Parse the new source back, and make sure it produces the target tree."
        start_options = list(self.parser.options.start)
        if len(start_options) == 1:
            starts = [None]
        elif target_tree.data in start_options:
            starts = [target_tree.data]
        else:
            starts = start_options

        errors = []
        for start in starts:
            try:
                parsed = self.parser.parse(new_source, start=start)
            except UnexpectedInput as e:
                errors.append((start, e))
                continue

            if _trees_equal(parsed, target_tree):
                return

            errors.append((start, None))
            if len(starts) == 1 or start == target_tree.data:
                raise ReconstructionError(
                    "Reconstructed source doesn't parse back into the target tree.\n"
                    "Expected:\n%s\nGot:\n%s" % (target_tree.pretty(), parsed.pretty()))

        details = '\n'.join('start=%r: %s' % (start or start_options[0], e if e is not None else 'parsed into a different tree')
                            for start, e in errors)
        raise ReconstructionError("Reconstructed source doesn't parse back into the target tree.\n%s" % details)

