# coding=utf-8

import copy
import json
import os
import re
import sys
import tempfile
import unittest
from itertools import product
from unittest import TestCase

from lark import Lark, Token, Tree
from lark.exceptions import ReconstructionError
from lark.reconstruct import Reconstructor

common = """
%import common (WS_INLINE, NUMBER, WORD)
%ignore WS_INLINE
"""


def _remove_ws(s):
    return s.replace(' ', '').replace('\n', '')


class TestReconstructor(TestCase):

    def assert_reconstruct(self, grammar, code, **options):
        parser = Lark(grammar, parser='lalr', maybe_placeholders=False, **options)
        tree = parser.parse(code)
        new = Reconstructor(parser).reconstruct(tree)
        self.assertEqual(_remove_ws(code), _remove_ws(new))

    def test_starred_rule(self):
        g = """
        start: item*
        item: NL
            | rule
        rule: WORD ":" NUMBER
        NL: /(\\r?\\n)+\\s*/
        """ + common

        code = """
        Elephants: 12
        """

        self.assert_reconstruct(g, code)

    def test_starred_group(self):
        g = """
        start: (rule | NL)*
        rule: WORD ":" NUMBER
        NL: /(\\r?\\n)+\\s*/
        """ + common

        code = """
        Elephants: 12
        """

        self.assert_reconstruct(g, code)

    def test_alias(self):
        g = """
        start: line*
        line: NL
            | rule
            | "hello" -> hi
        rule: WORD ":" NUMBER
        NL: /(\\r?\\n)+\\s*/
        """ + common

        code = """
        Elephants: 12
        hello
        """

        self.assert_reconstruct(g, code)

    def test_keep_tokens(self):
        g = """
        start: (NL | stmt)*
        stmt: var op var
        !op: ("+" | "-" | "*" | "/")
        var: WORD
        NL: /(\\r?\\n)+\\s*/
        """ + common

        code = """
        a+b
        """

        self.assert_reconstruct(g, code)

    def test_expand_rule(self):
        g = """
        ?start: (NL | mult_stmt)*
        ?mult_stmt: sum_stmt ["*" sum_stmt]
        ?sum_stmt: var ["+" var]
        var: WORD
        NL: /(\\r?\\n)+\\s*/
        """ + common

        code = ['a', 'a*b', 'a+b', 'a*b+c', 'a+b*c', 'a+b*c+d']

        for c in code:
            self.assert_reconstruct(g, c)

    def test_template_rule(self):
        g = """
        start: (NL | binfunc)*
        binfunc: WORD "(" binarg{WORD} ")"
        binarg{arg}: arg "," arg
        NL: /(\\r?\\n)+\\s*/
        """ + common

        code = """
        func(a, b)
        """

        self.assert_reconstruct(g, code)

    def test_json_example(self):
        test_json = '''
            {
                "empty_object" : {},
                "empty_array"  : [],
                "booleans"     : { "YES" : true, "NO" : false },
                "numbers"      : [ 0, 1, -2, 3.3, 4.4e5, 6.6e-7 ],
                "strings"      : [ "This", [ "And" , "That", "And a \\"b" ] ],
                "nothing"      : null
            }
        '''

        json_grammar = r"""
            ?start: value

            ?value: object
                  | array
                  | string
                  | SIGNED_NUMBER      -> number
                  | "true"             -> true
                  | "false"            -> false
                  | "null"             -> null

            array  : "[" [value ("," value)*] "]"
            object : "{" [pair ("," pair)*] "}"
            pair   : string ":" value

            string : ESCAPED_STRING

            %import common.ESCAPED_STRING
            %import common.SIGNED_NUMBER
            %import common.WS

            %ignore WS
        """

        json_parser = Lark(json_grammar, parser='lalr', maybe_placeholders=False)
        tree = json_parser.parse(test_json)

        new_json = Reconstructor(json_parser).reconstruct(tree)
        self.assertEqual(json.loads(new_json), json.loads(test_json))

    def test_keep_all_tokens(self):
        g = """
        start: "a"? _B? c? _d?
        _B: "b"
        c: "c"
        _d: "d"
        """
        examples = list(map(''.join, product(('', 'a'), ('', 'b'), ('', 'c'), ('', 'd'), )))
        for code in examples:
            self.assert_reconstruct(g, code, keep_all_tokens=True)

    def test_switch_grammar_unicode_terminal(self):
        """
        This test checks that a parse tree built with a grammar containing only ascii characters can be reconstructed
        with a grammar that has unicode rules (or vice versa). The original bug assigned ANON terminals to unicode
        keywords, which offsets the ANON terminal count in the unicode grammar and causes subsequent identical ANON
        tokens (e.g., `+=`) to mismatch between the two grammars.
        """

        g1 = """
        start: (NL | stmt)*
        stmt: "keyword" var op var
        !op: ("+=" | "-=" | "*=" | "/=")
        var: WORD
        NL: /(\\r?\\n)+\\s*/
        """ + common

        g2 = """
        start: (NL | stmt)*
        stmt: "குறிப்பு" var op var
        !op: ("+=" | "-=" | "*=" | "/=")
        var: WORD
        NL: /(\\r?\\n)+\\s*/
        """ + common

        code = """
        keyword x += y
        """

        l1 = Lark(g1, parser='lalr', maybe_placeholders=False)
        l2 = Lark(g2, parser='lalr', maybe_placeholders=False)
        r = Reconstructor(l2)

        tree = l1.parse(code)
        code2 = r.reconstruct(tree)
        assert l2.parse(code2) == tree


def _copy_with_replacements(tree, replacements, counter=None):
    """Deep-copy a tree, replacing token values by traversal index.

    ``replacements`` maps a zero-based token traversal index to its new value.
    """
    if counter is None:
        counter = iter(range(len(list(tree.scan_values(lambda v: isinstance(v, Token))))))

    if isinstance(tree, Tree):
        return Tree(tree.data, [_copy_with_replacements(c, replacements, counter) for c in tree.children])
    if isinstance(tree, Token):
        i = next(counter)
        if i in replacements:
            return tree.update(value=replacements[i])
        return tree
    return copy.deepcopy(tree)


class TestSourcePreservingReconstructor(TestCase):
    """Reconstruct by editing the original source, keeping comments and formatting."""

    grammar = r"""
        start: (stmt | COMMENT)*
        stmt: NAME "=" NAME ";"
        NAME: /[a-z_][a-z0-9_]*/
        COMMENT: /#[^\n]*/
        %ignore /[ \t]+/
        %ignore /\n+/
    """

    source = (
        "# Handover notes: edit only the values, keep this file's layout.\n"
        "# Values are duplicated on purpose; each occurrence is edited separately.\n"
        "\n"
        "alpha   = beta;      # first line, trailing note\n"
        "gamma = alpha;\n"
        "beta = beta;\n"
        "\n"
        "# ------------------------------------------------------------------\n"
        "# Section two: repeated values below\n"
        "delta\t=\tbeta;     # tabs are intentional\n"
        "\n"
        "\n"
        "beta    = gamma;\n"
    )

    def setUp(self):
        self.parser = Lark(self.grammar, parser='lalr', maybe_placeholders=False)
        self.reconstructor = Reconstructor(self.parser)
        self.original = self.parser.parse(self.source)

    def _tokens(self, tree, type=None):
        return list(tree.scan_values(lambda v: isinstance(v, Token) and (type is None or v.type == type)))

    def test_unchanged_source_is_returned_verbatim(self):
        target = copy.deepcopy(self.original)
        self.assertEqual(self.reconstructor.reconstruct(target, source=self.source, original_tree=self.original),
                         self.source)

    def test_only_changed_values_are_replaced(self):
        tokens = self._tokens(self.original)
        name_positions = [i for i, t in enumerate(tokens) if t.type == 'NAME']
        # Replace the first and the last `beta`, with values of different
        # lengths, and leave the middle occurrences alone.
        beta_positions = [i for i in name_positions if tokens[i].value == 'beta']
        self.assertEqual(len(beta_positions), 5)
        replacements = {beta_positions[0]: 'b', beta_positions[-1]: 'beta_renamed_to_something_longer'}
        target = _copy_with_replacements(self.original, replacements)

        new_source = self.reconstructor.reconstruct(target, source=self.source, original_tree=self.original)

        expected = (
            "# Handover notes: edit only the values, keep this file's layout.\n"
            "# Values are duplicated on purpose; each occurrence is edited separately.\n"
            "\n"
            "alpha   = b;      # first line, trailing note\n"
            "gamma = alpha;\n"
            "beta = beta;\n"
            "\n"
            "# ------------------------------------------------------------------\n"
            "# Section two: repeated values below\n"
            "delta\t=\tbeta;     # tabs are intentional\n"
            "\n"
            "\n"
            "beta_renamed_to_something_longer    = gamma;\n"
        )
        self.assertEqual(new_source, expected)
        # The edited text reads back into the target tree.
        self.assertEqual(self.parser.parse(new_source), target)

    def test_length_changes_do_not_shift_later_edits(self):
        # Make every other occurrence much longer/shorter, and check that later
        # replacements still land in the right place.
        tokens = self._tokens(self.original)
        name_positions = [i for i, t in enumerate(tokens) if t.type == 'NAME']
        replacements = {}
        for n, i in enumerate(name_positions):
            if tokens[i].value == 'beta':
                replacements[i] = ('beta' * (n + 1)) if n % 2 else 'b'

        target = _copy_with_replacements(self.original, replacements)
        new_source = self.reconstructor.reconstruct(target, source=self.source, original_tree=self.original)

        expected = self.source
        # Build the expected text by applying replacements from the end, against
        # the *original* offsets.
        for i in sorted(replacements, reverse=True):
            tok = tokens[i]
            expected = expected[:tok.start_pos] + replacements[i] + expected[tok.end_pos:]
        self.assertEqual(new_source, expected)
        self.assertEqual(self.parser.parse(new_source), target)

    def test_same_value_at_several_positions_is_independent(self):
        tokens = self._tokens(self.original)
        indices = [i for i, t in enumerate(tokens) if t.value == 'beta']
        target = _copy_with_replacements(self.original, {indices[1]: 'beta_one', indices[2]: 'beta_two'})
        new_source = self.reconstructor.reconstruct(target, source=self.source, original_tree=self.original)

        self.assertEqual(new_source.count('beta_one'), 1)
        self.assertEqual(new_source.count('beta_two'), 1)
        # Occurrences that weren't edited are untouched.
        self.assertEqual(len(re.findall(r'\bbeta\b', new_source)), len(indices) - 2)
        self.assertEqual(self.parser.parse(new_source), target)

    def test_rejects_missing_original_tree(self):
        with self.assertRaises(ReconstructionError):
            self.reconstructor.reconstruct(self.original, source=self.source)
        with self.assertRaises(ReconstructionError):
            self.reconstructor.reconstruct(self.original, original_tree=self.original)

    def test_rejects_added_nodes(self):
        target = copy.deepcopy(self.original)
        target.children.append(self.parser.parse("x = y;").children[0])
        with self.assertRaisesRegex(ReconstructionError, 'Structure mismatch'):
            self.reconstructor.reconstruct(target, source=self.source, original_tree=self.original)

    def test_rejects_removed_nodes(self):
        target = copy.deepcopy(self.original)
        del target.children[0]
        with self.assertRaisesRegex(ReconstructionError, 'Structure mismatch'):
            self.reconstructor.reconstruct(target, source=self.source, original_tree=self.original)

    def test_rejects_changed_rule(self):
        target = copy.deepcopy(self.original)
        stmt = next(target.find_data('stmt'))
        stmt.data = 'other_rule'
        with self.assertRaisesRegex(ReconstructionError, 'Rule mismatch'):
            self.reconstructor.reconstruct(target, source=self.source, original_tree=self.original)

    def test_rejects_missing_positions(self):
        original = copy.deepcopy(self.original)
        tokens = self._tokens(original)
        tokens[0].start_pos = None
        tokens[0].end_pos = None
        with self.assertRaisesRegex(ReconstructionError, 'no source position'):
            self.reconstructor.reconstruct(copy.deepcopy(self.original), source=self.source, original_tree=original)

    def test_rejects_source_that_doesnt_match_original_tree(self):
        with self.assertRaisesRegex(ReconstructionError, "doesn't match the original tree"):
            self.reconstructor.reconstruct(copy.deepcopy(self.original), source='# shifted\n' + self.source,
                                           original_tree=self.original)

    def test_rejects_output_that_doesnt_parse_into_target(self):
        tokens = self._tokens(self.original)
        name_index = next(i for i, t in enumerate(tokens) if t.type == 'NAME')
        # A value that can't be read back as a NAME. The original tree's position
        # is still valid, but verification catches the bad output.
        target = _copy_with_replacements(self.original, {name_index: 'not a name!'})
        with self.assertRaises(ReconstructionError):
            self.reconstructor.reconstruct(target, source=self.source, original_tree=self.original)

    def test_legacy_reconstruction_is_unchanged(self):
        # The old path still generates text from the target tree, without source.
        generated = self.reconstructor.reconstruct(self.original)
        strip_ws = lambda s: re.sub(r'\s', '', s)
        self.assertEqual(strip_ws(self.source), strip_ws(generated))


class TestSourcePreservingRealFile(TestCase):
    """Save-then-reload a real grammar file, keeping its comments and layout."""

    def test_edit_real_grammar_file_and_reload(self):
        path = os.path.join(os.path.dirname(__file__), '..', 'examples', 'advanced', 'python2.lark')
        with open(path, encoding='utf-8') as f:
            source = f.read()

        # The grammar-of-grammars parses .lark files, giving every RULE token a
        # position in the real file.
        parser = Lark.open_from_package('lark', 'lark.lark', ['grammars'], parser='lalr',
                                        maybe_placeholders=False)
        original = parser.parse(source)
        tokens = list(original.scan_values(lambda v: isinstance(v, Token)))

        # `test` occurs many times in this file. Edit three separate occurrences,
        # including the first and the last, with different value lengths.
        occurrences = [i for i, t in enumerate(tokens) if t.type == 'RULE' and t.value == 'test']
        self.assertGreater(len(occurrences), 10)
        replacements = {
            occurrences[0]: 'production',
            occurrences[3]: 't',
            occurrences[-1]: 'test_with_a_much_longer_name',
        }
        target = _copy_with_replacements(original, replacements)

        new_source = Reconstructor(parser).reconstruct(target, source=source, original_tree=original)

        # Only the chosen occurrences changed.
        expected = source
        for i in sorted(replacements, reverse=True):
            tok = tokens[i]
            expected = expected[:tok.start_pos] + replacements[i] + expected[tok.end_pos:]
        self.assertEqual(new_source, expected)

        # Comments, blank lines, trailing notes, and alignment are untouched.
        self.assertEqual(new_source.count('\n'), source.count('\n'))
        for line in source.splitlines():
            if line.lstrip().startswith('//'):
                self.assertIn(line, new_source)

        # Save and read back. The reloaded file parses into the target tree.
        with tempfile.NamedTemporaryFile('w', suffix='.lark', delete=False, encoding='utf-8') as f:
            f.write(new_source)
            saved_path = f.name
        try:
            with open(saved_path, encoding='utf-8') as f:
                saved = f.read()
            self.assertEqual(saved, new_source)
            self.assertEqual(parser.parse(saved), target)
        finally:
            os.unlink(saved_path)


if __name__ == '__main__':
    unittest.main()
