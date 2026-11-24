from unittest import TestCase, main

from lark.indenter import PythonIndenter, Indenter, DedentError
from lark.lexer import Token


def T(type, value):
    return Token(type, value)


def _types(tokens):
    return [(token.type, str(token)) for token in tokens]


class RecordingIndenter(PythonIndenter):
    """Records the depth/indentation stack seen by handle_NL for its own stream."""

    def __init__(self):
        super().__init__()
        self.seen = []

    def handle_NL(self, token):
        self.seen.append((self.paren_level, list(self.indent_level)))
        return super().handle_NL(token)


class TestIndenter(TestCase):
    def _streams(self):
        # A uses parentheses and an indented block.
        def stream_a():
            yield T('NAME', 'a')
            yield T('LPAR', '(')
            yield T('_NEWLINE', '\n')            # swallowed: inside parens
            yield T('NAME', 'b')
            yield T('RPAR', ')')
            yield T('_NEWLINE', '\n')
            yield T('NAME', 'c')
            yield T('_NEWLINE', '\n    ')        # NEWLINE + INDENT
            yield T('NAME', 'd')
            yield T('_NEWLINE', '\n')            # NEWLINE + DEDENT

        # B opens a parenthesis and stays paused inside it.
        def stream_b_open_paren():
            yield T('NAME', 'g')
            yield T('LPAR', '(')
            yield T('NAME', 'h')
            yield T('RPAR', ')')                 # never pulled

        # B enters an indented block and stays paused inside it.
        def stream_b_indent():
            yield T('NAME', 'g')
            yield T('_NEWLINE', '\n        ')    # NEWLINE + INDENT(8)
            yield T('NAME', 'h')
            yield T('NAME', 'i')                 # never pulled

        # B contains an illegal dedent and raises while being processed.
        def stream_b_error():
            yield T('NAME', 'g')
            yield T('_NEWLINE', '\n    ')        # NEWLINE + INDENT(4)
            yield T('NAME', 'h')
            yield T('_NEWLINE', '\n  ')          # DedentError: column 2
            yield T('NAME', 'never_pulled')

        return stream_a, stream_b_open_paren, stream_b_indent, stream_b_error

    def test_baseline_separate_runs(self):
        stream_a, _, _, _ = self._streams()
        expected = [
            ('NAME', 'a'), ('LPAR', '('), ('NAME', 'b'), ('RPAR', ')'),
            ('_NEWLINE', '\n'),
            ('NAME', 'c'), ('_NEWLINE', '\n    '), ('_INDENT', '    '),
            ('NAME', 'd'), ('_NEWLINE', '\n'), ('_DEDENT', ''),
        ]
        self.assertEqual(_types(PythonIndenter().process(stream_a())), expected)

    def _interleave(self, make_b):
        """Prime both streams, then alternate pulling tokens from each."""
        stream_a, _, _, _ = self._streams()
        indenter = PythonIndenter()
        a = indenter.process(stream_a())
        b = indenter.process(make_b())

        result_a, result_b = [], []
        while True:
            try:
                result_a.append(next(a))
            except StopIteration:
                break
            try:
                result_b.append(next(b))
            except StopIteration:
                pass
        return indenter, result_a, result_b

    def test_interleaved_streams_match_separate_runs(self):
        stream_a, open_paren, _, _ = self._streams()
        expected = _types(PythonIndenter().process(stream_a()))
        _, result_a, _ = self._interleave(open_paren)
        self.assertEqual(_types(result_a), expected)

    def test_paused_open_paren_does_not_leak(self):
        stream_a, open_paren, _, _ = self._streams()
        expected = _types(PythonIndenter().process(stream_a()))

        indenter = PythonIndenter()
        a = indenter.process(stream_a())
        b = indenter.process(open_paren())

        first = next(a)                                   # 'a'
        next(b); next(b); next(b)                         # g, (, h -> B is at paren depth 1
        self.assertEqual(indenter.paren_level, 0)         # A's view of self must stay clean
        result = [first] + list(a)
        self.assertEqual(_types(result), expected)

    def test_paused_indentation_does_not_leak(self):
        stream_a, _, indent, _ = self._streams()
        expected = _types(PythonIndenter().process(stream_a()))

        indenter = PythonIndenter()
        a = indenter.process(stream_a())
        b = indenter.process(indent())

        first = next(a)
        next(b); next(b); next(b)                         # g, NEWLINE+INDENT(8), h
        self.assertEqual(indenter.indent_level, [0])      # B's block must not show up for A
        result = [first] + list(a)
        self.assertEqual(_types(result), expected)

    def test_error_in_one_stream_does_not_affect_the_other(self):
        stream_a, _, _, error = self._streams()
        expected = _types(PythonIndenter().process(stream_a()))

        indenter = PythonIndenter()
        a = indenter.process(stream_a())
        b = indenter.process(error())

        first = next(a)
        with self.assertRaises(DedentError):
            list(b)
        self.assertEqual(indenter.paren_level, 0)
        self.assertEqual(indenter.indent_level, [0])
        result = [first] + list(a)
        self.assertEqual(_types(result), expected)

    def test_early_closed_stream_does_not_affect_the_other(self):
        stream_a, open_paren, _, _ = self._streams()
        expected = _types(PythonIndenter().process(stream_a()))

        indenter = PythonIndenter()
        a = indenter.process(stream_a())
        b = indenter.process(open_paren())

        first = next(a)
        next(b); next(b); next(b)                         # g, (, h -> paused inside parens
        b.close()                                         # abandoned mid-stream
        self.assertEqual(indenter.paren_level, 0)
        result = [first] + list(a)
        self.assertEqual(_types(result), expected)

    def test_subclass_handle_nl_sees_its_own_stream_state(self):
        stream_a, open_paren, _, _ = self._streams()

        indenter = RecordingIndenter()
        a = indenter.process(stream_a())
        b = indenter.process(open_paren())

        next(a)                                           # 'a'
        next(b); next(b); next(b)                         # B paused inside its paren
        list(a)

        # Stream A: the first NL is inside parens (depth 1), all later ones at
        # depth 0, starting from the empty indentation stack of stream A.
        self.assertEqual(indenter.seen[0], (1, [0]))
        for paren_level, indent_level in indenter.seen[1:]:
            self.assertEqual(paren_level, 0)
            self.assertEqual(indent_level[0], 0)

    def test_instance_state_restored_between_pulls(self):
        # While no generator is being driven, the instance attributes must hold
        # the initial, neutral state.
        indenter = PythonIndenter()
        g1 = indenter.process(iter([T('LPAR', '('), T('NAME', 'x'), T('RPAR', ')')]))
        g2 = indenter.process(iter([T('NAME', 'y')]))
        next(g1)                                          # '('
        next(g2)                                          # 'y'
        self.assertEqual(indenter.paren_level, 0)
        self.assertEqual(indenter.indent_level, [0])
        self.assertEqual(_types(g1), [('NAME', 'x'), ('RPAR', ')')])

    def test_token_positions_are_preserved(self):
        nl = Token('_NEWLINE', '\n    ', 3, 2, 1, 2, 5, 7)
        indenter = PythonIndenter()
        out = list(indenter.process(iter([T('NAME', 'a'), nl])))
        indent_token = next(t for t in out if t.type == '_INDENT')
        for attr in ('start_pos', 'line', 'column', 'end_line', 'end_column', 'end_pos'):
            self.assertEqual(getattr(indent_token, attr), getattr(nl, attr))

    def test_unmatched_close_paren_still_asserts(self):
        indenter = PythonIndenter()
        with self.assertRaises(AssertionError):
            list(indenter.process(iter([T('RPAR', ')')])))


if __name__ == '__main__':
    main()
