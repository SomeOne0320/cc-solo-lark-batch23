# Unreleased

- Earley parser: when `propagate_positions` is enabled, explicit ambiguity nodes
  (`_ambig`/`_iambig`, and their `_inter` children) now cover the source text
  occupied by their alternative derivations, instead of carrying no position.
  Parent rules include that range when computing their own, including for
  nested ambiguities and inlined rules. Disabling position propagation, or using
  a custom tree type, keeps the previous interface.

# v1.0

- `maybe_placeholders` is now True by default

- Renamed TraditionalLexer to BasicLexer, and 'standard' lexer option to 'basic'

- Default priority is now 0, for both terminals and rules (used to be 1 for terminals)

- Discard mechanism is now done by returning Discard, instead of raising it as an exception.

- `use_accepts` in `UnexpectedInput.match_examples()` is now True by default

- `v_args(meta=True)` now gives meta as the first argument. i.e. `(meta, children)`
