# Proposal

## Why

The library can apply individual flations but cannot answer the natural next question: is a
bigraph flation-equivalent to a solid bigraph with no dotted edges, and if so, what is the
witness? The investigation in `tmp/flations-investigation.md` shows that the answer is settled for
positive semidefinite forms but remains open for indefinite forms. A decision API must therefore
return a real sequence, a proof-backed impossibility certificate, or an explicit inconclusive
result instead of guessing.

## What Changes

- Add `Bigraph.solidify(...)` returning an immutable `SolidificationResult` with status
  `SOLIDIFIED`, `IMPOSSIBLE`, or `INCONCLUSIVE`.
- Return a verifiable `FlationSequence` and final solid `Bigraph` when a sequence is found.
- Return a verifiable `NoSolidCertificate` for connected positive semidefinite components with
  corank at least two. This is the rigorous impossibility branch: connected PSD Cartan matrices
  have corank at most one, and flation preserves inertia and connectedness.
- For indefinite inputs, run an exact bounded search and return `INCONCLUSIVE` when the configured
  state or entry bounds are exhausted. The open case is never reported as impossible.
- Deduplicate search states exactly, report frame-isomorphism classes, and keep sign-inverted states
  separate where they can affect the next move. The search accepts sign inversions and flations
  over dotted edges, processes disconnected components independently, and uses exact Python
  integers so intermediate entries cannot silently overflow.
- Add public `SolidificationStatus`, `NoSolidCertificate`, and `SolidificationResult` values with
  Markdown/notebook rendering and verification methods, and export them from `dynkinlab`.
- Document the decision boundary, the search caps, and the indefinite caveat in the README,
  reference docs, and changelog.

## Capabilities

### New Capabilities

- `bigraph-solidification`: the solidify-or-certify API, result and certificate values, exact
  bounded search behavior, frame/sign deduplication, componentwise composition, and public
  exports.

### Modified Capabilities

None. Existing flation, inertia, classification, and structural behavior is unchanged.

## Impact

- `src/dynkinlab/solidification.py` (new), `src/dynkinlab/bigraph.py`, and
  `src/dynkinlab/__init__.py`.
- Public API and `dynkinlab.__all__`; `tests/test_package.py` must be updated verbatim.
- New tests for finite, affine, PSD corank at least two, indefinite-inconclusive, disconnected,
  label-preserving, and frame-deduplicated cases.
- README and Sphinx reference documentation, plus `CHANGELOG.md` under `[Unreleased]`.
- No new runtime dependencies. The dev container installs the OpenSpec CLI so the planning workflow
  is available outside `npx`.
