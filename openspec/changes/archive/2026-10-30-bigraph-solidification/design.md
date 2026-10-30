# Design

## Context

See `proposal.md` - Why. The public flation values already provide replay and verification, and
`Inertia` already distinguishes positive semidefinite, affine, and indefinite forms exactly. The
investigation in `tmp/flations-investigation.md` supplies the mathematical split. A new module can
therefore stay at the bigraph/inertia layer without changing the classifier.

The investigation also found a subtle limit on certificates: the indefinite candidate is
Z-equivalent to a solid matrix, yet no flation path to it has been found. Congruence invariants
alone therefore cannot certify that a flation sequence is impossible.

## Goals / Non-Goals

**Goals:**

- A public decision result with explicit `SOLIDIFIED`, `IMPOSSIBLE`, and `INCONCLUSIVE` branches.
- A verifiable sequence for every `SOLIDIFIED` result and a rigorous certificate for every
  `IMPOSSIBLE` result.
- Deterministic, exact, resumable-by-caps search for the open indefinite case.
- Frame-isomorphism accounting so relabelled states do not hide the size of the search.

**Non-Goals:**

- Proving that every indefinite input solidifies, or producing a no-solid certificate for an
  indefinite input.
- Replacing `DynkinClassifier` or changing classification behavior.
- Non-symmetric quasi-Cartan matrices in this change.

## Decisions

### New `solidification.py` module, lazy import from `Bigraph`

`Bigraph.solidify` imports the new module inside the method, matching the existing lazy-import
pattern that avoids `bigraph.py <-> solidification.py` cycles. `solidification.py` may import
`Bigraph`, `FlationSequence`, `FlationStep`, and `Inertia`, but must not import private helpers from
other modules.

Alternative considered: put the search in `flation.py`. Rejected because `FlationStep` and
`FlationSequence` are value types, and the decision logic, certificate, and bounded search are a
separate capability.

### Three-state result instead of a boolean

`SolidificationStatus` has three values. `IMPOSSIBLE` is reserved for the PSD corank-at-least-two
theorem. `INCONCLUSIVE` carries the search counters and caps and never carries a certificate. This
prevents the API from turning a bounded negative search into a mathematical claim.

Alternative considered: `bool` plus `None`. Rejected because it loses the reason and makes it too
easy to treat an exhausted search as a proof.

### Exact state search, frame accounting, sign states kept separate

The visited set is keyed by exact integer matrix states, with vertex-permutation
canonicalization when the component is small enough for exhaustive canonicalization. Frame keys
are maintained separately for reporting. Sign-inverted states are not collapsed in the exact
visited set because a sign inversion can change which edges are dotted and therefore which flation
moves exist.

For larger components, exact state tuples remain the correctness key; frame classes are verified
with a Weisfeiler-Lehman bucket plus an exact `networkx` isomorphism check. This keeps the frame
count honest without making correctness depend on a hash.

### Search moves

Each component expands, deterministically:

1. a sign inversion at every vertex;
2. a flation over each dotted edge, in both directions.

Solid flations are omitted because the investigation's positive-definite and affine algorithms use
dotted flations plus sign inversions. Omitting them keeps the branching factor smaller. The
`max_states` and `max_entry` caps bound the open indefinite search.

Alternatives considered: include solid flations for completeness. Rejected for the first version;
it multiplies the branching factor and the documented progress criteria use dotted moves.

### Positive definite fast path

For a positive definite component, run the same dotted-edge flation loop without a full frontier:
the finite root-count argument makes every dotted flation progress and the orbit finite. This
mirrors the classifier's bounded symmetric loop but does not import its private helpers. The
`max_states` cap is still checked so a defect cannot loop forever.

### Componentwise composition

`Symmetrizer.components` already returns connected index groups in index order. Each component is
solved separately, and component-local steps are translated to the original labels before being
concatenated. An impossible component dominates; otherwise an inconclusive component dominates an
otherwise successful result.

### Certificate contents

`NoSolidCertificate` stores the offending component labels, its exact `Inertia`, and a reason
string. `verify(bigraph)` re-extracts the induced submatrix, checks connectedness through nonzero
off-diagonal entries, checks the same inertia, and checks `negative == 0` and `zero >= 2`.

### Rendering and verification

`SolidificationResult.verify()` returns true only for a valid sequence or a valid certificate; an
inconclusive result has nothing to verify. Markdown hooks follow the existing display convention and
do not depend on IPython.

## Risks / Trade-offs

- **Search explosion on indefinite inputs** -> the state and entry caps make termination
  guaranteed; the result is explicitly inconclusive. The frame counter exposes how much of the
  orbit was visited.
- **Frame canonicalization cost** -> exact permutation canonicalization is limited to small
  components; larger components use exact states for correctness and bucket-based isomorphism for
  frame accounting.
- **A user mistakes `INCONCLUSIVE` for a negative answer** -> the status is a separate enum value,
  the certificate slot is empty, and the Markdown summary states that the bounds were exhausted.
- **Positive definite inputs with unusual entries** -> the dotted-edge loop is still exact and is
  bounded by `max_states`; the finite root-count theorem guarantees termination, but the default
  cap may still need raising for very large diagrams.
- **Certificate scope** -> only connected PSD corank-at-least-two components can be certified;
  disconnected inputs compose the component certificates.

## Migration Plan

No migration is required. The change only adds public names and a method; existing flation,
classification, and display behavior is unchanged. Package tests must update the verbatim
`dynkinlab.__all__` assertion.
