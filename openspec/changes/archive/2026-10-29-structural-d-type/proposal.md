# Proposal

## Why

The A-type structural theorem is executable and certifiable, but the D-type side is only available
through the exact classifier. The thesis characterizes D by two independent constructions: D-core
gluing and A-cyclization. Neither construction can be built or verified with the current public API,
so a paper-like notebook cannot show both D certificates without reimplementing graph operations
locally.

## What Changes

- Add `EdgeMultiset`, a labeled multiset of undirected solid and dotted edges. It SHALL preserve
  opposite parallel edges until an explicit simplification cancels them.
- Add `Bigraph.sum(*pieces, labels=None)`, which identifies shared labels, adds edge weights, and
  simplifies the resulting edge multiset.
- Add `Bigraph.identify(source, target)`, which merges two labels, drops resulting loops, and
  simplifies opposite parallel edges.
- Add `DCore`, a cyclic core of length at least two that also represents the two-edge double link
  used by the D_n construction. It SHALL expose the cycle condition as an odd number of dotted core
  edges.
- Add `DCore.gluing(pieces, labels=None)` and `DCoreGluingCertificate`. The certificate SHALL verify
  that the core and A-type pieces satisfy the thesis's D-core gluing hypotheses and that their sum is
  the claimed D_n bigraph.
- Add `Bigraph.a_cyclization(source, target)` and `ACyclizationCertificate`. The certificate SHALL
  verify that the source is a connected A-type bigraph with two internal vertices joined by a unique
  shortest path of length at least two and odd dotted parity, and that identifying those vertices
  produces the claimed D_{n-1} bigraph.
- Export `EdgeMultiset`, `DCore`, `DCoreGluingCertificate`, and `ACyclizationCertificate` from the
  top-level package and add them to `__all__`.
- Document the new structural values and add a changelog entry.

The existing exact classifier and its classification behavior SHALL NOT change. No automatic
D-type recognition is added; the certificates certify constructions supplied by the caller.

## Capabilities

### New Capabilities

None.

### Modified Capabilities

- `bigraph-structure`: add edge-multiset arithmetic, bigraph sum and vertex identification, D-core
  values, D-core gluing certificates, and A-cyclization certificates.

## Impact

- New modules for edge multisets and D structural values.
- New `Bigraph.sum`, `Bigraph.identify`, and `Bigraph.a_cyclization` methods.
- Public surface grows by four names; the exact `__all__` assertion and package import tests must be
  updated.
- New tests for D13, D4, A9 to D8, cancellation, invalid cores, invalid pieces, and tampered
  certificates.
- API reference and changelog updates.
- No dependency changes, no classifier changes, and no notebook changes in this slice.
