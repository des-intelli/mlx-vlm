# DeepSeek V4.1 block verification (experimental)

The native cache supports `start_speculation(width, block=True)` to verify an entire
DSpark proposal block in one target forward. The default transaction still verifies
one token at a time. This is a low-level opt-in primitive; the CLI, generic
speculative generator and serving scheduler do not enable it automatically.

Small distributed verification blocks also reuse the on-device route dispatch
used by one-token decode (up to 128 routed entries). This avoids the sorted
route path's host readback at every MoE layer. Ordinary prefill and the default
transaction retain their existing dispatch selection.

After partial acceptance, block mode commits projected ring values, compressor
partial groups, completed compressed/index pools, Engram history and the final
retained query's cross-layer handoffs. It does not rerun target weights. Full
acceptance keeps the resulting state; abort or zero retention restores the initial
snapshot. Journal size grows with verification width rather than recording a full
context snapshot for every candidate.

```python
# cache is an already-prefilled DeepseekV41Cache, not BatchDeepseekV41Cache.
# verification_ids contains the current anchor followed by proposed tokens.
width = verification_ids.shape[1]
generation = cache.start_speculation(width, block=True)
try:
    result = model(verification_ids, cache=[cache])
    mx.eval(result.logits)
    target = mx.argmax(result.logits[0], axis=-1).tolist()
    proposed = verification_ids[0, 1:].tolist()
    accepted = 0
    for candidate, expected in zip(proposed, target):
        if candidate != expected:
            break
        accepted += 1
    # The anchor occupies a cache position. The correction/bonus token is not
    # cached yet and becomes the next round's anchor.
    cache.commit_speculation([accepted + 1], generation)
    emitted = proposed[:accepted] + [target[accepted]]
finally:
    cache.abort_speculation(generation)
```

A transaction requires exactly one forward of its declared width. It rejects an
empty cache, incomplete/multiple forwards, stale generation handles and acceptance
outside the declared range. Multiple native rows must retain the same number of
positions; ragged or independently accepted requests continue using the existing
sequential batch-cache protocol. Draft cache management and target hidden-state
selection remain the caller's responsibility.

Batch and one-token kernels can produce different rounding, including in greedy
selection. Validate both cache transitions and output parity before selecting a
verification width. A fixed width can help code generation and hurt less
predictable text; this primitive does not implement confidence-based width
selection or claim a universal serving throughput gain.

Regression tests cover every retained length of a six-position block, compression
group boundaries, ring wraparound, masked Engram history, candidate-mask reuse,
equal-length multirow commits, the following decode token, abort and stale handles.
They also assert that verification calls each target attention layer once and
commit calls it zero times.
