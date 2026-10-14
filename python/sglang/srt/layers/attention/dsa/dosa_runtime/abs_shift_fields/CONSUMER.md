Field contract: abs-shift-dd2-v1. Preserve original query sorting/plan epoch,
physical key mapping, starts/ends/tau and H32 source. This is original H16.

prepare(FineU8[Q,16,128],ScaleF32[Q,16],MetaF32[Q,64],RawF32[Q],OutF32[Q],
        HintI32[Q],BudgetI32[Q/4],NewFine,NewScale,NewMeta,NewRaw,NewOut,
        ValidI32[Q],DiagnosticsF64[Q,16])

Outputs are same-layout contiguous tensors; all16 codes/scales and64 metadata
fields are written and paid each call. NewFine[:,2:] equals original paid14
bytes. NewFine[:,0:2] and NewScale[:,0:2] describe V. NewMeta differs only at
C column16; original paid weights at2:16 remain exact. Raw/Out are increased.
Valid accepts exactly2. An original Trusted path must not override a failed
new flag. The output set needs (2048+64+256+4+4+4)=2380B/query plus128B/query
optional diagnostics; no per-key metadata or original-dot tensor is input.

Native hot change: same original four signed RN FMA chains, but use abs(raw)
instead of RN(raw+abs(raw)). Keep original joins and final*.5. The two new DD
raws are evaluated by original scale-mul/FMA RN to Vhat; subtract Vhat directly,
without fmax. This conservatively linearizes -ReLU(originalDD), so a full
negative-DD distribution can loosen survivors even when this point does not.
Use the new RP tail in PROOF.md. Original H32 scoring/compaction/selection/
carry/recovery must remain and be paid. No core-speed claim follows from the
standalone Triton diagnostic evaluator.

Field preparation and complete packing are current-call costs. Query-dependent
fields must not be treated as free cached inputs.
