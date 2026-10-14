# Original16 rows, absolute paid reduction and shifted DD

Let a_h be the original14 paid F32 raw dot results, d the original two-DD
F32 RN result, and w the original signed F32 weights. Define the exact real
quantity v_target=d-.5 sum_h w_h a_h. Then

    sum w ReLU(a) - ReLU(d)
    <= .5 sum w |a| - v_target.

The only structural slack here is max(-d,0). No positive-only selection or
negative-filler removal occurs. Build V=D_eff-.5 sum w q from actual stored
E4 rows/scales/weights; encode it as two stored E4 rows with POT scales.
Those replace the original DD2. All14 paid rows and weight order remain.
There is no second TC, extra row or extra TMEM roundtrip.

Use the original normal binary arithmetic model
|TC(q,k)-q.k|<=D sum_i |q_i k_i|, D=gamma128(2^-23)=1/65535, with original
rho>=||K||/sqrt128. Define M(x)=sqrt128||x||_2 and

    MP=M(sum_h |w_h q_h|), MD=M(|s0 qDD0|+|s1 qDD1|),
    MV=M(|t0 c0|+|t1 c1|).

All coordinate sums/norms are directed F64 upper bounds. The exact V is
formed in a directed F64 interval. F64 w*E4 products are exact; reductions
are enclosed. E is M of a coordinatewise interval covering the difference
between that exact V and the two actually decoded stored E4 rows. It includes
formation, F32 conversion and both quantizations, rather than a sampled error.

Let gD=gamma3(2^-24), gA=gamma64(2^-24), gO=gamma80(2^-24),
gC=gamma4(2^-24), gT=gamma8(2^-24). Set

    Pm=(1+D)MP, Dm=(1+gD)(1+D)MD, Vm=(1+gD)(1+D)MV,
    EV=E + D(.5MP+MD+MV) + gD(1+D)(MD+MV),
    alpha=EV + gO(Pm+Dm) + .5gA Pm
              + gC(.5(1+gA)Pm+Vm),
    Mc=(1+gO)(Pm+Dm).

EV encloses new Vhat minus v_target: it pays original paid/DD raw errors,
new2-row raw errors and original/new two-scale RN evaluation. The actual
new core is RN(RN(.5*four_chain_sum_RN(w*abs(a)))-Vhat), with original chain
order and joins. Its reduction/subtraction error plus the original RN paid/
core error gives old_core<=new_core+alpha*rho+absolute_payment. The counts64/
80/4 are conservative path budgets, not measured fit constants.

To cover the original RN tail as well, store upward F32 fields

    Cnew >= (1+gT)Cold + alpha + gT Mc,
    rawnew >= (1+gT)rawold + 2^-120,
    outnew >= (1+gT)outold + 2^-120.

Original C/raw/out are nonnegative and never reduced. Apply explicit RP
addition/multiplication from new_core through rawnew, rho*Cnew, positive KS,
and outnew. This bounds the old complete H16 U, including its RN tail. The
proof does not currently authorize replacing this new RP tail by RN. With
normal E4 dot intermediates and DD POT scales, the two 2^-120 payments dominate
fewer than128 scalar subnormal-rounding residues (including their propagation
before positive KS). No FTZ is permitted. Nonfinite results fail open.

Admission: current original H16 budget14/hint2, finite actual E4 codes and
weights |w|<=65504; original and new DD scales positive POT in[2^-64,2^16];
C/raw/out finite>=0; original RMS contract; 0<KS<=2^64; finite final value.
New valid==2 must be consumed even if the old emitter was Trusted. Unknown,
failed encoding, equality and nonfinite retain candidates/use original H32
recovery. The finite-domain proof is universal; the137→137 result and zero
negative-d slack are only the retained Q8/threeK1024 development point.
