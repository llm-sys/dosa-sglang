# Direct construction of the ABS support vector

Let P be the paid heads and S a subset of the omitted negative-weight heads. Admission requires every positive-weight head to be in P. For real x_h = q_h dot k, the exact score obeys

    sum_h w_h ReLU(x_h) <= sum_{h in P} w_h ReLU(x_h) + sum_{h in S} w_h x_h
                         = 0.5 sum_{h in P} w_h abs(x_h) - V dot k,
    V = -sum_{h in S} w_h q_h - 0.5 sum_{h in P} w_h q_h.

The first inequality uses w ReLU(x) <= w x for negative w in S and w ReLU(x) <= 0 for other omitted heads. The identity applies equally to positive and negative paid weights. P and S remain the original minority planner's input-derived choices; this change does not substitute a fixed layer pattern.

Construct V directly, then encode it once as the two scaled E4 vectors consumed by the existing ABS H16 kernel. The previous path first encoded the negative support vector, decoded that approximation, subtracted the paid sum, and encoded again. Direct construction removes that intermediate encoding error and the second field-preparation launch. Every actual plan epoch still constructs its fields; no query, field, or threshold contents are cached.

The native branch encloses V component sums with F64 directed rounding. Its residual enclosure includes the direct encoding error, screened-dot/RN errors, the official target error, and the original absolute floors; the RMS-key Cauchy bound is rounded outward. DA_SCREEN, DA_TARGET, DA_G64, DA_G18, DA_INFLATE and DA_GT retain the measured direct planner constants. This is not permission to shrink the H32 arithmetic error terms. See PROOF.md for the common score/ABS and SM100 arithmetic assumptions, and native/few/planner.cu for the interval construction.

Paid14 and paid16 use these direct fields; the original planner module and original arithmetic/flags serve paid20. One owner-local BudgetPlanner is installed before any call. There is no preliminary in-place transform or post-plan callback. Ordinary/live H16 require abs-shift-dd2-v1, and both H18 consumers require abs-shift-dd2-h18-v1. Invalid capture, nonfinite or out-of-domain weights/scales, invalid selection or nonfinite bounds leave hint unqualified; existing strict comparison, ties, nonfinite retention and genuine recovery remain.

The direct and original variants share one planner source. A compile-only macro selects the direct field construction; the two native modules and the Python dispatcher are separately reported by workspace_report()["field_preparation"]. No compiled research path is required by the default builder.
