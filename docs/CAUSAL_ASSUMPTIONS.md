# Causal Assumptions

Uplift modelling relies on stronger assumptions than ordinary prediction.

## Randomized treatment assignment

The cleanest setting is a randomized experiment where treatment and control groups are comparable before treatment.

`validate_randomization` reports a sample-ratio mismatch (SRM) test and
standardized mean differences for observed pre-treatment covariates. It
requires fully observed binary assignment with both treatment arms present;
missing assignments cannot be counted as controls. The SRM test remains
available when there are no covariates, in which case the balance tables are
empty. These diagnostics can identify problems with assignment or logging,
but passing them does not prove that treatment was randomized.

When `build_model_card` runs this check on a scored dataset, it automatically
excludes predicted potential-outcome probabilities (`p_control`, `p_treated`),
uplift scores, derived decile/segment and recommendation fields, and ID
columns. These are not baseline covariates and could otherwise create
misleading imbalance alarms. Genuine pre-treatment covariates are still
checked; pass `extra_excluded` for other dataset-specific non-features.

The Qini, cumulative uplift, and normalized Qini functions also require
one-dimensional inputs with finite outcomes and scores, plus a fully
observed treatment indicator containing both 0 and 1. Treatment values are
validated before integer conversion, avoiding accidental reassignment of
fractional or non-finite labels. Outcome values may be continuous when the
analysis measures incremental amounts instead of binary conversions.

Observed top-fraction policy estimates and decile bootstrap intervals likewise
require fully observed 0/1 treatment assignments and finite numeric outcomes.
They validate the entire evaluation set, including customers below a contact
threshold, before computing rates or drawing bootstrap samples. Outcomes may
be continuous. Bootstrap draws that happen to contain only one treatment arm
are still marked as unidentifiable, whereas malformed original labels are
rejected instead of silently excluded from a rate calculation.

Group-level uplift diagnostics also require complete group identifiers, binary
0/1 treatment assignments, and finite numeric outcomes across every group.
`uplift_by_group` reports treated-minus-control means, including for continuous
outcomes. `subgroup_uplift_with_ci` uses the Newcombe (1998) Wilson-score
confidence interval for a difference of independent binomial proportions, so
it requires binary 0/1 outcomes and a positive integer minimum arm size;
continuous outcomes require a different interval estimator. Unlike the former
Wald interval, Wilson-based bounds remain non-degenerate when both arms have
all successes or all failures. Groups with no observations in one treatment
arm retain unidentifiable (NaN) uplift estimates instead of silently
contributing to the comparison.

Method reference: Newcombe, R. G. (1998), *Interval estimation for the
difference between independent proportions: comparison of eleven methods*,
Statistics in Medicine 17(8), 873–890 ([PubMed](https://pubmed.ncbi.nlm.nih.gov/9595617/)).

The modelling feature splitter and `honest_train_test_split` require observed
binary treatment and conversion labels. They reject fractional and non-finite
values before integer conversion. An honest split keeps both treatment arms
in its training and test partitions. When treatment-outcome cells are too rare
to preserve joint stratification, it emits a warning and stratifies on
treatment alone; conversion balance within each arm is then not guaranteed.
If even the treatment-arm balance is impossible, the split raises an error.
The end-to-end `run_pipeline` uses this same helper, so rare conversion
strata produce an explicit fallback warning instead of a scikit-learn
stratification error. The cross-fitted uplift and doubly robust learners use
the same principle: when a treatment-outcome cell has fewer observations than
the requested number of folds, they warn and fall back to treatment-only
stratification. Every fold still has a held-out portion of each treatment arm,
but conversion rates within arms may vary substantially across folds.

Default logistic-regression outcome models (including cross-validated
calibration) also require two observed outcome classes. When an individual
training arm or fold contains only conversions or only non-conversions, the
learner warns and fits an empirical constant-probability classifier for that
arm. User-supplied classifiers that support single-class fitting retain their
own behavior. This avoids a training exception but **does not manufacture
evidence about the missing outcome class**: effects from extremely sparse
conversion samples remain uncertain and should not drive targeting without
additional data or uncertainty analysis.

S-, T-, X-, and doubly robust learners also validate the original treatment
and binary conversion labels when fitted directly. Models require both arms,
and cross-fitting requires at least as many observations per arm as folds.
Validation is performed before integer conversion, including for OOF uplift
predictions, so malformed labels cannot silently change the learned effects.

Classifier-derived potential outcomes and nuisance propensity estimates must
be finite, valid probability distributions: each probability is in [0, 1] and
each row sums to one (within numerical tolerance). The X-learner also checks
that supplied scalar or row-wise propensity blending weights are finite and
in [0, 1]. These checks prevent invalid nuisance predictions from propagating
into uplift scores and doubly robust pseudo-outcomes.

For randomized experiments, the X-learner defaults to the treatment share
observed during `fit`, rather than assuming a universally balanced 50/50
allocation. Its two arm-specific effect regressions are blended using that
constant assignment propensity. Explicit scalar or per-customer propensities
still override the default; observational treatment assignment generally
requires a conditional propensity estimate rather than the marginal share.

The DR-learner's default propensity classifier uses **unweighted** logistic
regression. Unlike a rare-conversion outcome model, its target is the actual
probability of treatment assignment. Applying inverse-class-frequency weights
to treatment labels would artificially rebalance a campaign with, for example,
10% treated customers toward an estimated 50% propensity, distorting the
augmented inverse-propensity correction. The outcome classifier may still use
class weights independently; callers supplying a custom propensity estimator
must ensure its probabilities represent the actual assignment mechanism.

The standalone `attach_uplift_scores` function applies the same [0, 1]
probability-range and finite-value requirements to both supplied potential
outcome vectors, including predictions from external models. It also requires
one-dimensional, row-aligned inputs and prevents the uplift column from
overwriting either probability column. A negative treated-minus-control
effect is valid and is preserved as negative uplift.

Out-of-fold predictions from `cross_fit_uplift_oof` and the second-stage
effects returned by the X- and DR-learners must likewise contain exactly one
real, finite score per customer. Scalar, multi-column, shortened, complex,
or non-finite predictions are rejected before they can contaminate policy
metrics or customer rankings. Unlike classifier probabilities, unconstrained
effect-regression estimates are not clipped to [-1, 1]; extrapolated scores
should be investigated and calibrated before operational use.

## No leakage

Features must be measured before campaign exposure. Variables created after treatment can leak outcome information or treatment effects.

## Stable unit treatment value

One customer's treatment should not materially change another customer's outcome. This may fail in referral systems, social networks, or shared households.

## Consistent treatment

The treatment should mean roughly the same thing for all users. If different customers receive very different campaigns, treatment-effect estimates become harder to interpret.

## External validity

The estimated uplift is valid for the population and period represented by the experiment. It may not transfer to a different season, product, campaign, or channel.
