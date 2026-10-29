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

The modelling feature splitter and `honest_train_test_split` require observed
binary treatment and conversion labels. They reject fractional and non-finite
values before integer conversion. An honest split keeps both treatment arms
in its training and test partitions. When treatment-outcome cells are too rare
to preserve joint stratification, it emits a warning and stratifies on
treatment alone; conversion balance within each arm is then not guaranteed.
If even the treatment-arm balance is impossible, the split raises an error.

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

The standalone `attach_uplift_scores` function applies the same [0, 1]
probability-range and finite-value requirements to both supplied potential
outcome vectors, including predictions from external models. It also requires
one-dimensional, row-aligned inputs and prevents the uplift column from
overwriting either probability column. A negative treated-minus-control
effect is valid and is preserved as negative uplift.

## No leakage

Features must be measured before campaign exposure. Variables created after treatment can leak outcome information or treatment effects.

## Stable unit treatment value

One customer's treatment should not materially change another customer's outcome. This may fail in referral systems, social networks, or shared households.

## Consistent treatment

The treatment should mean roughly the same thing for all users. If different customers receive very different campaigns, treatment-effect estimates become harder to interpret.

## External validity

The estimated uplift is valid for the population and period represented by the experiment. It may not transfer to a different season, product, campaign, or channel.
