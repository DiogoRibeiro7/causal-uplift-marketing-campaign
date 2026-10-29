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

## No leakage

Features must be measured before campaign exposure. Variables created after treatment can leak outcome information or treatment effects.

## Stable unit treatment value

One customer's treatment should not materially change another customer's outcome. This may fail in referral systems, social networks, or shared households.

## Consistent treatment

The treatment should mean roughly the same thing for all users. If different customers receive very different campaigns, treatment-effect estimates become harder to interpret.

## External validity

The estimated uplift is valid for the population and period represented by the experiment. It may not transfer to a different season, product, campaign, or channel.
