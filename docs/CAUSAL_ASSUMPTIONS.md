# Causal Assumptions

Uplift modelling relies on stronger assumptions than ordinary prediction.

## Randomized treatment assignment

The cleanest setting is a randomized experiment where treatment and control groups are comparable before treatment.

## No leakage

Features must be measured before campaign exposure. Variables created after treatment can leak outcome information or treatment effects.

## Stable unit treatment value

One customer's treatment should not materially change another customer's outcome. This may fail in referral systems, social networks, or shared households.

## Consistent treatment

The treatment should mean roughly the same thing for all users. If different customers receive very different campaigns, treatment-effect estimates become harder to interpret.

## External validity

The estimated uplift is valid for the population and period represented by the experiment. It may not transfer to a different season, product, campaign, or channel.
