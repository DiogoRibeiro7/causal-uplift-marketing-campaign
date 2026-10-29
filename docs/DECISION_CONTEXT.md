# Decision Context

The business decision is not simply whether a customer is likely to convert.

The decision is whether contacting a customer is expected to change their behaviour enough to justify the campaign cost.

## Standard response modelling question

> Who is most likely to convert?

## Uplift modelling question

> Who is most likely to convert because of the campaign?

This difference changes the campaign policy.

A high-probability customer may be a poor target if they would have converted anyway. A lower-probability customer may be a valuable target if the campaign meaningfully increases their probability of conversion.

## Segment logic

- **Persuadables** should usually receive the campaign.
- **Sure things** may not need paid contact.
- **Lost causes** should not receive expensive targeting.
- **Do-not-disturb** users may be harmed by contact and should be avoided.
- **Review** users need more evidence or cheaper experimentation.

The segment rules never classify zero or negative estimated uplift as
"persuadable", even when no positive scores exist and the automatically
calculated threshold becomes zero. Negative uplift takes priority over
high baseline conversion probability: these customers remain
"do-not-disturb". Non-finite scores and probabilities outside the unit
interval are rejected before generating recommendations.

## Policy estimates at tied uplift scores

`policy_value_at_fraction` evaluates an exact expected contact budget of
`ceil(N * fraction)` customers. If the budget falls inside a block of equal
uplift scores, there is no model-based reason to prefer one customer in that
block over another. It therefore gives every customer in the boundary block
the same fractional contact weight, with fully selected customers above the
boundary receiving weight 1 and customers below it receiving weight 0.

Reported treated and control outcome rates are weighted means. This is a
deterministic, row-order-invariant estimate for uniform tie-breaking, not the
realised outcome of one randomly chosen contact list. The same convention is
applied in policy bootstrap resamples; actual deployment must still resolve
ties using an independent, reproducible random rule.
