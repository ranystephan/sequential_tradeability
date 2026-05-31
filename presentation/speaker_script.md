# Speaker Script: Sequential Tradeability Testing for Alpha Signals

## Slide 1: Sequential Tradeability Testing

My project is about a decision that comes before portfolio construction. Suppose we have a candidate alpha. We have some evidence, but that evidence is noisy. The question is not only whether the alpha is statistically significant. The question is: when should we stop researching it and either activate it or reject it? The central idea is that alpha validation is a sequential real-options problem. Continuing research has option value, but waiting also costs time, capital, and decay. The project builds a Bayesian optimal stopping rule for that tradeoff.

## Slide 2: The Question

The workflow is simple: a signal is proposed, evidence arrives, and eventually the signal is either admitted into live trading or rejected. A fixed-horizon test forces this into one date, for example after 12 months or 60 months. But that is not how research actually feels. Sometimes the signal is clearly dead early. Sometimes it is clearly tradeable early. Sometimes it is close enough that another month of evidence is valuable. So the decision should depend on the current posterior state, not just a calendar endpoint.

## Slide 3: EKV Gives The State

The mathematical anchor is the EKV Bayesian drift estimation model. We observe evidence
\(Y_t = X t + W_t\), where \(X\) is the unknown drift and \(W_t\) is Brownian noise. In the alpha interpretation, \(X\) is the unknown strength of the signal and \(Y_t\) is cumulative normalized evidence. The state is the posterior belief, especially the posterior mean and variance. This is useful because it compresses the whole evidence path into the quantities that matter for decision making.

## Slide 4: The Core Identity

The key EKV identity is that posterior uncertainty plays two roles at once. It is the remaining estimation error, but it is also the volatility of the posterior mean. Intuitively, if I am very uncertain, my estimate is poor, but new evidence can move my belief quickly. If I am already certain, there is less left to learn. That is why optimal stopping appears naturally: one more instant costs \(c\), while learning reduces posterior variance at a rate related to \(\Psi^2\).

## Slide 5: The Move

This is where the project departs from EKV. EKV asks when to stop observing and estimate the unknown drift accurately. I keep the filtering state from EKV, but I change the terminal reward. I am not trying to minimize statistical error directly. I am asking whether the signal is economically tradeable after uncertainty, residual risk, and implementation costs. So \(X\) becomes alpha strength, \(Y_t\) becomes normalized evidence, \(m_t\) becomes posterior mean alpha, \(q_t\) becomes uncertainty, and \(\tau\) becomes an admission or rejection time.

## Slide 6: Economic Payoff

The activation payoff comes from a posterior mean-variance objective. If the signal is traded with signed exposure \(z\), the posterior benefit is \(m z\). The risk penalty is proportional to \((\sigma_r^2+q)z^2\), which combines realized return noise and parameter uncertainty. Then we subtract an implementation hurdle \(\kappa\). Optimizing over \(z\) gives a clean payoff: alpha must be large enough relative to risk, uncertainty, and costs. This is the finance version of the stopping reward.

## Slide 7: Stopping Rule

Once the payoff is fixed, the problem becomes an optimal stopping problem. At every time and posterior mean, the rule compares immediate payoff with continuation value. The variational inequality says: stop if the obstacle dominates, continue if the dynamic programming equation dominates. This is exactly the kind of HJB and optimal stopping object from the course. The important point is that the boundary is not just a t-stat threshold. It is a value boundary.

## Slide 8: First Boundary

This figure shows the first computed Gaussian-belief boundary. Red means stop and activate, gray means stop and reject, and blue means continue observing. The dashed lines are the static activation thresholds. The blue continuation region is largest early because future evidence still has time to change the decision. As the horizon approaches, the option value of waiting disappears. This is the first visual sign that the model is genuinely sequential.

## Slide 9: Gaussian Stress Test

Inside the Gaussian model, the dynamic boundary does what we want for strong positive or negative alphas: it produces higher realized value than the static payoff threshold or a simple z-stat rule. But the same experiment reveals a serious problem near zero. When the true alpha is dead, the Gaussian prior still assumes the drift is continuously distributed, so it keeps looking for a nonzero drift. That creates too many false activations. This failure is useful because it tells us the prior is missing a dead state.

## Slide 10: Dead/Alive Prior

The next model uses a three-state prior: live short, dead, and live long. This is much closer to actual alpha research, where many candidate signals are not merely small but truly useless. The posterior probabilities update by Bayes rule. If evidence stays near zero for a long time, the posterior probability of the dead state rises. I also add a false-discovery penalty, proportional to the posterior probability that the signal is dead. That explicitly prices the cost of admitting bad signals into production.

## Slide 11: Controlled Validation

The dead/alive stress test shows the intended tradeoff. At true alpha zero, the matched Gaussian boundary activates 47.1 percent of paths, while the dead/alive boundary activates 39.8 percent. Null realized value improves. But this is not free dominance. For live alphas, the dead/alive rule gives up some value because it is more conservative. That is actually a strength of the model. It exposes the false-discovery versus power tradeoff rather than pretending we can get both for free.

## Slide 12: Real Data Calibration

The first real-data experiment uses OSAP monthly anomaly returns. Each return stream is normalized into Brownian evidence increments. To check that the rule is responding to directional evidence rather than volatility alone, I compare real signals to sign-flipped placebos. The placebo keeps the volatility structure but destroys persistent drift. As the false-discovery penalty increases, both real and placebo activation fall, but placebo activation falls much faster. At \(\eta=0.04\), real activation is 72.9 percent, while placebo activation is 27.1 percent.

## Slide 13: Sequential Or Fixed Horizon?

This is the main empirical comparison. I calibrate fixed-horizon tests so they activate the same number of sign-flipped placebos as the sequential rule. The sequential rule activates 72.9 percent of real signals with an average decision time of 47.9 months. The 60-month fixed test is close in time but has lower activation-weighted holdout return. The 120-month test activates more signals, but it waits much longer and has lower weighted holdout performance. So the sequential rule is not just a fixed-horizon test in disguise.

## Slide 14: WRDS And Conclusion

Finally, I rebuild three signals from WRDS using CRSP and Compustat: momentum, book-to-market, and operating profitability. Gross returns look encouraging: all three activate in the standard direction. But under a deliberately harsh range-cost stress test, none survive in the standard direction. This is the practical message. A signal can be statistically real and still not be tradeable. The final thesis is that alpha validation should be modeled as a sequential admission problem, where learning, false discoveries, decay, and implementation costs are all part of the same decision.
