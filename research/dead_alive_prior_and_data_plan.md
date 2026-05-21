---
title: Dead/Alive Prior and Real-Data Plan
date: 2026-05-21
status: research-note
---

# Dead/Alive Prior and Real-Data Plan

## 0. Research verdict

The best next project direction is not a larger multiperiod portfolio problem. It is a
sequential *signal-admission* problem:

> Before a signal is allowed into a portfolio optimizer, decide whether the evidence says
> it is dead, tradeable long/short, or still worth observing.

This keeps the project close to the MSE 342 course material because the mathematical core
is Bayesian filtering plus optimal stopping. It also makes the finance contribution
specific: most portfolio papers assume an alpha process is already available, whereas this
project asks when a noisy candidate alpha should become actionable after costs and
posterior uncertainty.

The research hypothesis is:

> A posterior dead-alpha state should reduce false activation of noisy zero-alpha signals
> relative to a Gaussian prior, while preserving activation power for genuinely strong
> signals.

This is testable in two ways:

1. controlled Monte Carlo experiments where the true drift is known;
2. real anomaly evidence streams from Open Source Asset Pricing or WRDS/CRSP, evaluated
   by out-of-sample net value after the model decides to activate.

## 1. Why the Gaussian prior is not enough

The current Gaussian-belief model is mathematically useful, but the first simulation
exposed a real weakness:

> A continuous Gaussian prior assumes the candidate signal probably has some nonzero
> alpha strength. If the true alpha is exactly zero, noisy evidence eventually crosses the
> activation boundary too often.

That behavior is not a bug in the implementation; it is a modeling failure. A practical
alpha-validation model needs to explicitly represent:

$$
\theta = 0
$$

as a serious posterior state.

This points to a dead/alive prior.

## 2. Literature support

### EKV 2022: sequential estimation baseline

Ekstrom-Karatzas-Vaicenavicius study Bayesian sequential least-squares estimation for the
unknown drift of a Wiener process. Their key object is posterior uncertainty and the
optimal stopping boundary for when additional observation is no longer worth the cost.

Source: <https://arxiv.org/abs/1901.05410>

Our project already builds from this paper.

### Ekstrom-Vaicenavicius 2015: posterior probability as state

The paper *Bayesian sequential testing of the drift of a Brownian motion* formulates a
decision problem for the sign of an unknown Brownian drift. The optimal state is the
posterior probability

$$
\Pi_t = \mathbb P(B \ge 0 \mid \mathcal F_t^X),
$$

and the Bayes risk reduces to

$$
\inf_\tau \mathbb E[g(\Pi_\tau)+c\tau],
\qquad
g(\pi)=\pi\wedge(1-\pi).
$$

This is important because it says: for classification-style alpha decisions, the right
state is often a posterior class probability, not just a posterior mean.

Source: <https://www.numdam.org/item/10.1051/ps/2015012.pdf>

## 3. Most relevant paper for our next model: three drift values

Buonaguidi 2023 studies the problem of determining a Brownian drift that can take one of
three known values. This is almost exactly the structure we want:

$$
\theta \in \{-a, 0, +a\}.
$$

Interpretation for our project:

- \(-a\): signal is live but points the wrong way,
- \(0\): signal is dead,
- \(+a\): signal is live and tradeable in the expected direction.

The paper reduces the problem to optimal stopping for the two-dimensional posterior
probability process over the three states and finds that the boundaries can be
non-monotone. That warning matters: we should not expect the dead/alive tradeability
boundary to be a simple one-dimensional threshold.

Source: <https://www.sciencedirect.com/science/article/pii/S0304414923000273>

## 3.1. What is actually novel here

The separate ingredients exist:

- EKV gives Bayesian sequential drift estimation with observation cost.
- Sequential drift-testing papers give posterior-probability stopping rules for
  classification decisions.
- Dynamic trading papers study how to trade once expected returns and transaction costs
  are specified.
- Empirical anomaly papers study whether known signals survive implementation costs.

The project's specific contribution is the combination:

$$
\text{Bayesian evidence accumulation}
+\text{ dead/live alpha state}
+\text{ cost-adjusted tradeability payoff}
+\text{ real anomaly admission experiment}.
$$

That is different from a standard multiperiod portfolio problem. In dynamic portfolio
choice, the alpha signal is usually an input and the control is the position or trade. In
this project, the control is whether to keep collecting evidence, activate the signal, or
reject it. The output is a sequential research decision rule, not just a trading policy.

## 4. Recommended next mathematical model

Start with the finite three-state prior:

$$
\theta \in \{-a,0,+a\},
\qquad
\mathbb P(\theta=-a)=p_-,
\quad
\mathbb P(\theta=0)=p_0,
\quad
\mathbb P(\theta=+a)=p_+.
$$

The evidence process remains

$$
dY_t=\theta\,dt+dW_t.
$$

Given \(Y_t=y\), Bayes' rule gives posterior probabilities

$$
\pi_i(t,y)
=
\frac{p_i \exp\{\theta_i y-\theta_i^2t/2\}}
{\sum_j p_j \exp\{\theta_j y-\theta_j^2t/2\}},
\qquad
\theta_i\in\{-a,0,+a\}.
$$

Then

$$
m(t,y)=\mathbb E[\theta\mid Y_t=y]
=
\sum_i \pi_i(t,y)\theta_i,
$$

and

$$
q(t,y)=\operatorname{Var}(\theta\mid Y_t=y)
=
\sum_i \pi_i(t,y)\theta_i^2-m(t,y)^2.
$$

We can keep the same terminal tradeability payoff:

$$
\Phi(t,y)
=
e^{-\rho t}
\left(
\frac{m(t,y)^2}{2\lambda(\sigma_r^2+q(t,y))}
-\kappa
\right)^+.
$$

The dynamic stopping problem can be solved in \((t,y)\)-space. Under the observation
filtration, the innovation representation is

$$
dY_t = m(t,Y_t)\,dt + d\widehat W_t.
$$

So the continuation PDE is

$$
\partial_t V
+m(t,y)\partial_y V
+\frac12\partial_{yy}V
-c
=0,
$$

with obstacle

$$
V(t,y)\ge \Phi(t,y).
$$

Equivalently:

$$
\max\left\{
\Phi(t,y)-V(t,y),
\partial_t V+m(t,y)V_y+\frac12V_{yy}-c
\right\}=0.
$$

This is the next implementation target.

## 5. Why three-state before continuous spike-and-slab

A continuous spike-and-slab prior is also attractive:

$$
\theta \sim p_0\delta_0+(1-p_0)N(0,\tau^2).
$$

It has a closed-form Bayes factor. Conditional on the slab,

$$
m_{\text{slab}}(t,y)=\frac{\tau^2y}{1+\tau^2t},
\qquad
q_{\text{slab}}(t)=\frac{\tau^2}{1+\tau^2t}.
$$

The marginal likelihood ratio of slab versus spike is

$$
\mathrm{BF}_{\text{slab}/0}(t,y)
=
\frac{1}{\sqrt{1+\tau^2t}}
\exp\left\{
\frac{\tau^2y^2}{2(1+\tau^2t)}
\right\}.
$$

Thus

$$
\mathbb P(\text{alive}\mid t,y)
=
\frac{(1-p_0)\mathrm{BF}_{\text{slab}/0}(t,y)}
{p_0+(1-p_0)\mathrm{BF}_{\text{slab}/0}(t,y)}.
$$

This is elegant, but it is slightly more complex to present because "alive" does not imply
direction. The three-state prior is more visual and closer to the 2023 three-drift
sequential-testing paper.

Recommended order:

1. Implement \(\{-a,0,+a\}\) prior.
2. Validate that dead alpha \(\theta=0\) is rejected more often than under the Gaussian prior.
3. Then optionally add spike-and-slab as a smoother extension.

## 6. Validation tests for the next implementation

The three-state implementation should pass these tests before we trust any figure:

1. **Posterior sums to one**:

$$
\pi_-(t,y)+\pi_0(t,y)+\pi_+(t,y)=1.
$$

2. **Symmetry** under symmetric prior:

$$
\pi_+(t,y)=\pi_-(t,-y),
\qquad
\pi_0(t,y)=\pi_0(t,-y),
\qquad
m(t,-y)=-m(t,y).
$$

3. **Dead evidence behavior**: if \(y=0\) and \(t\) grows, the posterior mass on
\(\theta=0\) should increase relative to \(\pm a\), because live states would likely have
produced directional evidence.

4. **Large evidence behavior**:

$$
y\to+\infty \Rightarrow \pi_+\to1,
\qquad
y\to-\infty \Rightarrow \pi_-\to1.
$$

5. **Gaussian limit sanity**: if \(p_0=0\), the model reduces to a two-point sign model.

6. **Innovation validation**: simulate from the prior and verify that the normalized
innovation increments

$$
\Delta \widehat W_t
=
\Delta Y_t - m(t,Y_t)\Delta t
$$

have approximately zero mean and variance \(\Delta t\). This checks the generator used in
the PDE.

7. **Policy validation**: under \(\theta=0\), the three-state policy should have lower
activation frequency than the Gaussian policy at comparable power under
\(\theta=\pm a\). This is the central empirical test of the model, not a cosmetic check.

## 7. How to use real data

We should not start with a huge stock-level trading system. The first real-data version
should turn known signals into normalized evidence streams.

### Data sources

Two practical paths:

1. **Open Source Asset Pricing** for ready-made predictors and portfolio returns. Chen
   and Zimmermann provide replicated stock-level signals and test asset returns from the
   academic asset-pricing literature.

   Source: <https://www.openassetpricing.com/>

2. **WRDS / CRSP / Compustat** for full control. CRSP gives daily and monthly stock data,
   including returns, prices, bid/ask or high/low quote fields, market capitalization,
   shares outstanding, volume, delisting information, and identifiers.

   Source: <https://www.crsp.org/research/crsp-us-stock-databases/>

Because we have WRDS access, the robust course-project path is:

- use Open Source Asset Pricing for fast replication of monthly signals if time is tight;
- use WRDS/CRSP daily returns and volume for the cleaner daily evidence experiment if data
  access and wrangling are smooth.

WRDS tables we likely need:

- CRSP daily stock file: returns, prices, volume, bid/ask or high/low quote fields,
  delisting returns, shares outstanding;
- CRSP names/security file: share codes, exchange codes, active/inactive identifiers;
- CRSP/Compustat merged data if we want accounting signals such as value or profitability;
- Fama-French factors for risk adjustment if we evaluate abnormal returns rather than raw
  long-short returns.

## 8. Mapping a real signal into \(Y_t\)

For a candidate signal \(s\), define a daily or monthly normalized evidence increment:

$$
\Delta Y_t^{(s)}
=
\frac{\mathrm{PNL}_{t}^{(s)}}{\widehat\sigma_s}
$$

or, for cross-sectional evidence,

$$
\Delta Y_t^{(s)}
=
\frac{\mathrm{IC}_t^{(s)}}{\widehat\sigma_{\mathrm{IC},s}}.
$$

Then

$$
Y_T^{(s)}
=
\sum_{t\le T}\Delta Y_t^{(s)}.
$$

Under a dead alpha, this should behave approximately like a zero-drift unit-variance
random walk. Under a live alpha, it should have nonzero drift.

This is the bridge from data to the Brownian model:

$$
\text{normalized signal evidence}
\approx
\theta t + W_t.
$$

## 9. Which real signals to start with

Start with three known and interpretable signals:

1. **Short-term reversal**: high turnover, likely cost-constrained.
2. **Momentum**: more persistent, more likely to survive costs.
3. **Value or profitability**: slower-moving signal, useful contrast.

This connects to implementability literature. Jensen-Kelly-Malamud-Pedersen emphasize
that trading costs change which signals matter for the implementable frontier; short-term
reversal can matter before costs but little after costs for a large investor.

Source: <https://academic.oup.com/rfs/advance-article/doi/10.1093/rfs/hhag022/8524346>

Garleanu-Pedersen provide the dynamic-trading anchor: slower mean-reverting predictors
receive more weight in the optimal aim portfolio when trading is costly.

Source: <https://ideas.repec.org/a/bla/jfinan/v68y2013i6p2309-2340.html>

The reason these three are useful together is that they should produce different
admission behavior:

- reversal should trigger the cost hurdle often and may be rejected even with statistical
  evidence;
- momentum should look more tradeable because evidence persists and turnover is moderate;
- value/profitability should be slow, so the value of waiting can be high even when the
  immediate signal is not yet strong.

## 10. Transaction-cost connection

Our stopping rule should include an implementation hurdle \(\kappa\), not merely a
statistical threshold. Frazzini-Israel-Moskowitz estimate real-world trading costs and
show that capacity varies strongly by style; short-term reversal is especially constrained
by costs.

Source: <https://papers.ssrn.com/sol3/papers.cfm?abstract_id=2294498>

In the first real-data experiment, use a simple hurdle:

$$
\kappa_s
=
\text{estimated spread cost}
+\text{turnover/impact penalty}.
$$

This can be calibrated coarsely at first using signal turnover, ADV, and spread proxies
from CRSP.

## 10.1. Real-data experiment design

For each signal \(s\), construct a rolling evidence stream:

$$
\Delta Y_t^{(s)}
=
\frac{r_{t,\mathrm{long-short}}^{(s)}-\widehat\beta_s^\top f_t}
{\widehat\sigma_s},
$$

where the numerator can be raw long-short return at first, then factor-adjusted abnormal
return. The model only sees the evidence up to date \(t\).

At each decision date:

1. update posterior probabilities over \(-a,0,+a\);
2. compare stop/continue through the computed boundary;
3. if activated, record the future out-of-sample net return over a fixed holdout window;
4. if rejected, record that the model saved the implementation cost;
5. compare against static rules: z-stat threshold, t-stat threshold, and always-trade
   published anomaly.

Primary metrics:

- false activation rate on placebo/permuted signals;
- activation power on established signals;
- average stopping time;
- out-of-sample net Sharpe or certainty-equivalent value after activation;
- turnover and cost drag.

This makes the real-data test answer a precise question:

> Does sequential tradeability admission improve the research pipeline's net value, not
> merely its in-sample t-statistics?

## 11. Recommended next step

Implement the three-state prior first:

$$
\theta \in \{-a,0,+a\}.
$$

Then rerun the same simulation comparison against the Gaussian model.

The paper-level hypothesis is:

> Explicitly assigning posterior probability to a dead-alpha state reduces false activation
> of zero-alpha signals while preserving the ability to activate strong signals when
> evidence is decisive.

This is exactly the weakness our current simulation found, and it is a clean mathematical
extension of the EKV framework.
