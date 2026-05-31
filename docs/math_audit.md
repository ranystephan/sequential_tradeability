# Mathematical Audit

This note checks the project against the MSE342 lecture spine: stochastic calculus
and generators from Lecture 1, and dynamic programming, HJB, viscosity solutions, and
optimal stopping from Lectures 5-12.

## Verdict

The mathematical core is sound. The project is not claiming that the EKV stopping theorem
directly proves the finance rule. Instead, EKV supplies the filtering state dynamics,
and the project defines a new optimal stopping problem with an economic terminal payoff.
That distinction is now explicit in the paper.

## EKV Filtering Block

- Observation model: \(Y_t=Xt+W_t\) is the right Brownian drift-learning model.
- Posterior likelihood:
  \[
  \mu_{t,y}(du)\propto \exp\{uy-u^2t/2\}\mu(du)
  \]
  is correct by Gaussian likelihood under drift \(u\).
- Posterior mean and variance:
  \[
  G(t,y)=\mathbb E[X\mid Y_t=y],\qquad H(t,y)=\operatorname{Var}(X\mid Y_t=y)
  \]
  are correct.
- Derivative identity:
  \[
  \partial_yG(t,y)=H(t,y)
  \]
  follows by differentiating the exponential tilt. For non-degenerate priors this makes
  \(G_t\) increasing, so \(G_t^{-1}\) is well-defined on the posterior-mean range.
- Innovation identity:
  \[
  \widehat W_t=Y_t-\int_0^t\widehat X_s\,ds
  \]
  is Brownian in the observation filtration, and
  \[
  d\widehat X_t=\Psi(t,\widehat X_t)d\widehat W_t,\qquad
  \Psi(t,\widehat X_t)=\operatorname{Var}(X\mid\mathcal F_t^Y)
  \]
  is consistent with the filtering calculation.

## EKV Stopping Identity

The variance decay identity is correct:
\[
\mathbb E[\Psi(\tau,\widehat X_\tau)]
=\operatorname{Var}(X)-\mathbb E\int_0^\tau\Psi^2(s,\widehat X_s)\,ds .
\]
The reason is exactly Lecture 1's martingale/quadratic-variation machinery:
\(\widehat X_t\) is an \(L^2\) martingale with diffusion coefficient \(\Psi\), so its
quadratic variation accumulates at rate \(\Psi^2\). This gives the running-cost form
\[
\inf_\tau\mathbb E\int_0^\tau(c-\Psi^2)\,ds.
\]

## Checkable Priors

- Gaussian prior: \(q(t)=q_0/(1+q_0t)\), \(dm_t=q(t)d\widehat W_t\), and the EKV
  benchmark stopping time \(\tau^*=(1/\sqrt c-1/q_0)^+\) are correct.
- Symmetric Bernoulli prior: \(\Psi(x)=\beta^2-x^2\), immediate stopping when
  \(\beta^4\le c\), and the free-boundary integral equation match the EKV result.

## Finance Payoff

The activation payoff
\[
\sup_z\left\{mz-\frac{\lambda}{2}(\sigma_r^2+q)z^2-\kappa\right\}
=\frac{m^2}{2\lambda(\sigma_r^2+q)}-\kappa
\]
is algebraically correct. The reject option gives the positive part. The interpretation
of \(q\) as predictive parameter risk is a modeling assumption, not an EKV theorem; the
paper now says this explicitly.

## Dynamic Programming And VIs

For the Gaussian tradeability state \(dm_t=q(t)d\widehat W_t\), the generator is
\[
\mathcal L_t f(m)=\frac12 q(t)^2 f_{mm}(m).
\]
With value measured in calendar time from validation start, the obstacle problem
\[
\max\left\{e^{-\rho t}\Phi(m,q(t))-V,\ V_t+\frac12q(t)^2V_{mm}-c\right\}=0
\]
is consistent with the lecture DPP. If the value were measured relative to the current
time, the equivalent form would use \(e^{-\rho(\tau-t)}\) and a \(-\rho V\) continuation
term.

For the three-state model, posterior-predictive evidence satisfies
\[
dY_t=m(t,Y_t)dt+d\widehat W_t,
\]
so the generator is
\[
\mathcal L_t f(y)=m(t,y)f_y(y)+\frac12f_{yy}(y).
\]
Therefore
\[
\max\{\Phi_\eta(t,y)-V,\ V_t+mV_y+\tfrac12V_{yy}-c\}=0
\]
is the right finite-horizon optimal stopping variational inequality.

## Numerical Soundness

The finite-difference solvers are mathematically aligned with the lecture material:

- backward implicit Euler corresponds to backward dynamic programming;
- applying `max(payoff, continuation)` enforces the obstacle;
- the three-state solver uses nonnegative upwind transition rates for the drifted
  generator, giving a monotone scheme appropriate for viscosity-solution free boundaries;
- boundary values are pinned to the immediate payoff, which is acceptable when the grid
  domain is chosen wide enough that stopping is optimal at the edges.

The current diagnostics are strong: terminal value error is zero to numerical precision,
obstacle violation is zero in the reported dead/alive run, symmetry error is about
\(4.2\times10^{-16}\), and the grid-convergence table shows stable boundary locations.

## Empirical Scaling

The Brownian evidence transform
\[
\Delta Y_t=(r_t/\widehat\sigma_m)\sqrt{dt}
\]
is correct for monthly returns: under the null,
\(\operatorname{Var}(\Delta Y_t)\approx dt\). The drift is therefore an annualized
Sharpe-like evidence strength.

## Remaining Caveats

- Real anomaly returns are not literal Brownian increments. The paper should continue to
  report variance and autocorrelation diagnostics.
- The fixed \(\kappa\) cost is a tractable first-order hurdle. A more realistic version
  would make costs exposure, turnover, and liquidity dependent.
- The WRDS net-cost experiment is best described as a stress test until spread/effective
  cost data are added.
