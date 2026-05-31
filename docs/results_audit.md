# Results And Figure Audit

This note checks whether the plotted results in the paper match the generated CSV files
and whether their economic interpretation is coherent.

## Gaussian Boundary Figure

The original figure was mathematically coherent but visually too severe: with
\(c=0.02\), \(\rho=3\), and \(T=1\), continuation occupied only about \(0.72\%\) of the
grid. The regenerated figure uses a milder benchmark,
\(q_0=0.10\), \(\kappa=0.025\), \(c=0.005\), \(\rho=1\), and \(T=2\), and zooms into the
central posterior-mean range of a wider computational grid. Under this specification,
continuation occupies about \(11.9\%\) of the grid, rejection about \(2.2\%\), and
activation about \(85.8\%\). This makes the free-boundary geometry visible without
changing the model or solver.

## Gaussian Simulation Comparison

The simulation plot matches `results/simulation_policy_comparison.csv`.

- At \(\theta=\pm0.3\), the dynamic boundary has mean realized value about \(0.019\),
  far above the static payoff rule and the fixed \(z\)-stat rule.
- At \(\theta=0\), the Gaussian dynamic rule has mean value \(-0.0157\) and activates
  about \(90.1\%\) of paths. This is bad behavior, but it is exactly the intended
  diagnostic: a Gaussian prior cannot represent a truly dead alpha.
- The conclusion in the paper is correct: the Gaussian model is a useful benchmark, but
  it over-activates null signals and motivates the dead/alive prior.

## Dead/Alive Stress Test

The dead/alive plot matches `results/dead_alive_policy_comparison.csv`.

- At \(\theta=0\), the matched Gaussian boundary activates \(47.1\%\) of paths, while the
  dead/alive boundary activates \(39.8\%\).
- Null realized value improves from \(-0.00560\) to \(-0.00513\).
- Live-alpha value is slightly lower under the dead/alive boundary than under the matched
  Gaussian boundary, so the paper correctly describes this as a false-discovery/power
  tradeoff rather than dominance.
- The dead/alive \(z\)-stat baseline is nearly always rejecting at the chosen threshold,
  which makes it conservative but not very competitive. It is still a useful visual
  anchor.

## OSAP Penalty Sweep

The penalty sweep matches `results/osap_penalty_sweep.csv`.

- At \(\eta=0\), real activation is \(91.1\%\) and placebo activation is \(60.4\%\).
- At \(\eta=0.04\), real activation is \(72.9\%\) and placebo activation is \(27.1\%\).
- At \(\eta=0.08\), real activation falls to \(62.5\%\) and placebo activation to
  \(15.1\%\).
- The comparative static is exactly what the model predicts: higher false-discovery
  penalty lowers both activation rates, but lowers placebo activation faster.

## OSAP Fixed-Horizon Benchmark

The fixed-horizon plot matches `results/osap_static_benchmark_comparison.csv`.

- Sequential boundary: \(72.9\%\) real activation, \(27.1\%\) placebo activation, mean
  decision month \(47.9\), activated holdout return \(0.747\%\), activation-weighted
  holdout \(0.545\).
- Fixed 60-month benchmark: \(70.8\%\) real activation and activation-weighted holdout
  \(0.500\).
- Fixed 120-month benchmark: \(77.1\%\) real activation and activation-weighted holdout
  \(0.531\).
- Fixed 240-month benchmark: \(82.8\%\) real activation and activation-weighted holdout
  \(0.444\).

The paper's interpretation is sound: the sequential rule does not maximize raw activation
count, but it reaches near-long-horizon admission power much earlier and with stronger
activation-weighted holdout performance.

## WRDS Table

There is no WRDS figure in the paper, but the table matches
`results/wrds_stock_level_summary.csv`.

- Gross returns: all three reconstructed signals activate in the standard direction,
  with mean standard-direction holdout return \(1.227\%\) per month.
- Net range-cost stress: all three activate only in the reversed/symmetric sense, and
  zero activate in the standard direction. This supports the paper's wording that the
  range-cost version is a harsh transaction-cost stress test, not a final execution-cost
  result.

## Overall Assessment

The results are internally consistent and support the story. The strongest empirical plot
is the OSAP fixed-horizon comparison, because it directly tests whether the method is
more than a fixed-horizon posterior threshold. The Gaussian boundary plot is now also a
better teaching figure: it visibly shows continue, activate, and reject regions.
