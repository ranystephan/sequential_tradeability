# Slide Plan

## Audience And Timing

- Audience: MSE342 professor and classmates
- Expected length: 10 to 14 minutes
- Style: modern, sparse, math-forward, intuition-first
- Main objective: show that the project is a real stochastic control and optimal stopping extension, not a generic finance backtest

## Claim Spine

1. **Sequential Tradeability Testing**: Alpha validation is a sequential real-options problem.
2. **The Question**: Fixed-horizon tests miss the actual research decision.
3. **Paper Anchor**: EKV gives the posterior state for noisy drift learning.
4. **Core Identity**: Posterior uncertainty is both estimation error and learning speed.
5. **The Move**: Keep EKV filtering, replace statistical terminal loss with economic tradeability.
6. **Economic Payoff**: Activation must clear alpha, uncertainty, risk, and implementation costs.
7. **Stopping Rule**: The rule is a finite-horizon obstacle problem.
8. **First Boundary**: The computed boundary has reject, continue, and activate regions.
9. **Stress Test**: Gaussian learning works for strong alphas but over-activates dead signals.
10. **Dead State**: A three-state prior models live-short, dead, and live-long signals.
11. **Controlled Validation**: Dead/alive lowers false activation with a power tradeoff.
12. **Real Data Calibration**: OSAP sign-flip placebos fall faster as the penalty rises.
13. **Sequential Benchmark**: The rule is not just a fixed-horizon test in disguise.
14. **WRDS And Thesis**: Gross alpha is not enough once costs enter.

## Design System

- Format: 16:9, 1280 by 720
- Background: deep ink
- Typography: Aptos Display for claims, Aptos for labels
- Palette: blue for learning, green for activation, red for false discovery/costs, amber for uncertainty
- Layout rule: one dominant proof object per slide
- Formula rule: formulas are rendered as large transparent assets
- Empirical proof rule: use validated project figures, not redrawn approximations

## QA Notes

- No em dash characters are used.
- The first boundary figure was regenerated to make the continuation region visible.
- OSAP and WRDS numbers match the project CSVs and paper.
- The deck is built with artifact-tool and rendered to PNG previews for visual inspection.
