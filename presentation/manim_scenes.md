# Manim Scene Plan

## Overview

- **Topic**: Sequential tradeability testing for alpha signals
- **Hook**: When should a quant stop researching an alpha and actually trade it?
- **Target Audience**: MSE342 audience with stochastic calculus, filtering, HJB, and optimal stopping background
- **Estimated Length**: 10 to 14 minutes as a slide talk, with optional short Manim clips
- **Key Insight**: Alpha validation is a stopping problem. The same posterior uncertainty that measures estimation error also measures how fast beliefs can still move.

## Narrative Arc

Start with the practical research decision: continue, activate, or reject. Then reveal the EKV filtering state, replace the statistical terminal loss with an economic payoff, and show that the result is an obstacle problem. The empirical arc then stress-tests the model, fixes the Gaussian false-discovery problem with a dead/alive prior, and validates the method on real OSAP and WRDS evidence.

---

## Scene 1: The Research Funnel

**Duration**: ~30 seconds
**Purpose**: Make the problem concrete before any equations appear.

### Visual Elements

- Four blocks: candidate signal, noisy evidence, admission decision, live capital
- Three final branches: continue, activate, reject
- A small ticking clock above the evidence block

### Content

The signal starts as an idea. Evidence arrives over time. The decision is not a one-shot classification problem, because waiting can both help and hurt.

### Narration Notes

Emphasize that the problem is before portfolio optimization. We are deciding whether the alpha enters the trading system.

### Technical Notes

Use ManimCE `Scene`, `VGroup`, `Rectangle`, `Text`, and `Arrow`. Keep branch labels color-coded: blue for continue, green for activate, red for reject.

---

## Scene 2: Evidence As Drift Plus Noise

**Duration**: ~45 seconds
**Purpose**: Introduce EKV's observation model visually.

### Visual Elements

- A random noisy path \(Y_t\)
- A faint underlying linear drift \(X t\)
- Formula \(Y_t = X t + W_t\)

### Content

Show the noisy path moving around an invisible drift line. Then reveal the formula and label \(X\) as unknown alpha strength.

### Narration Notes

The important idea is not that returns are literally Brownian. The important idea is that normalized evidence can be modeled as signal plus noise.

### Technical Notes

Use `Axes`, `VMobject` path, and `MathTex`. Animate path drawing with `Create`.

---

## Scene 3: Posterior Mean And Posterior Variance

**Duration**: ~50 seconds
**Purpose**: Convert noisy evidence into a belief state.

### Visual Elements

- A prior distribution over \(X\)
- Evidence arrives and posterior narrows
- Two labels: posterior mean \(m_t\), posterior variance \(q_t\)

### Content

Start wide, then shift and narrow as evidence accumulates. The state is not the full path. The state is the posterior belief.

### Narration Notes

This is the first compression step: evidence history becomes posterior mean and variance.

### Technical Notes

Use plotted Gaussian curves with changing mean and variance through `ValueTracker`.

---

## Scene 4: The EKV Aha Moment

**Duration**: ~60 seconds
**Purpose**: Explain why uncertainty creates a stopping problem.

### Visual Elements

- Formula \(d\widehat X_t=\Psi(t,\widehat X_t)d\widehat W_t\)
- A variance bar labeled remaining error
- A second bar labeled learning speed
- Both bars controlled by the same \(\Psi\)

### Content

The same posterior variance is the current error and the local volatility of the posterior mean. Then show the running tradeoff \(c-\Psi^2\).

### Narration Notes

This is the conceptual bridge from filtering to stopping.

### Technical Notes

Use `TransformMatchingTex` between the filtering equation and the variance-decay identity.

---

## Scene 5: Replace Statistical Error With Tradeability

**Duration**: ~45 seconds
**Purpose**: Show the project's contribution.

### Visual Elements

- Left: EKV terminal loss
- Right: economic activation payoff
- A morphing arrow from estimate accurately to trade only if useful

### Content

The posterior state is inherited from EKV, but the terminal reward changes.

### Narration Notes

Say explicitly that EKV does not prove the finance boundary. It supplies the state process.

### Technical Notes

Use `MathTex` blocks and `ReplacementTransform`.

---

## Scene 6: The Tradeability Threshold

**Duration**: ~60 seconds
**Purpose**: Explain the mean-variance payoff.

### Visual Elements

- Formula \(mz-\frac{\lambda}{2}(\sigma_r^2+q)z^2-\kappa\)
- Parabola in exposure \(z\)
- Maximum point \(z^\star\)
- Threshold \(|m|>\sqrt{2\lambda(\sigma_r^2+q)\kappa}\)

### Content

Alpha must clear a hurdle that rises with uncertainty, volatility, and costs.

### Narration Notes

This is the finance intuition behind the terminal payoff.

### Technical Notes

Use `Axes`, a concave quadratic, and a moving dot at the optimum.

---

## Scene 7: The Obstacle Boundary

**Duration**: ~60 seconds
**Purpose**: Make the HJB/VI concrete.

### Visual Elements

- Plane with time on x-axis and posterior mean on y-axis
- Regions: reject, continue, activate
- Formula for the VI appears below

### Content

At every state, compare immediate payoff with the value of waiting.

### Narration Notes

This is where course concepts enter most directly: DPP, HJB, viscosity/free-boundary logic.

### Technical Notes

Use the existing boundary plot as a static image or recreate with colored polygons.

---

## Scene 8: Gaussian Failure

**Duration**: ~45 seconds
**Purpose**: Show why the first model is incomplete.

### Visual Elements

- Gaussian prior with no atom at zero
- Null path wandering until it crosses an activation boundary
- Label: false activation

### Content

If the prior cannot say "dead," it will over-interpret noise as weak alpha.

### Narration Notes

Frame this as a good failure because it motivates the next model.

### Technical Notes

Animate a random walk crossing a boundary. Highlight the crossing in red.

---

## Scene 9: Dead/Alive Prior

**Duration**: ~60 seconds
**Purpose**: Introduce the three-state extension.

### Visual Elements

- Three masses at \(-a,0,+a\)
- Posterior mass shifts as evidence arrives
- Dead probability rises when evidence stays near zero

### Content

The state now includes the posterior probability that the signal is dead.

### Narration Notes

This is the finance-specific modeling move.

### Technical Notes

Use three vertical bars controlled by posterior probabilities.

---

## Scene 10: Real Data Answer

**Duration**: ~60 seconds
**Purpose**: End with empirical validation.

### Visual Elements

- OSAP penalty sweep
- Sequential star versus fixed-horizon points
- WRDS gross versus net summary

### Content

The sequential rule separates real signals from sign-flipped placebos, beats the fixed-horizon tradeoff, and shows that transaction costs matter.

### Narration Notes

End with the thesis: not fixed-date hypothesis testing, but sequential admission under uncertainty and costs.

### Technical Notes

Use existing validated figures as static images in the slide deck. If animated, reveal one result at a time with `FadeIn`.

## Color Palette

- Background: `#07111F`
- Primary: `#4BA3FF` for learning and continuation
- Secondary: `#4DD4A1` for activation and validated alpha
- Warning: `#F05263` for false discovery and costs
- Accent: `#F7B955` for uncertainty and thresholds
- Text: `#F8FAFC`

## Mathematical Content

- \(Y_t = X t + W_t\)
- \(\widehat X_t=\mathbb E[X\mid\mathcal F_t^Y]\)
- \(d\widehat X_t=\Psi(t,\widehat X_t)d\widehat W_t\)
- Variance decay identity
- Economic payoff \(\Phi(m,q)\)
- Finite-horizon value \(V(t,m)\)
- Variational inequality
- Three-state posterior
- Dead/alive payoff \(\Phi_\eta(t,y)\)

## Implementation Order

1. Research funnel and observation model
2. Posterior belief animation
3. EKV identity animation
4. Tradeability payoff animation
5. Boundary region animation
6. Dead/alive prior animation
7. Empirical result reveals
