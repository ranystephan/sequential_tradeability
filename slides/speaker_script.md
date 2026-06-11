# Speaker Script — Beamer Deck (`slides/main.pdf`)

This is the narration for the Beamer deck (29 slides). Read the text after **SAY**.
**CLICK** = press advance once to reveal the next piece *within* the current slide;
**→** = advance to the *next* slide. Most slides are a single block (just **→** at the end).
Five slides — **9, 14, 15, 21, 25** — reveal incrementally, so they carry **CLICK** markers
between pieces; say each beat, then click to reveal the next. Target ~30 minutes, roughly a
minute a slide; don't rush the math slides (5–14).

Register: first-person, seminar-level, plain. The audience can read the equations, so use
the narration to say *why* each object is there and *what I did with it*. Keep EKV's results
attributed to EKV; everything I built, I say in the first person.

---

## Slide 1 — Title

SAY: "Most candidate trading signals look good in a backtest but stop working once you account
for uncertainty, decay, turnover, and trading costs. The question this project asks sits upstream
of all of that: when is a signal trustworthy enough to actually put into live trading? The tools
are Bayesian filtering and optimal stopping. Let me start with where a signal ends up being
used."

→

## Slide 2 — Portfolio optimization needs trusted alpha

SAY: "Here's the motivation from the portfolio side. In a Markowitz or Boyd-style optimizer,
you choose portfolio weights to
maximize expected alpha, minus a risk penalty, minus trading costs — subject to a budget
constraint, position limits, factor-exposure targets, and a turnover budget. We have good
convex machinery for all of that: the risk model, the constraints, the cost model. But every
piece of it takes alpha, the expected-return vector, as a *given input* — and by alpha I just
mean a signal's risk-adjusted excess-return edge. My project is the upstream question. Alpha
isn't really given; it's an uncertain object you learn from noisy
evidence, and the most fragile input in the whole pipeline. So before a signal is admitted as
alpha into an optimizer like this, how much evidence do I need that it's genuinely, economically
tradeable? I'm not replacing the optimizer; I'm studying the alpha-validation layer that decides
what gets allowed in."

→

## Slide 3 — The object of the project

SAY: "Here's the thesis in one line. Alpha validation isn't naturally a fixed-date significance
test; it's a stopping problem under posterior uncertainty, where more evidence has option value
but delay, alpha decay, and implementation costs push back. My starting point is the
Ekström–Karatzas–Vaicenavicius model for the posterior dynamics: I keep their state and change
what gets optimized, which I'll make precise in a few slides."

→

## Slide 4 — Fixed horizons are not admission rules

SAY: "To sharpen the contrast. The standard practice fixes a horizon T and asks, only at that
date, whether the signal is statistically significant — it conditions on an exogenous date and
separates statistical evidence from implementation value. The sequential formulation instead
makes the decision time τ endogenous: it comes out of the posterior state, not the calendar.
At τ the stopping region splits into activation and rejection, and continuation prices one more
observation. The sequential machinery is EKV's; what's mine is posing *validation* this way,
and the three-way decision — continue, activate, reject — where EKV have only two."

→

## Slide 5 — Storyline

SAY: "The talk has six steps. First EKV's drift-learning model. Then the identity that makes
the stopping problem clean. Then my move from a statistical to an economic reward. Then a
stress test showing Gaussian beliefs can't represent a dead signal. Then the three-state fix.
And finally the empirical tests on OSAP and WRDS data. Let me start with EKV."

→

## Slide 6 — EKV starts with drift learning

SAY: "EKV observe a process Y-t equal to X-t plus a Brownian motion: an unknown constant drift
X, plus noise. Their paper is pure sequential *estimation*, with no economics in it yet. For my
purposes, X is the latent alpha strength and Y-t is cumulative normalized evidence. The filtration is just
the information in the observed path; the drift and the noise are never seen separately, and
disentangling them is the whole inference problem."

→

## Slide 7 — Bayes' rule gives the posterior state

SAY: "Bayes' rule turns that path into a posterior over X — the prior tilted by the Gaussian
likelihood. The two summaries I keep are the posterior mean G, my estimate of X, and the
posterior variance H, my uncertainty. There's a useful identity here: the derivative of the
posterior mean with respect to the evidence equals the posterior variance. In words, how strongly
my estimate reacts to a new observation is exactly my current uncertainty — when I'm unsure the
estimate swings a lot, and as I grow confident it barely moves. That same quantity comes back in a
moment as the learning rate. Those two summaries are the low-dimensional state of the whole
problem."

→

## Slide 8 — The filtering identity

SAY: "This is the identity everything downstream leans on. Define the innovation W-tilde by
subtracting the running estimate from the evidence; it's a Brownian motion in the observation
filtration. Then the posterior mean satisfies d-X-hat equals Psi d-W-tilde, where Psi is the
posterior variance. Because there's no drift term, the posterior mean is a martingale — before
the next observation it's already my best estimate. And the same Psi plays two roles: it's the
remaining estimation error *and* the volatility of belief updates. That coincidence — the error
equals the learning rate — is why the stopping problem stays clean."

→

## Slide 9 — EKV's optimal stopping problem

SAY: "EKV minimize squared estimation error plus a cost for time — they study the optimal time
to stop observing and report the posterior mean."

CLICK

SAY: "The key step is the identity that appears now — the variance-decay identity. Expected
posterior variance equals the prior variance minus the accumulated squared learning rate. Two
facts give it: posterior variance equals mean-squared error, and by Itô the martingale's
quadratic variation is Psi-squared. Plainly, observing burns variance at rate Psi-squared."

→

## Slide 10 — The EKV problem becomes a running-cost problem

SAY: "Substituting that identity, EKV's estimation problem becomes a running cost — integrate c
minus Psi-squared. Each instant of observing costs c and earns variance reduction Psi-squared,
so you continue only while the benefit of information exceeds its cost. This value-of-information
logic is exactly what I carry into the finance problem. I do not carry over their stopping time
— only the posterior state and this principle."

→

## Slide 11 — Two EKV benchmarks

SAY: "Two closed-form cases anchor my code. The Gaussian prior gives a deterministic posterior
variance and an explicit stopping time, so I can test the filter against an exact answer. The
symmetric Bernoulli prior gives a continuation band in posterior-mean space — and that is
exactly the two-threshold sequential-probability-ratio-test geometry from the course: keep
observing inside the band, stop once you leave it. It's the shape I reuse for admission."

→

## Slide 12 — The project changes the terminal reward

SAY: "Here is that change, stated precisely. I keep EKV's posterior state process exactly as it
is, but I replace their statistical terminal loss with an economic admission payoff. This is
*not* a corollary of their theorem: the moment the reward changes, it's a new obstacle problem
that I solve and validate myself, using their filter only as the state."

→

## Slide 13 — Variable translation

SAY: "The dictionary is direct. The drift becomes alpha strength; the posterior mean becomes my
estimate of alpha; the posterior variance becomes both parameter risk and learning speed; the
cost becomes research and opportunity cost; and the stopping time becomes the
activation-or-rejection decision."

→

## Slide 14 — Economic activation payoff

SAY: "The reward itself. If I activate the signal with exposure z — sized by risk aversion
lambda against return variance sigma-r-squared and parameter uncertainty q — the posterior
value is expected return, minus a risk penalty, minus a fixed implementation hurdle."

CLICK

SAY: "It's concave in z, so the optimal exposure is closed form: z-star is m, divided by lambda
times the quantity sigma-r-squared plus q. So the position scales with my estimate of alpha and
shrinks as either return variance or parameter uncertainty grows — bigger edge, bigger bet; more
risk or more doubt, smaller bet. Substituting it back gives the payoff Phi as a positive part,
because I can always decline to trade."

CLICK

SAY: "That yields a static activation threshold — but only the *terminal* one; the dynamic
boundary differs, because waiting can still raise the payoff."

→

## Slide 15 — The finite-horizon stopping problem

SAY: "Recall the state here is the posterior mean m — that's X-hat — and its variance q, which
is Psi. I write the finite-horizon value function: the best expected discounted payoff, net of
observation cost, over all stopping times up to the horizon."

CLICK

SAY: "Writing the obstacle g and the generator L-t, dynamic programming gives the variational
inequality — the max of two terms is zero, with V at least g."

CLICK

SAY: "At each state, either it's optimal to stop, where V equals g — the stopping region S — or
the continuation equation holds, where V exceeds g — the continuation region C; the free
boundary between them is fixed by value matching, with smooth pasting where the value is
regular. I solve the inequality numerically rather than imposing those by hand. One convention
note, since it differs from the textbook HJB: I keep discounting inside the obstacle, so there
is no minus-rho-V term — that's the calendar-time formulation, and rho is the same discount rate
as the American-option example, here standing for alpha decay."

→

## Slide 16 — What the boundary means

SAY: "This is the computed boundary for the Gaussian model. Red is activate, gray is reject —
together the stopping set S — and blue is continue, the continuation set C; the dashed curves
are the static threshold. The important feature is that blue band: it's a value-of-information
region, not where the signal already pays. Early on, almost everything is continue or still
ambiguous; as the deadline approaches, the gray reject wedge in the middle opens up and squeezes
the continue band out toward the activation thresholds. The two black lines are simulated
posterior-mean paths: one accumulates enough directional evidence to cross into activation — the
red marker — and the other stays ambiguous and drops into the reject wedge."

→

## Slide 17 — First stress test: Gaussian beliefs

SAY: "Before any real data, I stress-test the rule inside its own model. I simulate evidence
paths at fixed true alphas and compare the dynamic boundary against two simple baselines. For
strong true alphas it works well — it extracts the learning option value. But the same
experiment exposes a limitation: at a true alpha of zero, the Gaussian rule activates about
ninety percent of null paths. This is in-model simulation, not data — and it's the diagnosis
that drives the rest of the talk."

→

## Slide 18 — The Gaussian prior rules out a point-mass null

SAY: "The reason is structural, not a tuning failure. Under a diffuse Gaussian prior the
probability that X is exactly zero is zero, so the model has no state for a dead signal — and as
pure noise accumulates the posterior mean eventually wanders across the activation line. No
choice of cost fixes that, because it lowers all activation, including true alpha, without
creating a null-probability variable. So the fix is to add an explicit null-alpha state and let
evidence move mass into or out of it."

→

## Slide 19 — A three-state prior for alpha admission

SAY: "The extension is a three-state prior — negative alpha, null alpha, positive alpha. It's a
spike-and-slab model: a spike at the null and two directional live states. The state space now
separates the *sign* of the alpha from the probability that there is no admissible alpha at
all."

→

## Slide 20 — Posterior probabilities under the three-state prior

SAY: "The posterior is finite-state Bayes; the mean and variance follow from the three
probabilities. And here is the mechanism the Gaussian lacked. Each live state's likelihood carries
the exp-minus-theta-squared-t-over-two factor on the slide — it decays with time unless matching
directional evidence arrives to offset it. So when the cumulative evidence stays flat, that decay
alone pushes mass onto
the null: a genuine alpha would most likely have drifted by now, so the *absence* of a signal is
itself evidence the signal is dead. When evidence does trend, mass moves to the matching live
state. That inference is exactly what the point-mass-free Gaussian could not make."

→

## Slide 21 — False-discovery penalty

SAY: "But the dead state alone is not enough. In simulation, with the pure trading payoff, the
rule still over-activates null paths. The reason is that the payoff is myopic: it scores expected
profit under current beliefs and never charges for the *chance* those beliefs are wrong — for
admitting a false discovery. So I subtract a penalty: eta times the posterior null probability,
which is exactly the expected cost of admitting a dead signal. The decision now prices not just
expected profit, but the probability the alpha is not there at all."

CLICK

SAY: "One technical note: in the three-state case the posterior mean's volatility now depends on
the state and vanishes at the extremes, so it's cleaner to solve in the raw evidence y, where the
process has constant volatility and a simple drift. The generator, written L-t f on the slide, is
m times f-y plus a half f-y-y, solved with a monotone upwind scheme. As eta rises, signals
carrying high null probability fall below the threshold and are rejected."

→

## Slide 22 — Controlled validation of the three-state rule

SAY: "The controlled test of that fix, in simulation, with a deliberately fair comparison: the
Gaussian benchmark is given the same prior second moment, so the only difference is the explicit
dead state and its penalty. The three-state rule cuts null activation from 47.1 to 39.8 percent
and improves the realized value on null signals — and by realized value I mean the actual payoff
under the true theta, sizing the position at z-star from the estimate. It isn't free: value on
the genuinely live alphas is slightly lower. So this is a disciplined false-discovery-versus-power
trade-off, not a claim of dominance."

→

## Slide 23 — Real-data design

SAY: "Now real data. I use 192 Chen–Zimmermann open-source anomaly portfolios — these are
long–short strategies, long the high-signal stocks and short the low ones, and they're real
returns, not simulated. Each signal gets 120 months to calibrate volatility, then up to 240
months of sequential validation under the rule, and a 120-month out-of-sample holdout to score
the decision. I turn monthly returns into Brownian evidence by dividing by calibrated volatility
and multiplying by root-dt, so under the null the increments have variance about dt. I have no
ground truth on real data, so I build a placebo: I sign-flip each signal's validation returns.
That keeps the volatility but destroys persistent direction — it's my stand-in for a dead
signal, and I use it to calibrate the penalty."

→

## Slide 24 — Penalty calibration with real signals and placebos

SAY: "This is how I set eta. With no penalty the rule is too loose; as eta rises, both real and
placebo activation fall, but placebo activation falls faster, so the two separate. At eta equal
to 0.04, the rule activates 72.9 percent of real signals but only 27.1 percent of the
sign-flipped placebos — concretely, 140 of 192 real against 52 of 192 placebos. That separation
is the first real-data evidence that the rule responds to persistent direction, not just to
volatility. One honest caveat: the Brownian scaling is only approximate — the increment variance
is about 1.24 times dt, with mild autocorrelation — which is part of what motivates the
stock-level experiment."

→

## Slide 25 — Is the rule really sequential?

SAY: "This is the benchmark I cared about most: is the rule genuinely sequential, or just a
fixed-horizon test in disguise? I compare against fixed horizons calibrated on the placebos to
admit the same number of false positives, so every rule faces the same false-positive rate. The
left panel is admission power; the right is the post-decision holdout."

CLICK

SAY: "The sequential rule — the star — activates 72.9 percent of real signals, at an average
decision month of 47.9, with the highest activation-weighted holdout of the set, 0.545."

CLICK

SAY: "Longer fixed horizons activate more real signals — but only by waiting ten or twenty years,
and their holdout is weaker. So the advantage isn't raw count; it's *earlier* admission at matched
placebo activation with stronger post-decision returns. And the reason is structural: a fixed
horizon must commit to its decision date in advance, while the sequential rule lets the evidence
choose the date — it stops the moment the posterior is conclusive. That freedom to choose *when*
to decide is precisely the optimal-stopping value EKV formalizes, and here it pays off on real
anomaly data."

→

## Slide 26 — WRDS stock-level reconstruction

SAY: "A stock-level robustness check, one layer closer to implementation. I rebuild momentum,
book-to-market, and operating profitability directly from CRSP and Compustat — value-weighted
decile portfolios, long the top tenth of stocks by signal and short the bottom tenth, built from
millions of stock-months — and I apply the same three-state rule with no retuning. On gross returns, all three are admitted in the standard direction, with a positive
holdout. But under a deliberately harsh range-cost transaction stress, none survive in the
standard direction. The practical point: statistical evidence that a signal exists is not
evidence you can trade it net of costs. I'd read the net line as a stress test, not a final cost
estimate — the proxy is monthly high-low ranges, and the next step is a cleaner daily spread."

→

## Slide 27 — Mathematical and numerical checks

SAY: "Because these are all computed objects, I validated them rather than trusting them:
posterior normalization and the derivative identities, the Gaussian precision update against its
closed form, the obstacle solver's terminal, symmetry, and monotonicity conditions,
finite-difference diagnostics to machine precision, and grid convergence. The point is that the
project sits directly on the course material — filtering, generators, dynamic programming, and
variational inequalities with free boundaries."

→

## Slide 28 — What is novel here?

SAY: "What's new is the combination: the EKV filtering state used for a validation *decision*
rather than estimation; an economic terminal payoff; an explicit null-alpha state with a
false-discovery penalty; and a placebo-calibrated comparison against fixed-horizon tests. The
one-line conclusion: alpha validation is a sequential admission problem, not a fixed-date
significance test."

→

## Slide 29 — Final takeaways

SAY: "To close, the whole arc in one breath: EKV's filter gives the learning state; an economic
reward turns estimation into an admission decision; the Gaussian prior can't see a dead signal,
so a three-state prior with a false-discovery penalty fixes that; and on real, placebo-calibrated
data the rule admits earlier than fixed horizons at matched power."

→

## Slide 30 — References

SAY: "This builds on Ekström, Karatzas, and Vaicenavicius for the filtering and stopping theory,
and on Chen–Zimmermann and Chen–Velikov for the anomaly data and the transaction-cost side.
Thank you — I'm happy to take questions."

---

## Number tags (memorize — these live close together and are easy to swap)

- **90.1%** — in-model *simulation*, Gaussian rule at true alpha = 0 (slide 17). NOT real data.
- **72.9% / 27.1%** — *real* vs sign-flipped *placebo* activation at η = 0.04 (slides 24–25);
  i.e. 140/192 vs 52/192.
- **47.1% → 39.8%** — null activation, matched Gaussian vs three-state, *simulation* (slide 22).
- **0.545** — activation-weighted holdout for the sequential rule (slide 25). This is *not* the
  0.747%/month conditional holdout (a different column); don't quote them interchangeably.
- **1.24·dt, 0.10** — Brownian-scaling diagnostics: increment variance ratio and median
  |lag-1 autocorr| (slide 24).

## Q&A cheat sheet (the questions the dry-runs flagged as most likely)

1. **"Derive the variance-decay identity."** Posterior variance equals mean-squared error
   (tower property). X̂ is a martingale, dX̂ = Ψ dW̃, so its quadratic variation is Ψ²; the law
   of total variance then gives E[Ψ(τ)] = Var(X) − E[∫₀^τ Ψ² ds].

2. **"Your VI has no −ρV term; the course's does."** Calendar-time convention: discounting is
   inside the obstacle e^{−ρt}Φ, so no −ρV. The current-time formulation e^{−ρ(τ−t)} would put
   −ρV in the continuation PDE. Same problem, equivalent.

3. **"Is your finance boundary optimal — does EKV's theorem cover you?"** No. EKV prove
   optimality for the squared-error loss only. Changing the reward gives a *new* obstacle
   problem; I solve the VI numerically and validate it (terminal, symmetry, monotonicity, grid
   convergence). I borrow their state process, not their optimality result.

4. **"Real returns aren't IID Brownian."** Yes, approximately — increment variance ≈ 1.24·dt,
   median |lag-1 autocorr| ≈ 0.10. It's a reasonable first approximation, and it's exactly why I
   add the WRDS stock-level experiment.

5. **"Is sign-flip a fair 'dead-signal' null?"** It keeps the volatility and kills *persistent
   direction* — precisely what the rule is meant to detect, which is why placebo activation falls
   faster in η than real. It isn't ground truth (I have none on real data), so I also ran the
   in-model simulation where θ = 0 is genuinely dead.

6. **"Why three states, and why a ≈ 0.387?"** Minimal model with an explicit null atom and two
   directional states. a = √(0.045/(1−p₀)) ≈ 0.387 is pinned by matching the prior second moment
   (0.045) to the Gaussian benchmark, so the comparison is fair.

7. **"Are σ_r² and q commensurate (slide 14)?"** Predictive-risk approximation:
   Var(r | F) = σ_r² + q — return noise plus parameter uncertainty about the drift — in the same
   annualized evidence units.

8. **"WRDS: all three die under costs — does the method find anything tradeable?"** The net result
   is a deliberately harsh stress test (high-low range proxy removing 5–13%/month), not a real
   cost estimate. The contribution there is separating "exists statistically" from "tradeable net
   of costs." Next step: a clean daily effective-spread cost (Chen–Velikov).

9. **"Where is smooth pasting — did you verify it?"** I keep the variational inequality as the
   primary object because the time-dependent obstacle can kink; I don't impose smooth pasting by
   hand, I read the boundary off the VI solution.
