#!/usr/bin/env node

import fs from "node:fs/promises";
import path from "node:path";
import { spawnSync } from "node:child_process";

import {
  createSlideContext,
  ensureArtifactToolWorkspace,
  importArtifactTool,
  saveBlobToFile,
} from "/Users/ranystephan/.codex/plugins/cache/openai-primary-runtime/presentations/26.521.10419/skills/presentations/scripts/artifact_tool_utils.mjs";

const REPO = path.resolve(path.dirname(new URL(import.meta.url).pathname), "..");
const PRESENTATION_DIR = path.join(REPO, "presentation");
const WORKSPACE = path.join(REPO, "outputs", "manual-mse342-slides", "presentations", "sequential-tradeability");
const PREVIEW_DIR = path.join(PRESENTATION_DIR, "preview");
const OUTPUT_DIR = path.join(PRESENTATION_DIR, "output");
const QA_DIR = path.join(PRESENTATION_DIR, "qa");
const FINAL_PPTX = path.join(OUTPUT_DIR, "sequential-tradeability-alpha-signals.pptx");
const CONTACT_SHEET = path.join(PREVIEW_DIR, "contact-sheet.png");
const SKILL_DIR = "/Users/ranystephan/.codex/plugins/cache/openai-primary-runtime/presentations/26.521.10419/skills/presentations";

const C = {
  bg: "#07111F",
  bg2: "#0D1B2D",
  ink: "#F8FAFC",
  muted: "#AAB7C8",
  dim: "#607086",
  blue: "#4BA3FF",
  green: "#4DD4A1",
  red: "#F05263",
  amber: "#F7B955",
  panel: "#101F34",
  panel2: "#142842",
  line: "#263A56",
  paper: "#F8FAFC",
  darkText: "#0B1220",
};

const fig = (name) => path.join(REPO, "figures", name);
const formula = (name) => path.join(PRESENTATION_DIR, "assets", "formulas", `${name}.png`);

function rect(ctx, slide, x, y, w, h, fill, line = "#00000000", name) {
  return ctx.addShape(slide, {
    left: x,
    top: y,
    width: w,
    height: h,
    fill,
    line: { style: "solid", fill: line, width: line === "#00000000" ? 0 : 1 },
    name,
  });
}

function text(ctx, slide, value, x, y, w, h, options = {}) {
  return ctx.addText(slide, {
    text: value,
    left: x,
    top: y,
    width: w,
    height: h,
    fontSize: options.size ?? 24,
    color: options.color ?? C.ink,
    bold: options.bold ?? false,
    typeface: options.face ?? "Aptos",
    align: options.align ?? "left",
    valign: options.valign ?? "top",
    insets: options.insets ?? { left: 0, right: 0, top: 0, bottom: 0 },
    name: options.name,
  });
}

function bg(ctx, slide) {
  rect(ctx, slide, 0, 0, 1280, 720, C.bg);
  rect(ctx, slide, 0, 0, 1280, 8, C.blue);
}

function kicker(ctx, slide, label, n) {
  rect(ctx, slide, 64, 46, 16, 16, C.blue, "#00000000", `kicker-${n}-marker`);
  text(ctx, slide, label.toUpperCase(), 92, 40, 360, 30, {
    size: 14,
    color: C.muted,
    bold: true,
    name: `kicker-${n}-label`,
    valign: "middle",
  });
}

function title(ctx, slide, claim, sub) {
  text(ctx, slide, claim, 64, 86, 760, 120, {
    size: 44,
    bold: true,
    face: "Aptos Display",
  });
  if (sub) {
    text(ctx, slide, sub, 66, 198, 680, 58, { size: 20, color: C.muted });
  }
}

function footer(ctx, slide, n) {
  text(ctx, slide, "MSE342  |  Sequential Tradeability Testing", 64, 682, 520, 20, {
    size: 12,
    color: C.dim,
  });
  text(ctx, slide, String(n).padStart(2, "0"), 1160, 676, 56, 26, {
    size: 14,
    color: C.dim,
    align: "right",
  });
}

async function image(ctx, slide, source, x, y, w, h, options = {}) {
  return ctx.addImage(slide, {
    path: source,
    left: x,
    top: y,
    width: w,
    height: h,
    fit: options.fit ?? "contain",
    alt: options.alt ?? "",
  });
}

async function formulaImg(ctx, slide, name, x, y, w, h) {
  return image(ctx, slide, formula(name), x, y, w, h, { fit: "contain", alt: `${name} formula` });
}

function bullet(ctx, slide, value, x, y, w, options = {}) {
  rect(ctx, slide, x, y + 9, 8, 8, options.color ?? C.blue);
  text(ctx, slide, value, x + 22, y, w - 22, 54, {
    size: options.size ?? 20,
    color: options.textColor ?? C.ink,
  });
}

function metric(ctx, slide, value, label, x, y, w, color) {
  text(ctx, slide, value, x, y, w, 54, { size: 42, bold: true, color });
  text(ctx, slide, label, x, y + 52, w, 46, { size: 15, color: C.muted });
}

function panel(ctx, slide, x, y, w, h, fill = C.panel) {
  rect(ctx, slide, x, y, w, h, fill, C.line);
}

async function slide01(presentation, ctx) {
  const slide = presentation.slides.add();
  bg(ctx, slide);
  text(ctx, slide, "Sequential\nTradeability Testing", 64, 88, 780, 160, {
    size: 62,
    bold: true,
    face: "Aptos Display",
  });
  text(ctx, slide, "When should a quant team stop researching an alpha and actually trade it?", 70, 284, 780, 72, {
    size: 26,
    color: C.muted,
  });
  await formulaImg(ctx, slide, "value", 552, 386, 650, 96);
  metric(ctx, slide, "continue", "research has option value", 70, 506, 250, C.blue);
  metric(ctx, slide, "activate", "trade after costs and uncertainty", 382, 506, 330, C.green);
  metric(ctx, slide, "reject", "avoid false discoveries", 780, 506, 260, C.red);
  text(ctx, slide, "Rany Stephan", 70, 638, 300, 28, { size: 18, color: C.muted });
  footer(ctx, slide, 1);
  return slide;
}

async function slide02(presentation, ctx) {
  const slide = presentation.slides.add();
  bg(ctx, slide);
  kicker(ctx, slide, "the question", 2);
  title(ctx, slide, "An alpha is not validated at a date. It is admitted through time.", "Fixed horizons answer: is it significant now? A research desk needs: is it worth trading yet?");
  const steps = [
    ["candidate signal", "many ideas are weak or dead", C.dim],
    ["evidence arrives", "returns, ICs, spreads, decay", C.blue],
    ["decision", "continue, activate, reject", C.amber],
    ["live capital", "value must survive costs", C.green],
  ];
  steps.forEach((s, i) => {
    const x = 72 + i * 286;
    panel(ctx, slide, x, 344, 220, 146, C.panel2);
    text(ctx, slide, s[0], x + 22, 372, 176, 34, { size: 25, bold: true, color: s[2] });
    text(ctx, slide, s[1], x + 22, 420, 176, 56, { size: 15, color: C.muted });
    if (i < steps.length - 1) {
      text(ctx, slide, ">", x + 236, 390, 26, 36, { size: 30, color: C.dim, bold: true });
    }
  });
  bullet(ctx, slide, "Waiting can be valuable, because tomorrow's evidence may change the decision.", 106, 550, 500);
  bullet(ctx, slide, "Waiting is also costly, because alpha decays and research capacity is scarce.", 676, 550, 500, { color: C.red });
  footer(ctx, slide, 2);
  return slide;
}

async function slide03(presentation, ctx) {
  const slide = presentation.slides.add();
  bg(ctx, slide);
  kicker(ctx, slide, "paper anchor", 3);
  title(ctx, slide, "EKV gives the state: learn an unknown drift from noisy evidence.", "The mathematical object is a posterior belief, updated continuously as evidence accumulates.");
  await formulaImg(ctx, slide, "obs", 78, 292, 440, 84);
  await formulaImg(ctx, slide, "posterior", 640, 292, 520, 84);
  rect(ctx, slide, 560, 328, 46, 3, C.dim);
  text(ctx, slide, "evidence path", 94, 400, 300, 30, { size: 16, color: C.muted });
  text(ctx, slide, "posterior belief", 694, 400, 300, 30, { size: 16, color: C.muted });
  for (let i = 0; i < 9; i += 1) {
    const x = 130 + i * 34;
    const y = 540 - Math.sin(i / 1.3) * 44 - i * 6;
    rect(ctx, slide, x, y, 20, 20, i > 4 ? C.green : C.blue);
  }
  text(ctx, slide, "Noisy observations reveal signal strength only gradually.", 92, 610, 500, 44, { size: 21 });
  panel(ctx, slide, 728, 506, 326, 94);
  text(ctx, slide, "State variables", 752, 526, 200, 26, { size: 18, color: C.muted, bold: true });
  text(ctx, slide, "posterior mean m\nposterior variance q", 752, 556, 246, 58, { size: 24 });
  footer(ctx, slide, 3);
  return slide;
}

async function slide04(presentation, ctx) {
  const slide = presentation.slides.add();
  bg(ctx, slide);
  kicker(ctx, slide, "core identity", 4);
  title(ctx, slide, "Uncertainty is not only error. It is also the speed of learning.", "This is the EKV insight that makes optimal stopping natural.");
  await formulaImg(ctx, slide, "filter", 116, 262, 820, 82);
  await formulaImg(ctx, slide, "variance_decay", 108, 420, 984, 86);
  panel(ctx, slide, 884, 224, 262, 116);
  text(ctx, slide, "Intuition", 908, 244, 180, 28, { size: 18, color: C.muted, bold: true });
  text(ctx, slide, "High posterior variance means bad current estimate, but also faster possible learning.", 908, 276, 202, 64, { size: 18 });
  bullet(ctx, slide, "Observing costs c per unit time.", 128, 558, 380, { color: C.red });
  bullet(ctx, slide, "Learning reduces variance at rate Psi squared.", 622, 558, 470, { color: C.green });
  footer(ctx, slide, 4);
  return slide;
}

async function slide05(presentation, ctx) {
  const slide = presentation.slides.add();
  bg(ctx, slide);
  kicker(ctx, slide, "the move", 5);
  title(ctx, slide, "We keep EKV's filtering state, but change the stopping reward.", "EKV asks when to estimate accurately. We ask when to admit a signal into production.");
  const rows = [
    ["X", "unknown alpha strength"],
    ["Y_t", "cumulative normalized evidence"],
    ["m_t", "posterior mean alpha"],
    ["q_t", "posterior uncertainty and learning speed"],
    ["tau", "activation or rejection time"],
  ];
  panel(ctx, slide, 96, 268, 468, 292);
  text(ctx, slide, "EKV estimation", 122, 290, 220, 30, { size: 20, color: C.blue, bold: true });
  text(ctx, slide, "minimize statistical error\nplus observation cost", 122, 334, 320, 80, { size: 30, bold: true });
  await formulaImg(ctx, slide, "ekv_running", 108, 450, 420, 56);
  panel(ctx, slide, 666, 268, 468, 292);
  text(ctx, slide, "Our admission problem", 692, 290, 260, 30, { size: 20, color: C.green, bold: true });
  rows.forEach((row, i) => {
    text(ctx, slide, row[0], 700, 334 + i * 40, 150, 28, { size: 22, bold: true, color: C.ink });
    text(ctx, slide, row[1], 842, 334 + i * 40, 246, 34, { size: 18, color: C.muted });
  });
  footer(ctx, slide, 5);
  return slide;
}

async function slide06(presentation, ctx) {
  const slide = presentation.slides.add();
  bg(ctx, slide);
  kicker(ctx, slide, "economic payoff", 6);
  title(ctx, slide, "Tradeability means alpha must clear uncertainty and costs.", "A large posterior mean is not enough if it is uncertain, volatile, or expensive to trade.");
  await formulaImg(ctx, slide, "payoff_objective", 100, 258, 760, 70);
  await formulaImg(ctx, slide, "payoff", 100, 382, 820, 82);
  await formulaImg(ctx, slide, "threshold", 100, 520, 730, 72);
  panel(ctx, slide, 938, 256, 220, 310);
  text(ctx, slide, "What changes the threshold?", 962, 280, 166, 50, { size: 19, bold: true });
  bullet(ctx, slide, "Higher q means less confidence.", 966, 350, 168, { size: 16, color: C.amber });
  bullet(ctx, slide, "Higher sigma means noisier returns.", 966, 420, 168, { size: 16, color: C.blue });
  bullet(ctx, slide, "Higher kappa means larger hurdle.", 966, 490, 168, { size: 16, color: C.red });
  footer(ctx, slide, 6);
  return slide;
}

async function slide07(presentation, ctx) {
  const slide = presentation.slides.add();
  bg(ctx, slide);
  kicker(ctx, slide, "stopping rule", 7);
  title(ctx, slide, "The decision boundary is an obstacle problem.", "At every posterior state, compare immediate tradeability with the value of one more observation.");
  await formulaImg(ctx, slide, "value", 70, 242, 1010, 80);
  await formulaImg(ctx, slide, "vi", 86, 358, 1030, 88);
  const labels = [
    ["stop: reject", "payoff is zero", C.dim],
    ["continue", "learning option remains", C.blue],
    ["stop: activate", "economic payoff dominates", C.green],
  ];
  labels.forEach((row, i) => {
    panel(ctx, slide, 120 + i * 350, 526, 278, 82, C.panel2);
    rect(ctx, slide, 144 + i * 350, 552, 18, 18, row[2]);
    text(ctx, slide, row[0], 176 + i * 350, 542, 180, 28, { size: 20, bold: true });
    text(ctx, slide, row[1], 176 + i * 350, 572, 180, 26, { size: 15, color: C.muted });
  });
  footer(ctx, slide, 7);
  return slide;
}

async function slide08(presentation, ctx) {
  const slide = presentation.slides.add();
  bg(ctx, slide);
  kicker(ctx, slide, "first boundary", 8);
  title(ctx, slide, "The boundary creates a real continuation region.", "Blue is the option value of learning. Red and gray are the two ways to stop.");
  await image(ctx, slide, fig("gaussian_tradeability_boundary.png"), 134, 228, 1000, 420, { fit: "contain" });
  text(ctx, slide, "This is a central zoom. The full computational grid is wider.", 146, 650, 640, 24, {
    size: 15,
    color: C.muted,
  });
  footer(ctx, slide, 8);
  return slide;
}

async function slide09(presentation, ctx) {
  const slide = presentation.slides.add();
  bg(ctx, slide);
  kicker(ctx, slide, "stress test", 9);
  title(ctx, slide, "The Gaussian benchmark exposes a false-discovery problem.", "This failure is useful: it tells us what the prior is missing.");
  await image(ctx, slide, fig("simulation_policy_comparison.png"), 84, 246, 720, 410, { fit: "contain" });
  panel(ctx, slide, 850, 286, 276, 212);
  text(ctx, slide, "What we learn", 878, 312, 190, 30, { size: 22, bold: true, color: C.blue });
  text(ctx, slide, "Strong live alphas: dynamic boundary wins.\n\nDead alpha: Gaussian prior keeps looking for a nonzero drift.", 878, 354, 218, 122, { size: 20 });
  metric(ctx, slide, "90.1%", "null paths activated by Gaussian dynamic rule", 868, 532, 260, C.red);
  footer(ctx, slide, 9);
  return slide;
}

async function slide10(presentation, ctx) {
  const slide = presentation.slides.add();
  bg(ctx, slide);
  kicker(ctx, slide, "dead state", 10);
  title(ctx, slide, "Most researched signals are not just weak. Many are dead.", "So the prior should put mass on exactly zero alpha.");
  await formulaImg(ctx, slide, "three_state", 72, 236, 460, 72);
  await formulaImg(ctx, slide, "three_state_posterior", 72, 356, 650, 82);
  await formulaImg(ctx, slide, "dead_payoff", 72, 506, 760, 86);
  const states = [
    ["-a", "live short", C.red],
    ["0", "dead", C.dim],
    ["+a", "live long", C.green],
  ];
  states.forEach((s, i) => {
    panel(ctx, slide, 872, 246 + i * 112, 210, 78, C.panel2);
    text(ctx, slide, s[0], 896, 260 + i * 112, 54, 42, { size: 36, bold: true, color: s[2] });
    text(ctx, slide, s[1], 966, 274 + i * 112, 100, 24, { size: 20 });
  });
  footer(ctx, slide, 10);
  return slide;
}

async function slide11(presentation, ctx) {
  const slide = presentation.slides.add();
  bg(ctx, slide);
  kicker(ctx, slide, "controlled validation", 11);
  title(ctx, slide, "The dead/alive rule buys lower false activation, not free dominance.", "That tradeoff is exactly what a serious admission rule should expose.");
  await image(ctx, slide, fig("dead_alive_policy_comparison.png"), 58, 228, 820, 410, { fit: "contain" });
  metric(ctx, slide, "47.1% -> 39.8%", "false activation at theta = 0", 918, 286, 250, C.green);
  metric(ctx, slide, "-0.00560 -> -0.00513", "null realized value improves", 918, 428, 270, C.blue);
  text(ctx, slide, "The cost is lower live-alpha power. That is a power versus false-discovery tradeoff.", 918, 570, 260, 58, { size: 18, color: C.muted });
  footer(ctx, slide, 11);
  return slide;
}

async function slide12(presentation, ctx) {
  const slide = presentation.slides.add();
  bg(ctx, slide);
  kicker(ctx, slide, "real data calibration", 12);
  title(ctx, slide, "OSAP placebos fall faster as the false-discovery penalty rises.", "The placebo keeps volatility structure but destroys persistent directional drift.");
  await formulaImg(ctx, slide, "evidence", 84, 220, 570, 58);
  await image(ctx, slide, fig("osap_penalty_sweep.png"), 96, 306, 700, 340, { fit: "contain" });
  metric(ctx, slide, "72.9%", "real OSAP activation at eta = 0.04", 850, 320, 260, C.green);
  metric(ctx, slide, "27.1%", "sign-flip placebo activation", 850, 458, 260, C.red);
  footer(ctx, slide, 12);
  return slide;
}

async function slide13(presentation, ctx) {
  const slide = presentation.slides.add();
  bg(ctx, slide);
  kicker(ctx, slide, "sequential or fixed horizon?", 13);
  title(ctx, slide, "The sequential rule is not just a fixed-horizon test in disguise.", "It reaches near long-horizon admission power earlier, with better post-decision performance.");
  await image(ctx, slide, fig("osap_static_benchmark_comparison.png"), 74, 226, 760, 400, { fit: "contain" });
  metric(ctx, slide, "47.9 mo", "average sequential decision time", 874, 278, 250, C.blue);
  metric(ctx, slide, "0.545", "activation-weighted holdout return", 874, 414, 250, C.green);
  text(ctx, slide, "Compared with 60, 120, and 240 month fixed tests: 0.500, 0.531, 0.444.", 874, 552, 270, 54, { size: 17, color: C.muted });
  footer(ctx, slide, 13);
  return slide;
}

async function slide14(presentation, ctx) {
  const slide = presentation.slides.add();
  bg(ctx, slide);
  kicker(ctx, slide, "real implementation", 14);
  title(ctx, slide, "WRDS confirms the practical point: gross alpha is not enough.", "A usable admission rule must survive uncertainty and trading frictions.");
  panel(ctx, slide, 96, 266, 456, 246, C.panel2);
  text(ctx, slide, "Gross reconstructed signals", 126, 296, 300, 30, { size: 22, bold: true, color: C.green });
  metric(ctx, slide, "3 / 3", "activate in standard direction", 130, 344, 250, C.green);
  metric(ctx, slide, "1.227%", "monthly holdout among activated", 130, 446, 300, C.ink);
  panel(ctx, slide, 640, 266, 456, 246, C.panel2);
  text(ctx, slide, "Net range-cost stress", 670, 296, 300, 30, { size: 22, bold: true, color: C.red });
  metric(ctx, slide, "0 / 3", "standard-direction activation", 674, 344, 270, C.red);
  text(ctx, slide, "The range-cost proxy is deliberately harsh, so this is a stress test rather than a final cost model.", 674, 446, 330, 56, { size: 19 });
  text(ctx, slide, "Final thesis: alpha validation is a sequential real-options problem, not a fixed-date hypothesis test.", 104, 584, 920, 48, { size: 28, bold: true, color: C.ink });
  footer(ctx, slide, 14);
  return slide;
}

const SLIDES = [
  slide01,
  slide02,
  slide03,
  slide04,
  slide05,
  slide06,
  slide07,
  slide08,
  slide09,
  slide10,
  slide11,
  slide12,
  slide13,
  slide14,
];

async function makeContactSheet(previewPaths) {
  const script = path.join(SKILL_DIR, "scripts", "make_contact_sheet.py");
  const result = spawnSync("python3", [script, "--output", CONTACT_SHEET, ...previewPaths], {
    encoding: "utf8",
  });
  if (result.status !== 0) {
    throw new Error([result.stdout, result.stderr].filter(Boolean).join("\n"));
  }
}

async function main() {
  await fs.mkdir(PREVIEW_DIR, { recursive: true });
  await fs.mkdir(OUTPUT_DIR, { recursive: true });
  await fs.mkdir(QA_DIR, { recursive: true });
  await ensureArtifactToolWorkspace(WORKSPACE);
  const artifact = await importArtifactTool(WORKSPACE);
  const { Presentation, PresentationFile } = artifact;
  const presentation = Presentation.create({ slideSize: { width: 1280, height: 720 } });
  const previewPaths = [];

  for (let i = 0; i < SLIDES.length; i += 1) {
    const ctx = createSlideContext(artifact, {
      slideSize: { width: 1280, height: 720 },
      slideNumber: i + 1,
      outputDir: OUTPUT_DIR,
      assetDir: path.join(PRESENTATION_DIR, "assets"),
      workspaceDir: WORKSPACE,
    });
    const slide = await SLIDES[i](presentation, ctx);
    const preview = await presentation.export({ slide, format: "png", scale: 1 });
    const previewPath = path.join(PREVIEW_DIR, `slide-${String(i + 1).padStart(2, "0")}.png`);
    await saveBlobToFile(preview, previewPath);
    previewPaths.push(previewPath);
  }

  const pptx = await PresentationFile.exportPptx(presentation);
  await pptx.save(FINAL_PPTX);
  await makeContactSheet(previewPaths);
  const stat = await fs.stat(FINAL_PPTX);
  await fs.writeFile(
    path.join(QA_DIR, "build-manifest.json"),
    JSON.stringify(
      {
        output: FINAL_PPTX,
        bytes: stat.size,
        slideCount: SLIDES.length,
        previewDir: PREVIEW_DIR,
        contactSheet: CONTACT_SHEET,
      },
      null,
      2,
    ),
  );
  console.log(JSON.stringify({ output: FINAL_PPTX, slideCount: SLIDES.length, bytes: stat.size }, null, 2));
}

main().catch((error) => {
  console.error(error.stack || error.message || String(error));
  process.exit(1);
});
