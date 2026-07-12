const pptxgen = require("pptxgenjs");
const path = require("path");

const A = path.join(__dirname, "assets");

// ---- Palette ----
const NAVY = "1B2A41";
const NAVY_LIGHT = "2E4258";
const OFFWHITE = "F4F6F8";
const WHITE = "FFFFFF";
const AMBER = "F5A623";
const BLUE = "2E86AB";
const GREEN = "5B8C5A";
const TEXT_DARK = "1B2A41";
const TEXT_MUTED = "5A6B7A";
const ICE = "CADCFC";

const FONT_HEAD = "Cambria";
const FONT_BODY = "Calibri";

let pres = new pptxgen();
pres.layout = "LAYOUT_WIDE"; // 13.333 x 7.5 in
const PW = 13.333, PH = 7.5;

function bgSlide(color) {
  const s = pres.addSlide();
  s.background = { color };
  return s;
}

function iconCircle(slide, x, y, d, fillColor, label, labelColor, fontSize) {
  slide.addShape("ellipse", { x, y, w: d, h: d, fill: { color: fillColor }, line: { type: "none" } });
  slide.addText(label, {
    x, y, w: d, h: d, align: "center", valign: "middle",
    fontFace: FONT_HEAD, fontSize: fontSize || 20, bold: true, color: labelColor || WHITE,
    margin: 0,
  });
}

function footer(slide, pageNum, dark) {
  slide.addText(`Vision Robustness Under Distortion  \u2022  ${pageNum}`, {
    x: 0.5, y: PH - 0.45, w: PW - 1, h: 0.3, fontFace: FONT_BODY, fontSize: 9,
    color: dark ? "8A9BB0" : "A0ADB8", align: "right", margin: 0,
  });
}

// ============================================================
// Slide 1 - Title
// ============================================================
{
  const s = bgSlide(NAVY);
  // faint decorative circles (motif), not a stripe
  s.addShape("ellipse", { x: 10.6, y: -1.4, w: 4.5, h: 4.5, fill: { color: NAVY_LIGHT }, line: { type: "none" } });
  s.addShape("ellipse", { x: -1.6, y: 4.8, w: 3.6, h: 3.6, fill: { color: NAVY_LIGHT }, line: { type: "none" } });

  s.addText("ROBUSTNESS OF VISION ALGORITHMS", {
    x: 0.9, y: 2.35, w: 11.5, h: 1.0, fontFace: FONT_HEAD, fontSize: 40, bold: true,
    color: WHITE, align: "left", margin: 0,
  });
  s.addText("Under Real-World Image Distortions", {
    x: 0.9, y: 3.15, w: 11.5, h: 0.7, fontFace: FONT_HEAD, fontSize: 26, bold: false,
    color: AMBER, align: "left", margin: 0,
  });
  s.addText("Classical & deep-learning vision tasks, evaluated on real driving imagery \u2014\ncompression, low-light, and motion-blur distortions, classical restoration,\nand YOLO fine-tuning.", {
    x: 0.9, y: 4.05, w: 9.5, h: 1.0, fontFace: FONT_BODY, fontSize: 15,
    color: ICE, align: "left", margin: 0, lineSpacingMultiple: 1.25,
  });
  s.addText("Digital Image Processing & Computer Vision \u2014 Course Project", {
    x: 0.9, y: 6.55, w: 9, h: 0.4, fontFace: FONT_BODY, fontSize: 12,
    color: "8A9BB0", align: "left", margin: 0,
  });
}

// ============================================================
// Slide 2 - Project at a glance
// ============================================================
{
  const s = bgSlide(OFFWHITE);
  s.addText("Project at a Glance", {
    x: 0.7, y: 0.45, w: 11.9, h: 0.7, fontFace: FONT_HEAD, fontSize: 32, bold: true, color: TEXT_DARK, margin: 0,
  });

  const rows = [
    ["1", "Dataset", "BDD100K \u2014 150-image subset with real detection ground truth", BLUE],
    ["2", "Tasks", "Edge/Corner detection, Line detection (both low-level, classical) + Object detection (high-level, DL)", GREEN],
    ["3", "Distortions", "Compression, Low-light, Motion blur \u2014 all directly relevant to dashcam footage", AMBER],
    ["4", "Severity", "5 calibrated levels per distortion, measured in PSNR (dB)", BLUE],
    ["5", "Restoration", "Bilateral deblocking, Gamma+CLAHE, Wiener deconvolution \u2014 matched to each distortion's cause", GREEN],
    ["6", "Fine-tuning", "YOLOv8n only \u2014 the classical tasks have no trainable weights", AMBER],
  ];
  let y = 1.5;
  const rowH = 0.88;
  rows.forEach(([num, label, desc, color]) => {
    iconCircle(s, 0.75, y, 0.55, color, num, WHITE, 20);
    s.addText(label, {
      x: 1.55, y: y - 0.06, w: 3.0, h: 0.5, fontFace: FONT_HEAD, fontSize: 16, bold: true,
      color: TEXT_DARK, margin: 0, valign: "top",
    });
    s.addText(desc, {
      x: 4.65, y: y - 0.06, w: 8.0, h: 0.7, fontFace: FONT_BODY, fontSize: 13,
      color: TEXT_MUTED, margin: 0, valign: "top", lineSpacingMultiple: 1.15,
    });
    y += rowH;
  });
  footer(s, 2);
}

// ============================================================
// Slide 3 - Pipeline architecture
// ============================================================
{
  const s = bgSlide(OFFWHITE);
  s.addText("Four-Stage Pipeline", {
    x: 0.7, y: 0.45, w: 11.9, h: 0.7, fontFace: FONT_HEAD, fontSize: 32, bold: true, color: TEXT_DARK, margin: 0,
  });
  s.addText("Every image passes through all four stages, evaluated on all 3 tasks", {
    x: 0.7, y: 1.05, w: 11, h: 0.4, fontFace: FONT_BODY, fontSize: 14, color: TEXT_MUTED, margin: 0,
  });

  const stages = [
    ["1", "CLEAN", "Baseline evaluation\non the original image", BLUE],
    ["2", "DISTORTED", "3 distortions \u00d7 5 severity\nlevels applied", AMBER],
    ["3", "RESTORED", "Classical restoration\napplied to distorted image", GREEN],
    ["4", "FINE-TUNED", "YOLOv8n fine-tuned on\ndistorted images (detection only)", NAVY],
  ];
  const boxW = 2.55, gap = 0.45, startX = 0.85, y = 2.6, boxH = 2.6;
  stages.forEach(([num, title, desc, color], i) => {
    const x = startX + i * (boxW + gap);
    s.addShape("roundRect", {
      x, y, w: boxW, h: boxH, rectRadius: 0.12,
      fill: { color: WHITE }, line: { color: "DDE3E9", width: 1 },
      shadow: { type: "outer", color: "1B2A41", opacity: 0.12, blur: 8, offset: 3, angle: 90 },
    });
    iconCircle(s, x + boxW / 2 - 0.35, y + 0.3, 0.7, color, num, WHITE, 24);
    s.addText(title, {
      x: x + 0.1, y: y + 1.15, w: boxW - 0.2, h: 0.4, fontFace: FONT_HEAD, fontSize: 15, bold: true,
      color: TEXT_DARK, align: "center", margin: 0,
    });
    s.addText(desc, {
      x: x + 0.15, y: y + 1.6, w: boxW - 0.3, h: 0.9, fontFace: FONT_BODY, fontSize: 11.5,
      color: TEXT_MUTED, align: "center", margin: 0, lineSpacingMultiple: 1.2,
    });
    if (i < stages.length - 1) {
      s.addText("\u2192", {
        x: x + boxW + 0.05, y: y + 0.9, w: gap - 0.1, h: 0.6, fontFace: FONT_BODY, fontSize: 26,
        color: "B7C2CC", align: "center", margin: 0,
      });
    }
  });

  s.addText("100 images \u00d7 (1 clean + 3 distortions \u00d7 5 levels \u00d7 2 stages) \u2192 1,600 image variants across the pipeline", {
    x: 0.85, y: 5.65, w: 11.6, h: 0.4, fontFace: FONT_BODY, fontSize: 12.5, italic: true,
    color: TEXT_MUTED, align: "center", margin: 0,
  });
  footer(s, 3);
}

// ============================================================
// Slide 4 - Dataset
// ============================================================
{
  const s = bgSlide(OFFWHITE);
  s.addText("Dataset: BDD100K", {
    x: 0.7, y: 0.45, w: 11.9, h: 0.7, fontFace: FONT_HEAD, fontSize: 32, bold: true, color: TEXT_DARK, margin: 0,
  });

  s.addText("Why BDD100K", {
    x: 0.7, y: 1.35, w: 5.6, h: 0.4, fontFace: FONT_HEAD, fontSize: 16, bold: true, color: BLUE, margin: 0,
  });
  const whyItems = [
    "Real driving footage with genuine detection ground truth",
    "Cityscapes and KITTI both required login-gated downloads that couldn't be scripted",
    "150-image subset, train split \u2014 small but sufficient for this scope",
  ];
  s.addText(whyItems.map((t, i) => ({ text: t, options: { bullet: { code: "25CF" }, breakLine: i < whyItems.length - 1, color: TEXT_DARK, fontSize: 13.5 } })), {
    x: 0.7, y: 1.8, w: 5.6, h: 1.8, fontFace: FONT_BODY, margin: 0, lineSpacingMultiple: 1.3,
  });

  s.addText("Class distribution (150-image subset)", {
    x: 0.7, y: 3.75, w: 5.6, h: 0.4, fontFace: FONT_HEAD, fontSize: 16, bold: true, color: BLUE, margin: 0,
  });
  const classes = [["car", 1543], ["traffic light", 400], ["truck", 83], ["bus", 43], ["person", 13]];
  const maxV = 1543;
  let cy = 4.25;
  classes.forEach(([name, count]) => {
    const barW = 3.6 * (count / maxV);
    s.addText(name, { x: 0.7, y: cy, w: 1.3, h: 0.32, fontFace: FONT_BODY, fontSize: 11.5, color: TEXT_DARK, margin: 0, valign: "middle" });
    s.addShape("roundRect", { x: 2.05, y: cy + 0.03, w: Math.max(barW, 0.08), h: 0.26, rectRadius: 0.04, fill: { color: AMBER }, line: { type: "none" } });
    s.addText(String(count), { x: 2.05 + Math.max(barW, 0.08) + 0.1, y: cy, w: 0.9, h: 0.32, fontFace: FONT_BODY, fontSize: 11, color: TEXT_MUTED, margin: 0, valign: "middle" });
    cy += 0.42;
  });

  s.addImage({ path: `${A}/detection_overlay.png`, x: 6.65, y: 1.4, w: 6.0, h: 2.36 });
  s.addText("GT sanity check \u2014 predictions (green) vs. ground truth (red), clean vs. severe low-light", {
    x: 6.65, y: 3.8, w: 6.0, h: 0.55, fontFace: FONT_BODY, fontSize: 10.5, italic: true, color: TEXT_MUTED, margin: 0,
  });

  s.addShape("roundRect", {
    x: 6.65, y: 4.55, w: 6.0, h: 1.75, rectRadius: 0.1, fill: { color: WHITE }, line: { color: "DDE3E9", width: 1 },
  });
  s.addText("Known limitation", {
    x: 6.95, y: 4.72, w: 5.4, h: 0.35, fontFace: FONT_HEAD, fontSize: 13, bold: true, color: NAVY, margin: 0,
  });
  s.addText("Zero bicycle/motorcycle/train instances in this sample; heavily skewed toward cars and traffic lights. Per-class metrics for rare classes aren't meaningful at this scale.", {
    x: 6.95, y: 5.1, w: 5.4, h: 1.1, fontFace: FONT_BODY, fontSize: 12, color: TEXT_MUTED, margin: 0, lineSpacingMultiple: 1.25,
  });
  footer(s, 4);
}

// ============================================================
// Slide 5 - Three tasks
// ============================================================
{
  const s = bgSlide(OFFWHITE);
  s.addText("Three Tasks: Low-Level + High-Level", {
    x: 0.7, y: 0.45, w: 11.9, h: 0.7, fontFace: FONT_HEAD, fontSize: 32, bold: true, color: TEXT_DARK, margin: 0,
  });

  const tasks = [
    ["Edge / Corner\nDetection", "LOW-LEVEL \u00b7 CLASSICAL", "cv2.Canny + goodFeaturesToTrack (Shi-Tomasi) + ORB", "No GT exists \u2014 measured against the clean image's own output (edge IoU, corner ratio, ORB match ratio)", BLUE],
    ["Line\nDetection", "LOW-LEVEL \u00b7 CLASSICAL", "cv2.Canny + HoughLinesP", "Replaced an original optical-flow/tracking plan that needed video frames we didn't have", GREEN],
    ["Object\nDetection", "HIGH-LEVEL \u00b7 DEEP LEARNING", "ultralytics YOLOv8n (COCO-pretrained)", "The one task with real ground truth \u2014 scored with proper IoU-matched precision/recall/F1", AMBER],
  ];
  const cardW = 3.75, gap = 0.35, startX = 0.7, y = 1.5, cardH = 5.25;
  tasks.forEach(([title, tag, method, note, color], i) => {
    const x = startX + i * (cardW + gap);
    s.addShape("roundRect", {
      x, y, w: cardW, h: cardH, rectRadius: 0.12, fill: { color: WHITE }, line: { color: "DDE3E9", width: 1 },
      shadow: { type: "outer", color: "1B2A41", opacity: 0.1, blur: 8, offset: 3, angle: 90 },
    });
    iconCircle(s, x + 0.35, y + 0.35, 0.65, color, String(i + 1), WHITE, 22);
    s.addText(title, {
      x: x + 0.3, y: y + 1.15, w: cardW - 0.6, h: 0.75, fontFace: FONT_HEAD, fontSize: 19, bold: true,
      color: TEXT_DARK, margin: 0, lineSpacingMultiple: 1.05, valign: "top",
    });
    s.addShape("roundRect", { x: x + 0.3, y: y + 2.05, w: cardW - 0.6, h: 0.36, rectRadius: 0.18, fill: { color: color }, line: { type: "none" } });
    s.addText(tag, {
      x: x + 0.3, y: y + 2.05, w: cardW - 0.6, h: 0.36, fontFace: FONT_BODY, fontSize: 10.5, bold: true,
      color: WHITE, align: "center", valign: "middle", margin: 0, charSpacing: 1,
    });
    s.addText(method, {
      x: x + 0.3, y: y + 2.65, w: cardW - 0.6, h: 0.85, fontFace: FONT_BODY, fontSize: 12.5, bold: true,
      color: TEXT_DARK, margin: 0, lineSpacingMultiple: 1.2,
    });
    s.addText(note, {
      x: x + 0.3, y: y + 3.55, w: cardW - 0.6, h: 1.55, fontFace: FONT_BODY, fontSize: 11.5, italic: true,
      color: TEXT_MUTED, margin: 0, lineSpacingMultiple: 1.25,
    });
  });
  footer(s, 5);
}

// ============================================================
// Slide 6 - Distortions
// ============================================================
{
  const s = bgSlide(OFFWHITE);
  s.addText("Three Distortions", {
    x: 0.7, y: 0.4, w: 11.9, h: 0.65, fontFace: FONT_HEAD, fontSize: 32, bold: true, color: TEXT_DARK, margin: 0,
  });
  s.addText("Chosen for direct relevance to dashcam/driving conditions \u2014 5 calibrated severity levels each, measured in PSNR (dB)", {
    x: 0.7, y: 0.98, w: 11.9, h: 0.4, fontFace: FONT_BODY, fontSize: 13, color: TEXT_MUTED, margin: 0,
  });

  s.addImage({ path: `${A}/before_after_grid.png`, x: 3.55, y: 1.5, w: 6.25, h: 4.33 });

  const legend = [
    ["Compression", "quality 50\u21923 \u2192 ~39\u219224 dB", BLUE],
    ["Low-light", "brightness -0.2\u2192-0.9 \u2192 ~18\u21927 dB", AMBER],
    ["Motion blur", "kernel 5\u219231px \u2192 ~33\u219225 dB", GREEN],
  ];
  let ly = 1.7;
  legend.forEach(([name, range, color]) => {
    s.addShape("roundRect", { x: 0.7, y: ly, w: 0.28, h: 0.28, rectRadius: 0.06, fill: { color }, line: { type: "none" } });
    s.addText(name, { x: 1.12, y: ly - 0.06, w: 2.2, h: 0.4, fontFace: FONT_HEAD, fontSize: 13.5, bold: true, color: TEXT_DARK, margin: 0 });
    s.addText(range, { x: 0.7, y: ly + 0.32, w: 2.55, h: 0.55, fontFace: FONT_BODY, fontSize: 10.5, color: TEXT_MUTED, margin: 0, lineSpacingMultiple: 1.15 });
    ly += 1.15;
  });

  s.addText("Motion blur uses a known, controlled linear kernel (not random) \u2014 so restoration can be genuine non-blind deconvolution.", {
    x: 0.55, y: 5.35, w: 2.7, h: 1.3, fontFace: FONT_BODY, fontSize: 10.5, italic: true, color: TEXT_MUTED, margin: 0, lineSpacingMultiple: 1.25,
  });
  footer(s, 6);
}

// ============================================================
// Slide 7 - Restoration
// ============================================================
{
  const s = bgSlide(OFFWHITE);
  s.addText("Restoration Methods (Stage 3)", {
    x: 0.7, y: 0.45, w: 11.9, h: 0.7, fontFace: FONT_HEAD, fontSize: 32, bold: true, color: TEXT_DARK, margin: 0,
  });

  const methods = [
    ["Compression", "Bilateral filter on Y (luma) channel only", "Smooths blocking artifacts while preserving color and most edges", BLUE],
    ["Low-light", "Gamma correction (\u03b3=0.35) + CLAHE on L channel", "Lifts shadow detail, then boosts local contrast without blowing out highlights", AMBER],
    ["Motion blur", "Wiener deconvolution using the known blur kernel", "Legitimate non-blind deconvolution \u2014 we control the exact kernel used to create the distortion", GREEN],
  ];
  let y = 1.6;
  methods.forEach(([name, method, why, color]) => {
    s.addShape("roundRect", {
      x: 0.7, y, w: 11.9, h: 1.5, rectRadius: 0.1, fill: { color: WHITE }, line: { color: "DDE3E9", width: 1 },
      shadow: { type: "outer", color: "1B2A41", opacity: 0.08, blur: 6, offset: 2, angle: 90 },
    });
    s.addShape("roundRect", { x: 0.7, y, w: 0.14, h: 1.5, fill: { color: "FFFFFF" }, line: { type: "none" } }); // spacer, no stripe
    iconCircle(s, 1.0, y + 0.45, 0.6, color, name[0], WHITE, 20);
    s.addText(name, {
      x: 1.85, y: y + 0.18, w: 2.6, h: 0.5, fontFace: FONT_HEAD, fontSize: 16, bold: true, color: TEXT_DARK, margin: 0,
    });
    s.addText(method, {
      x: 1.85, y: y + 0.68, w: 2.6, h: 0.7, fontFace: FONT_BODY, fontSize: 11, color: color, bold: true, margin: 0, lineSpacingMultiple: 1.15,
    });
    s.addText(why, {
      x: 4.7, y: y + 0.28, w: 7.6, h: 0.95, fontFace: FONT_BODY, fontSize: 13, color: TEXT_MUTED, margin: 0, valign: "middle", lineSpacingMultiple: 1.25,
    });
    y += 1.7;
  });
  footer(s, 7);
}

// ============================================================
// Slide 8 - Results: Object Detection
// ============================================================
{
  const s = bgSlide(WHITE);
  s.addText("Results: Object Detection Robustness", {
    x: 0.7, y: 0.4, w: 11.9, h: 0.65, fontFace: FONT_HEAD, fontSize: 30, bold: true, color: TEXT_DARK, margin: 0,
  });
  s.addText("Detection F1 vs. real BDD100K ground truth \u2014 solid = distorted, dashed = after restoration", {
    x: 0.7, y: 0.98, w: 11.9, h: 0.4, fontFace: FONT_BODY, fontSize: 13, color: TEXT_MUTED, margin: 0,
  });

  s.addImage({ path: `${A}/det_f1_slide.png`, x: 1.55, y: 1.55, w: 10.2, h: 5.58 });
  footer(s, 8, false);
}

// ============================================================
// Slide 9 - Results: low-level tasks
// ============================================================
{
  const s = bgSlide(WHITE);
  s.addText("Results: Edge/Corner & Line Detection", {
    x: 0.7, y: 0.4, w: 11.9, h: 0.65, fontFace: FONT_HEAD, fontSize: 30, bold: true, color: TEXT_DARK, margin: 0,
  });
  s.addText("Measured against each image's own clean-image output (no external GT available for these low-level tasks)", {
    x: 0.7, y: 0.98, w: 11.9, h: 0.4, fontFace: FONT_BODY, fontSize: 13, color: TEXT_MUTED, margin: 0,
  });
  s.addImage({ path: `${A}/edge_iou_slide.png`, x: 0.35, y: 1.6, w: 6.3, h: 3.45 });
  s.addImage({ path: `${A}/line_iou_slide.png`, x: 6.7, y: 1.6, w: 6.3, h: 3.45 });

  s.addShape("roundRect", { x: 0.7, y: 5.3, w: 11.9, h: 1.4, rectRadius: 0.1, fill: { color: OFFWHITE }, line: { type: "none" } });
  s.addText([
    { text: "Notable: ", options: { bold: true, color: NAVY } },
    { text: "with a properly regularized deconvolution (see Process Highlights), motion-blur restoration genuinely improves edge structure (edge IoU 0.44\u21920.51) \u2014 an earlier, under-regularized version looked better on paper but was actually inventing fake structure.", options: { color: TEXT_MUTED } },
  ], {
    x: 1.0, y: 5.45, w: 11.3, h: 1.1, fontFace: FONT_BODY, fontSize: 13.5, margin: 0, lineSpacingMultiple: 1.3, valign: "middle",
  });
  footer(s, 9);
}

// ============================================================
// Slide 10 - Key findings
// ============================================================
{
  const s = bgSlide(NAVY);
  s.addText("Key Findings", {
    x: 0.7, y: 0.45, w: 11.9, h: 0.7, fontFace: FONT_HEAD, fontSize: 32, bold: true, color: WHITE, margin: 0,
  });

  const findings = [
    ["Clear degradation with severity", "Compression detection F1 falls from 0.45 (mild) to 0.15 (severe) as SNR drops.", AMBER],
    ["Restoration helps most when things are worst", "Deblocking barely changes mild-compression F1, but meaningfully helps at severe levels.", BLUE],
    ["Pixel quality \u2260 downstream performance", "Low-light restoration improves SNR (9.4\u219212.2 dB) but slightly hurts detection F1.", GREEN],
    ["Regularization strength matters as much as the method", "A fixed Wiener deconvolution setting caused ringing artifacts at severe blur; scaling it with severity fixed both the visuals and the numbers (F1 0.32\u21920.36).", AMBER],
    ["Baseline domain gap motivates fine-tuning", "Even on clean images, stock YOLOv8n only reaches F1=0.45 against BDD100K's real GT.", BLUE],
  ];
  let y = 1.4;
  findings.forEach(([title, desc, color], i) => {
    iconCircle(s, 0.75, y, 0.5, color, String(i + 1), NAVY, 17);
    s.addText(title, {
      x: 1.5, y: y - 0.08, w: 10.8, h: 0.4, fontFace: FONT_HEAD, fontSize: 16, bold: true, color: WHITE, margin: 0,
    });
    s.addText(desc, {
      x: 1.5, y: y + 0.33, w: 10.8, h: 0.5, fontFace: FONT_BODY, fontSize: 12.5, color: ICE, margin: 0, lineSpacingMultiple: 1.2,
    });
    y += 1.05;
  });
  footer(s, 10, true);
}

// ============================================================
// Slide 11 - Fine-tuning
// ============================================================
{
  const s = bgSlide(WHITE);
  s.addText("Fine-Tuning (Stage 4)", {
    x: 0.7, y: 0.4, w: 11.9, h: 0.65, fontFace: FONT_HEAD, fontSize: 30, bold: true, color: TEXT_DARK, margin: 0,
  });

  s.addImage({ path: `${A}/finetune_slide.png`, x: 0.6, y: 1.15, w: 6.6, h: 3.81 });

  s.addShape("roundRect", { x: 7.55, y: 1.2, w: 5.1, h: 1.75, rectRadius: 0.1, fill: { color: OFFWHITE }, line: { type: "none" } });
  s.addText("Overall Detection F1", { x: 7.85, y: 1.35, w: 4.5, h: 0.3, fontFace: FONT_BODY, fontSize: 12, color: TEXT_MUTED, margin: 0 });
  s.addText([
    { text: "0.267 ", options: { color: TEXT_MUTED, strike: true, fontSize: 26 } },
    { text: " \u2192  ", options: { color: TEXT_MUTED, fontSize: 22 } },
    { text: "0.261", options: { color: AMBER, bold: true, fontSize: 30 } },
  ], { x: 7.85, y: 1.68, w: 4.5, h: 0.65, fontFace: FONT_HEAD, margin: 0, valign: "middle" });
  s.addText("baseline \u2192 fine-tuned (held-out distorted set)", { x: 7.85, y: 2.4, w: 4.5, h: 0.4, fontFace: FONT_BODY, fontSize: 11, italic: true, color: TEXT_MUTED, margin: 0 });

  s.addShape("roundRect", { x: 7.55, y: 3.15, w: 5.1, h: 2.45, rectRadius: 0.1, fill: { color: WHITE }, line: { color: "DDE3E9", width: 1 } });
  s.addText("Honest result: no clear win", { x: 7.85, y: 3.35, w: 4.5, h: 0.4, fontFace: FONT_HEAD, fontSize: 14.5, bold: true, color: NAVY, margin: 0 });
  s.addText(
    "F1 is essentially flat overall \u2014 small recall gain traded for a precision drop. Results are mixed per distortion (compression improved slightly, low-light got worse).\n\nExpected at this scale: 40\u201350 training images, 5\u20138 epochs, CPU-only \u2014 squarely the regime where catastrophic forgetting can outweigh adaptation.",
    { x: 7.85, y: 3.78, w: 4.5, h: 1.75, fontFace: FONT_BODY, fontSize: 11.5, color: TEXT_MUTED, margin: 0, lineSpacingMultiple: 1.25 }
  );
  footer(s, 11);
}

// ============================================================
// Slide 12 - Process / troubleshooting highlights
// ============================================================
{
  const s = bgSlide(OFFWHITE);
  s.addText("Process Highlights", {
    x: 0.7, y: 0.45, w: 11.9, h: 0.6, fontFace: FONT_HEAD, fontSize: 32, bold: true, color: TEXT_DARK, margin: 0,
  });
  s.addText("Real friction points hit while building this \u2014 documented, not hidden", {
    x: 0.7, y: 1.02, w: 11.9, h: 0.4, fontFace: FONT_BODY, fontSize: 13, color: TEXT_MUTED, margin: 0,
  });

  const items = [
    ["Dataset access", "Cityscapes and KITTI both required login-gated downloads. Switched to BDD100K, downloaded manually, filtered locally.", BLUE],
    ["1GB label file", "BDD100K's full label file (~70k entries) was too big to hand off. A local script filtered it down to the 150 images actually used, in under 1MB.", GREEN],
    ["The fine-tuning caching bug", "First attempt showed byte-identical baseline vs. fine-tuned results \u2014 a cached YOLO model got mutated in-place by .train(). Fixed with two independent model instances.", AMBER],
    ["A visibly broken restoration, caught on review", "Motion-blur restoration showed severe striping at high severity \u2014 a fixed deconvolution regularizer was amplifying noise. Scaling it with severity fixed both the look and the numbers (F1 0.32\u21920.36).", BLUE],
  ];
  let y = 1.5;
  items.forEach(([title, desc, color]) => {
    s.addShape("roundRect", {
      x: 0.7, y, w: 11.9, h: 1.2, rectRadius: 0.1, fill: { color: WHITE }, line: { color: "DDE3E9", width: 1 },
      shadow: { type: "outer", color: "1B2A41", opacity: 0.08, blur: 6, offset: 2, angle: 90 },
    });
    iconCircle(s, 0.95, y + 0.3, 0.5, color, "!", WHITE, 19);
    s.addText(title, { x: 1.7, y: y + 0.1, w: 2.85, h: 1.0, fontFace: FONT_HEAD, fontSize: 13.5, bold: true, color: TEXT_DARK, margin: 0, valign: "top", lineSpacingMultiple: 1.1 });
    s.addText(desc, { x: 4.75, y: y + 0.1, w: 7.55, h: 1.0, fontFace: FONT_BODY, fontSize: 11, color: TEXT_MUTED, margin: 0, valign: "top", lineSpacingMultiple: 1.16 });
    y += 1.35;
  });
  footer(s, 12);
}

// ============================================================
// Slide 13 - Limitations & closing
// ============================================================
{
  const s = bgSlide(NAVY);
  s.addShape("ellipse", { x: 10.9, y: 4.9, w: 4.0, h: 4.0, fill: { color: NAVY_LIGHT }, line: { type: "none" } });

  s.addText("Limitations & Takeaways", {
    x: 0.9, y: 0.6, w: 11.5, h: 0.65, fontFace: FONT_HEAD, fontSize: 30, bold: true, color: WHITE, margin: 0,
  });

  const lims = [
    "Small scale by design (150 images) \u2014 explicitly permitted for this course project; pipeline scales to any N",
    "Class imbalance: zero bicycle/motorcycle/train instances in this sample",
    "Edge/corner and line detection use clean-image-as-reference, not external ground truth",
    "BDD100K \u2192 COCO class mapping is approximate (\u201crider\u201d folds into \u201cperson\u201d; \u201ctraffic sign\u201d dropped)",
    "Fine-tuning is a proof of concept, not a production training run",
  ];
  s.addText(lims.map((t, i) => ({ text: t, options: { bullet: { code: "25CF" }, breakLine: i < lims.length - 1, color: ICE, fontSize: 14 } })), {
    x: 0.9, y: 1.5, w: 10.5, h: 2.6, fontFace: FONT_BODY, margin: 0, lineSpacingMultiple: 1.4,
  });

  s.addShape("line", { x: 0.9, y: 4.5, w: 8.5, h: 0, line: { color: NAVY_LIGHT, width: 1.5 } });

  s.addText("Full code, data, results, and this report:", {
    x: 0.9, y: 4.85, w: 8, h: 0.4, fontFace: FONT_BODY, fontSize: 14, color: "8A9BB0", margin: 0,
  });
  s.addText("github.com  \u2014  see README.md", {
    x: 0.9, y: 5.3, w: 8, h: 0.55, fontFace: FONT_HEAD, fontSize: 22, bold: true, color: AMBER, margin: 0,
  });
  s.addText("Thank you", {
    x: 0.9, y: 6.35, w: 8, h: 0.5, fontFace: FONT_HEAD, fontSize: 16, italic: true, color: WHITE, margin: 0,
  });
}

pres.writeFile({ fileName: path.join(__dirname, "vision_robustness_presentation.pptx") }).then(() => {
  console.log("Presentation written.");
});
