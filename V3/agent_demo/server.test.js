"use strict";

const test = require("node:test");
const assert = require("node:assert/strict");
const { parseDataUrl, validateSmilesLight, normalizeSettings, demoResult } = require("./server");

const onePixelPng = "data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mNk+A8AAQUBAScY42YAAAAASUVORK5CYII=";

test("accepts supported image data URLs", () => {
  const parsed = parseDataUrl(onePixelPng);
  assert.equal(parsed.mime, "image/png");
  assert.ok(parsed.buffer.length > 20);
});

test("rejects unsupported data", () => {
  assert.throws(() => parseDataUrl("data:text/plain;base64,SGVsbG8="));
});

test("light validator catches malformed or multi-fragment strings", () => {
  assert.equal(validateSmilesLight("CC(=O)O").valid, true);
  assert.equal(validateSmilesLight("CC.O").valid, false);
  assert.equal(validateSmilesLight("C(C").valid, false);
});

test("settings are constrained to released decoder bounds", () => {
  assert.deepEqual(normalizeSettings({ beams: 9, returns: 0, max_tokens: 999, tta: true }), { beams: 4, returns: 1, max_tokens: 512, tta: true });
});

test("guided demo is explicit and returns auditable trace", () => {
  const parsed = parseDataUrl(onePixelPng);
  const result = demoResult({ sample_id: "caffeine-v1", client_diagnostics: { width: 800, height: 600 }, settings: {} }, parsed);
  assert.equal(result.mode, "guided_demo");
  assert.equal(result.status, "completed");
  assert.equal(result.trace.length, 6);
  assert.equal(result.candidates[0].valid, true);
});
