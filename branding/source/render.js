// Render an HTML file to PNG with headless Chromium (needs `npm i playwright`).
// usage: node render.js <in.html> <out.png> <width> <height> [scale=2]
const { chromium } = require('playwright');
const path = require('path');

(async () => {
  const [inFile, outFile, w, h, scale = '2'] = process.argv.slice(2);
  const browser = await chromium.launch();
  const page = await browser.newPage({ viewport: { width: +w, height: +h }, deviceScaleFactor: +scale });
  await page.goto('file://' + path.resolve(inFile));
  await page.evaluate(() => document.fonts.ready);
  await page.screenshot({ path: outFile });
  await browser.close();
  console.log('rendered', outFile);
})();
