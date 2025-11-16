// save_png.js
const puppeteer = require('puppeteer');

(async () => {
  const browser = await puppeteer.launch({ headless: 'new' });
  const page = await browser.newPage();

  // Wider viewport + high device scale for print-quality text
  await page.setViewport({ width: 1400, height: 900, deviceScaleFactor: 3 });

  await page.goto('file:///home/chengine/Research/shadow_splat/results/tables/all_scenes_grouped_from_csv.html', { waitUntil: 'networkidle0' });
  await page.screenshot({ path: 'out.png', fullPage: true }); // SINGLE long image
  await browser.close();
})();
