import { chromium } from 'playwright';
import fs from 'fs';
const files = process.argv.slice(2);
const b = await chromium.launch({ executablePath: '/opt/pw-browsers/chromium' });
for (const f of files) {
  const svg = fs.readFileSync(f, 'utf8');
  const m = svg.match(/width="(\d+)" height="(\d+)"/);
  const [w, h] = [parseInt(m[1]), parseInt(m[2])];
  const p = await b.newPage({ viewportSize:{width:Math.min(w,1600), height:600}, deviceScaleFactor:2 });
  await p.setContent(`<body style="margin:0;background:#fff">${svg}</body>`);
  await p.waitForTimeout(250);
  await p.locator('svg').screenshot({ path: f.replace('.svg','.png') });
  await p.close();
  console.log('rendered', f.replace('.svg','.png'), w+'x'+h, '-> 2x');
}
await b.close();
