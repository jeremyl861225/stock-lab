/* 股市預測 · Service Worker
 *
 * 這個 origin（jeremyl861225.github.io）底下有多個 PWA。
 * caches.keys() 會列出「整個 origin」所有 app 的快取，
 * 所以清理時只能刪掉前綴符合自己的那些 —— 否則會清掉 todo-app 等其他 app。
 */
const CACHE = 'stocklab-v3';
const ASSETS = ['./', './index.html', './manifest.json', './charts.json',
                './icons/icon-192.png', './icons/icon-512.png'];

self.addEventListener('install', e => {
  e.waitUntil(caches.open(CACHE).then(c => c.addAll(ASSETS)).then(() => self.skipWaiting()));
});

self.addEventListener('activate', e => {
  e.waitUntil(
    caches.keys()
      .then(ks => Promise.all(
        ks.filter(k => k.startsWith('stocklab-') && k !== CACHE)   // ← 只刪自己的
          .map(k => caches.delete(k))))
      .then(() => self.clients.claim()));
});

/* 兩種策略，依這個檔案「過期的代價」決定：
 *
 * index.html 走 network-first（3 秒逾時後退回快取）。
 * 它裝的是當天的預測 —— 用 stale-while-revalidate 的話，每天早上第一次
 * 打開看到的是昨天的預測，要重整第二次才是今天的。對一個「開盤前看結果」
 * 的 app，這等於每天都給錯答案一次。（也因此我推壞版本時，
 * 修好之後使用者仍看到壞的。）
 * 逾時後仍退回快取，所以離線或網路慢時照樣秒開。
 *
 * 其餘（icons／manifest）走 stale-while-revalidate：它們幾乎不變，
 * 過期沒有代價，快取優先才能秒開。
 */
// 判準是「這個檔案裝不裝當天的資料」：HTML 都裝（index 是預測、
// detail 是驗證明細，兩者都帶當日日期），icons／manifest 都不裝。
const FRESH = /\/stock-lab\/([^/]*\.html)?$/;

function netFirst(req, c) {
  return new Promise(resolve => {
    let settled = false;
    const done = r => { if (!settled) { settled = true; resolve(r); } };
    const timer = setTimeout(() => c.match(req).then(hit => hit && done(hit)), 3000);
    fetch(req).then(r => {
      clearTimeout(timer);
      if (r && r.status === 200) c.put(req, r.clone());
      done(r);
    }).catch(() => {
      clearTimeout(timer);
      c.match(req).then(hit => done(hit || Response.error()));
    });
  });
}

self.addEventListener('fetch', e => {
  const u = new URL(e.request.url);
  if (e.request.method !== 'GET' || u.origin !== location.origin) return;
  if (!u.pathname.startsWith('/stock-lab/')) return;               // 不碰其他 app 的路徑
  e.respondWith(caches.open(CACHE).then(c =>
    FRESH.test(u.pathname)
      ? netFirst(e.request, c)
      : c.match(e.request).then(hit => {
          const net = fetch(e.request).then(r => {
            if (r && r.status === 200) c.put(e.request, r.clone());
            return r;
          }).catch(() => hit);
          return hit || net;
        })));
});
