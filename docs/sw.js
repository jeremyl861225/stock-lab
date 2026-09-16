/* 股市預測 · Service Worker
 *
 * 這個 origin（jeremyl861225.github.io）底下有多個 PWA。
 * caches.keys() 會列出「整個 origin」所有 app 的快取，
 * 所以清理時只能刪掉前綴符合自己的那些 —— 否則會清掉 todo-app 等其他 app。
 */
const CACHE = 'stocklab-v1';
const ASSETS = ['./', './index.html', './manifest.json',
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

// stale-while-revalidate：先給快取讓開盤前秒開，背景抓新版供下次使用
self.addEventListener('fetch', e => {
  const u = new URL(e.request.url);
  if (e.request.method !== 'GET' || u.origin !== location.origin) return;
  if (!u.pathname.startsWith('/stock-lab/')) return;               // 不碰其他 app 的路徑
  e.respondWith(
    caches.open(CACHE).then(c =>
      c.match(e.request).then(hit => {
        const net = fetch(e.request).then(r => {
          if (r && r.status === 200) c.put(e.request, r.clone());
          return r;
        }).catch(() => hit);
        return hit || net;
      })));
});
