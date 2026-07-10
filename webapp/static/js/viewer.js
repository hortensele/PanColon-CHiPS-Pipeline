/* Interactive per-slide viewer + charts for the PanColon-CHiPS web app.
 *
 * PanColonViewer wraps OpenSeadragon: it opens a slide's DeepZoom source and
 * places a single heatmap overlay (attention or HPC) over the full slide as an
 * <img>, with an opacity control. renderHpcChart draws the HPC composition as a
 * labelled horizontal bar chart whose colours match the map overlay. */
const PanColonViewer = (function () {
  let viewer = null;
  let slide = null;
  let overlayEl = null;
  let kind = "attention";
  let opacity = 0.6;

  function open(slideId, hasWsi) {
    slide = slideId;
    const host = document.getElementById("osd");
    host.innerHTML = "";
    overlayEl = null;
    if (viewer) { viewer.destroy(); viewer = null; }
    if (!hasWsi) {
      host.innerHTML = '<div class="osd-empty">Source WSI not found on the ' +
        'server, so the zoomable slide can’t be shown. The heatmap grids ' +
        'are still available below.</div>';
      return;
    }
    viewer = OpenSeadragon({
      element: host,
      prefixUrl: "/static/vendor/openseadragon/images/",
      tileSources: "/dzi/" + encodeURIComponent(slideId) + ".dzi",
      showNavigator: true,
      navigatorPosition: "TOP_RIGHT",
      crossOriginPolicy: false,
      maxZoomPixelRatio: 2,
    });
    viewer.addHandler("open", () => setOverlay(kind));
  }

  function setOverlay(which) {
    kind = which;
    if (!viewer) return;
    fetch(`/overlay/${encodeURIComponent(slide)}/${kind}.json`)
      .then(r => r.json())
      .then(meta => {
        if (overlayEl) { viewer.removeOverlay(overlayEl); overlayEl = null; }
        if (!meta.available || !meta.placement) return;
        const p = meta.placement;
        const img = document.createElement("img");
        img.src = `/overlay/${encodeURIComponent(slide)}/${kind}.png?` + Date.now();
        img.className = "osd-overlay-img";
        img.style.opacity = opacity;
        overlayEl = img;
        viewer.addOverlay({
          element: img,
          location: new OpenSeadragon.Rect(p.x, p.y, p.width, p.height),
        });
      })
      .catch(() => {});
  }

  function setOpacity(v) {
    opacity = v;
    if (overlayEl) overlayEl.style.opacity = v;
  }

  return { open, setOverlay, setOpacity, currentKind: () => kind };
})();


/* HPC composition -> horizontal bar chart. Each bar is one phenotype cluster,
 * sorted by prevalence, coloured to match the slide overlay, and directly
 * labelled with its HPC id and share so identity never rests on colour alone.
 * The long tail folds into a single "other" bar. */
function renderHpcChart(container, composition, topN = 12) {
  container.innerHTML = "";
  if (!composition || !composition.length) {
    container.innerHTML = '<p class="empty">No HPC assignment for this slide.</p>';
    return;
  }
  const sorted = composition.slice().sort((a, b) => b.n_tiles - a.n_tiles);
  const shown = sorted.slice(0, topN);
  const rest = sorted.slice(topN);
  if (rest.length) {
    shown.push({
      hpc: "other", color: "var(--muted)",
      n_tiles: rest.reduce((s, r) => s + r.n_tiles, 0),
      frac: rest.reduce((s, r) => s + r.frac, 0),
    });
  }
  const max = Math.max(...shown.map(r => r.frac));
  const total = composition.reduce((s, r) => s + r.n_tiles, 0);

  const head = document.createElement("div");
  head.className = "hpc-total";
  head.textContent = `${composition.length} phenotype clusters · ${total.toLocaleString()} tiles`;
  container.appendChild(head);

  shown.forEach(r => {
    const row = document.createElement("div");
    row.className = "hpc-row";
    row.title = `HPC ${r.hpc}: ${r.n_tiles.toLocaleString()} tiles (${(r.frac * 100).toFixed(1)}%)`;
    const label = document.createElement("span");
    label.className = "hpc-label";
    label.textContent = r.hpc === "other" ? "other" : `HPC ${r.hpc}`;
    const track = document.createElement("div");
    track.className = "hpc-track";
    const fill = document.createElement("div");
    fill.className = "hpc-fill";
    fill.style.width = (max ? (r.frac / max) * 100 : 0) + "%";
    fill.style.background = r.color;
    track.appendChild(fill);
    const val = document.createElement("span");
    val.className = "hpc-val";
    val.textContent = (r.frac * 100).toFixed(1) + "%";
    row.append(label, track, val);
    container.appendChild(row);
  });
}
