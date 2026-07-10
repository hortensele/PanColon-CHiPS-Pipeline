/* Static results viewer for a PanColon-CHiPS bundle.
 *
 * Loads manifest.json (produced by `pancolon_pipeline.py export`) and renders:
 *  - a cohort view: CHiPS table + ranked distribution;
 *  - a per-slide view: three OpenSeadragon panels (H&E, HPC, attention) whose
 *    pan/zoom are synchronized, plus the slide's CHiPS readout and HPC bars.
 *
 * Pure static: no server logic beyond serving files (python -m http.server).
 * All three layers are exported to the same normalized [0,1]x[0,aspect] extent,
 * so synchronizing viewport center+zoom keeps them pixel-aligned. */
(function () {
  const LAYERS = [
    { key: "he", label: "H&E", type: "dzi" },
    { key: "hpc", label: "HPC", type: "image" },
    { key: "attention", label: "Attention", type: "image" },
  ];
  let MANIFEST = null;
  let viewers = [];
  let syncing = false;

  document.addEventListener("DOMContentLoaded", init);

  async function init() {
    try {
      MANIFEST = await (await fetch("manifest.json")).json();
    } catch (e) {
      document.getElementById("app").innerHTML =
        '<p class="empty">Could not load manifest.json. Serve this folder with ' +
        '<code>python -m http.server</code> and open the printed URL.</p>';
      return;
    }
    document.getElementById("dataset").textContent = MANIFEST.dataset || "cohort";
    document.getElementById("nslides").textContent =
      (MANIFEST.slides || []).length + " slides";
    buildSlideList();
    renderCohort();
    wireTabs();
    if ((MANIFEST.slides || []).length) showSlide(MANIFEST.slides[0].slide_id);
  }

  // -- tabs -----------------------------------------------------------------
  function wireTabs() {
    document.querySelectorAll(".tab").forEach((t) => {
      t.onclick = () => {
        document.querySelectorAll(".tab").forEach((x) => x.classList.remove("on"));
        t.classList.add("on");
        const which = t.dataset.tab;
        document.getElementById("cohortView").style.display =
          which === "cohort" ? "" : "none";
        document.getElementById("slideView").style.display =
          which === "slide" ? "" : "none";
      };
    });
  }
  function goTab(which) {
    const t = document.querySelector(`.tab[data-tab="${which}"]`);
    if (t) t.click();
  }

  // -- slide list -----------------------------------------------------------
  function buildSlideList() {
    const list = document.getElementById("slideList");
    list.innerHTML = "";
    (MANIFEST.slides || []).forEach((s) => {
      const b = document.createElement("button");
      const sc = s.chips_score != null ? Number(s.chips_score).toFixed(3) : "—";
      b.innerHTML = `<span>${s.slide_id}</span><span class="sc">${sc}</span>`;
      b.dataset.slide = s.slide_id;
      b.onclick = () => { showSlide(s.slide_id); goTab("slide"); };
      list.appendChild(b);
    });
  }

  // -- cohort view ----------------------------------------------------------
  function renderCohort() {
    const slides = MANIFEST.slides || [];
    const scores = slides.map((s) => Number(s.chips_score)).filter((x) => !isNaN(x));
    const mean = scores.length ? scores.reduce((a, b) => a + b, 0) / scores.length : 0;
    const med = median(scores);
    document.getElementById("kpi").innerHTML = `
      <div class="c"><div class="n">${slides.length}</div><div class="l">Slides</div></div>
      <div class="c"><div class="n">${scores.length ? mean.toFixed(3) : "—"}</div><div class="l">Mean CHiPS</div></div>
      <div class="c"><div class="n">${scores.length ? med.toFixed(3) : "—"}</div><div class="l">Median CHiPS</div></div>`;
    renderDistribution(slides);
    renderTable(slides);
  }

  // Ranked horizontal bars: single measure (CHiPS) by identity (slide). Recessive
  // axis, one hue, selected slide highlighted; direct value labels, no legend.
  function renderDistribution(slides) {
    const el = document.getElementById("dist");
    const ranked = slides
      .filter((s) => s.chips_score != null && !isNaN(Number(s.chips_score)))
      .sort((a, b) => Number(b.chips_score) - Number(a.chips_score));
    if (!ranked.length) { el.innerHTML = '<p class="empty">No CHiPS scores.</p>'; return; }
    const max = Number(ranked[0].chips_score);
    const min = Number(ranked[ranked.length - 1].chips_score);
    const span = max - min || 1;
    el.innerHTML = "";
    ranked.forEach((s) => {
      const v = Number(s.chips_score);
      const row = document.createElement("div");
      row.className = "dist-row";
      row.title = `${s.slide_id}: CHiPS ${v.toFixed(3)}`;
      row.onclick = () => { showSlide(s.slide_id); goTab("slide"); };
      const label = document.createElement("span");
      label.className = "dist-label"; label.textContent = s.slide_id;
      const track = document.createElement("div"); track.className = "dist-track";
      const fill = document.createElement("div"); fill.className = "dist-fill";
      fill.style.width = (5 + 95 * (v - min) / span) + "%";
      track.appendChild(fill);
      const val = document.createElement("span");
      val.className = "dist-val"; val.textContent = v.toFixed(3);
      row.append(label, track, val);
      el.appendChild(row);
    });
  }

  function renderTable(slides) {
    const cols = ["slide_id", "chips_score", "chips_percentile", "chips_tertile"];
    const thead = document.querySelector("#chipsTable thead");
    const tbody = document.querySelector("#chipsTable tbody");
    thead.innerHTML = "<tr>" + cols.map((c) => `<th data-c="${c}">${c}</th>`).join("") + "</tr>";
    let sortCol = "chips_score", sortDir = -1;
    const draw = () => {
      const rows = slides.slice().sort((a, b) => {
        let x = a[sortCol], y = b[sortCol];
        const nx = Number(x), ny = Number(y);
        if (!isNaN(nx) && !isNaN(ny)) { x = nx; y = ny; }
        return x < y ? -sortDir : x > y ? sortDir : 0;
      });
      tbody.innerHTML = rows.map((r) => "<tr data-slide='" + r.slide_id + "'>" + cols.map((c) => {
        let v = r[c]; if (v == null) v = "";
        if (c === "chips_score") v = v === "" ? "" : Number(v).toFixed(3);
        if (c === "chips_percentile") v = v === "" ? "" : Number(v).toFixed(0);
        if (c === "chips_tertile" && v) v = `<span class="pill ${v}">${v}</span>`;
        return `<td>${v}</td>`;
      }).join("") + "</tr>").join("");
      tbody.querySelectorAll("tr").forEach((tr) => tr.onclick = () => {
        showSlide(tr.dataset.slide); goTab("slide");
      });
    };
    thead.querySelectorAll("th").forEach((th) => th.onclick = () => {
      const c = th.dataset.c;
      if (c === sortCol) sortDir *= -1; else { sortCol = c; sortDir = -1; }
      draw();
    });
    draw();
  }

  // -- slide view: three synchronized viewers -------------------------------
  function showSlide(sid) {
    const s = (MANIFEST.slides || []).find((x) => x.slide_id === sid);
    if (!s) return;
    document.querySelectorAll("#slideList button").forEach((b) =>
      b.classList.toggle("on", b.dataset.slide === sid));

    const tert = s.chips_tertile ? `<span class="pill ${s.chips_tertile}">${s.chips_tertile}</span>` : "";
    document.getElementById("slideHead").innerHTML =
      `<span class="sid">${sid}</span>${tert}`;
    document.getElementById("chipReadout").innerHTML = `
      <div class="c"><div class="n">${s.chips_score != null ? Number(s.chips_score).toFixed(3) : "—"}</div><div class="l">CHiPS score</div></div>
      <div class="c"><div class="n">${s.chips_percentile != null ? Number(s.chips_percentile).toFixed(0) : "—"}</div><div class="l">Cohort percentile</div></div>`;

    destroyViewers();
    const grid = document.getElementById("panels");
    grid.innerHTML = "";
    viewers = LAYERS.map((layer) => makePanel(grid, s, layer)).filter(Boolean);
    setTimeout(syncViewers, 0);

    renderHpcChart(document.getElementById("hpcChart"), s.composition || []);
  }

  function makePanel(grid, s, layer) {
    const wrap = document.createElement("div");
    wrap.className = "panel-cell";
    const title = document.createElement("div");
    title.className = "panel-title"; title.textContent = layer.label;
    const host = document.createElement("div");
    host.className = "osd"; host.id = "osd_" + layer.key;
    wrap.append(title, host);
    grid.appendChild(wrap);

    const val = s[layer.key];
    if (!val) {
      host.innerHTML = '<div class="osd-empty">not available</div>';
      return null;
    }
    const base = `slides/${encodeURIComponent(s.slide_id)}/`;
    const tileSources = layer.type === "dzi"
      ? base + val
      : { type: "image", url: base + val };
    const v = OpenSeadragon({
      element: host,
      prefixUrl: "openseadragon/images/",
      tileSources,
      showNavigator: false,
      crossOriginPolicy: false,
      minZoomImageRatio: 0.5,
      maxZoomPixelRatio: 4,
      gestureSettingsMouse: { clickToZoom: false },
    });
    v._layerKey = layer.key;
    return v;
  }

  // Propagate viewport center+zoom from whichever viewer the user drives to the
  // others. All layers share the same normalized extent, so this keeps the three
  // panels aligned. A reentrancy guard stops the echo.
  function syncViewers() {
    viewers.forEach((v) => {
      const push = () => {
        if (syncing) return;
        syncing = true;
        const c = v.viewport.getCenter(true);
        const z = v.viewport.getZoom(true);
        viewers.forEach((o) => {
          if (o === v) return;
          o.viewport.zoomTo(z, null, true);
          o.viewport.panTo(c, true);
        });
        syncing = false;
      };
      v.addHandler("zoom", push);
      v.addHandler("pan", push);
    });
  }

  function destroyViewers() {
    viewers.forEach((v) => { try { v.destroy(); } catch (e) {} });
    viewers = [];
  }

  // -- helpers --------------------------------------------------------------
  function median(a) {
    if (!a.length) return 0;
    const s = a.slice().sort((x, y) => x - y);
    const m = Math.floor(s.length / 2);
    return s.length % 2 ? s[m] : (s[m - 1] + s[m]) / 2;
  }

  // HPC composition -> labelled horizontal bars, colours matching the map overlay.
  function renderHpcChart(container, composition, topN = 12) {
    container.innerHTML = "";
    if (!composition || !composition.length) {
      container.innerHTML = '<p class="empty">No HPC assignment for this slide.</p>';
      return;
    }
    const sorted = composition.slice().sort((a, b) => b.n_tiles - a.n_tiles);
    const shown = sorted.slice(0, topN);
    const rest = sorted.slice(topN);
    if (rest.length) shown.push({
      hpc: "other", color: "var(--muted)",
      n_tiles: rest.reduce((s, r) => s + r.n_tiles, 0),
      frac: rest.reduce((s, r) => s + r.frac, 0),
    });
    const max = Math.max.apply(null, shown.map((r) => r.frac));
    const total = composition.reduce((s, r) => s + r.n_tiles, 0);
    const head = document.createElement("div");
    head.className = "hpc-total";
    head.textContent = `${composition.length} phenotype clusters · ${total.toLocaleString()} tiles`;
    container.appendChild(head);
    shown.forEach((r) => {
      const row = document.createElement("div");
      row.className = "hpc-row";
      row.title = `HPC ${r.hpc}: ${r.n_tiles.toLocaleString()} tiles (${(r.frac * 100).toFixed(1)}%)`;
      const label = document.createElement("span");
      label.className = "hpc-label";
      label.textContent = r.hpc === "other" ? "other" : `HPC ${r.hpc}`;
      const track = document.createElement("div"); track.className = "hpc-track";
      const fill = document.createElement("div"); fill.className = "hpc-fill";
      fill.style.width = (max ? (r.frac / max) * 100 : 0) + "%";
      fill.style.background = r.color || "var(--hema)";
      track.appendChild(fill);
      const val = document.createElement("span");
      val.className = "hpc-val"; val.textContent = (r.frac * 100).toFixed(1) + "%";
      row.append(label, track, val);
      container.appendChild(row);
    });
  }
})();
