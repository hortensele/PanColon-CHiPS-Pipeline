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
  let ATLAS = {};            // hpc id (int) -> {hpc,id,name,description,color}
  let currentHpcGrid = null; // {ncols,nrows,x0,y0,cells} for the slide on screen
  let currentAspect = 1;     // current slide's H/W, for grid<->viewport math
  let hoverCell = null;      // {col,row} last hovered tile, or null

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
    if (MANIFEST.hpc_atlas) loadAtlas(MANIFEST.hpc_atlas);
    if ((MANIFEST.slides || []).length) showSlide(MANIFEST.slides[0].slide_id);
  }

  // -- HPC atlas --------------------------------------------------------------
  async function loadAtlas(path) {
    let atlas;
    try {
      atlas = await (await fetch(path)).json();
    } catch (e) {
      return; // atlas is optional — viewer still works with bare HPC ids
    }
    ATLAS = {};
    atlas.forEach((a) => { ATLAS[a.hpc] = a; });
    renderAtlasPanel(atlas);
  }

  function renderAtlasPanel(atlas) {
    const el = document.getElementById("atlasList");
    if (!el) return;
    if (!atlas.length) { el.innerHTML = '<p class="empty">No HPC atlas provided.</p>'; return; }
    el.innerHTML = atlas.map((a) => `
      <div class="atlas-row">
        <span class="atlas-swatch" style="background:${a.color || "var(--muted)"}"></span>
        ${a.image
          ? `<img class="atlas-thumb" src="${a.image}" alt="HPC ${a.hpc} example tile">`
          : '<span class="atlas-thumb-ph"></span>'}
        <div class="atlas-body">
          <div class="atlas-name"><b>HPC ${a.hpc}</b> — ${escapeHtml(a.name || "")}${riskBadge(a.risk)}</div>
          <div class="atlas-desc">${escapeHtml(a.description || "")}</div>
        </div>
      </div>`).join("");
  }

  // Cohort-level finding (not recomputed per run): these HPCs correlate with
  // survival risk in the reference cohort's own analysis, and SurvCLAM gives
  // them elevated attention -- i.e. they're not just descriptive clusters,
  // they're what the model is actually keying on.
  function riskBadge(risk) {
    if (risk !== "high" && risk !== "low") return "";
    return ` <span class="pill ${risk}">${risk} risk</span> <span class="risk-note">elevated SurvCLAM attention</span>`;
  }

  function escapeHtml(s) {
    return String(s).replace(/[&<>"']/g, (c) => (
      { "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));
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
        const atlasView = document.getElementById("atlasView");
        if (atlasView) atlasView.style.display = which === "atlas" ? "" : "none";
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
      const tert = s.chips_tertile
        ? `<span class="pill ${s.chips_tertile}">${s.chips_tertile}</span>` : "";
      b.innerHTML =
        `<span>${s.slide_id}</span><span class="sc">${sc}${tert}</span>`;
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
    const tertCounts = { low: 0, intermediate: 0, high: 0 };
    slides.forEach((s) => { if (s.chips_tertile in tertCounts) tertCounts[s.chips_tertile]++; });
    document.getElementById("kpi").innerHTML = `
      <div class="c"><div class="n">${slides.length}</div><div class="l">Slides</div></div>
      <div class="c"><div class="n">${scores.length ? mean.toFixed(3) : "—"}</div><div class="l">Mean CHiPS</div></div>
      <div class="c"><div class="n">${tertCounts.low}</div><div class="l">Low</div></div>
      <div class="c"><div class="n">${tertCounts.intermediate}</div><div class="l">Intermediate</div></div>
      <div class="c"><div class="n">${tertCounts.high}</div><div class="l">High</div></div>`;
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
    const cols = ["slide_id", "chips_score", "chips_tertile"];
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
      <div class="c"><div class="n">${s.chips_score != null ? Number(s.chips_score).toFixed(3) : "—"}</div><div class="l">CHiPS score</div></div>`;

    currentHpcGrid = null;
    hoverCell = null;
    currentAspect = s.aspect || 1;
    if (s.hpc_grid) {
      fetch(`slides/${encodeURIComponent(sid)}/${s.hpc_grid}`)
        .then((r) => r.json()).then((g) => { currentHpcGrid = g; }).catch(() => {});
    }

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
    const hl = document.createElement("div");
    hl.className = "hpc-highlight";
    host.appendChild(hl);
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
    v._highlightEl = hl;
    wireHover(v, host);
    return v;
  }

  // Hovering ANY panel looks up the tile under the cursor via hpc_grid.json —
  // it carries HPC identity for every tile regardless of which layer you're
  // looking at — and (a) shows "HPC N — <name>" in a floating tooltip, and
  // (b) draws a matching highlight box on all three panels at that tile's
  // location, so the H&E/attention views make it obvious where on the slide
  // the hovered HPC tile actually is. All three layers share the same
  // normalized [0,1]x[0,aspect] extent (see file header), so pointFromPixel /
  // pixelFromPoint translate between a panel's screen and grid coordinates
  // without needing to know any layer's raw pixel dimensions.
  function wireHover(v, host) {
    host.addEventListener("mousemove", (e) => {
      if (!currentHpcGrid) { hideHover(); return; }
      const rect = host.getBoundingClientRect();
      const px = new OpenSeadragon.Point(e.clientX - rect.left, e.clientY - rect.top);
      const vp = v.viewport.pointFromPixel(px);
      const u = vp.x, w = vp.y / currentAspect;
      if (u < 0 || u > 1 || w < 0 || w > 1) { hideHover(); return; }
      const col = Math.floor(u * currentHpcGrid.ncols);
      const row = Math.floor(w * currentHpcGrid.nrows);
      const hpc = currentHpcGrid.cells[`${col},${row}`];
      if (hpc == null) { hideHover(); return; }
      hoverCell = { col, row };
      showTooltip(e.clientX, e.clientY, hpc);
      paintHighlights();
    });
    host.addEventListener("mouseleave", () => { hoverCell = null; hideHover(); });
  }

  function showTooltip(x, y, hpc) {
    const tip = document.getElementById("hpcTooltip");
    if (!tip) return;
    const a = ATLAS[hpc];
    tip.innerHTML = a && a.name
      ? `<b>HPC ${hpc}</b> — ${escapeHtml(a.name)}`
      : `<b>HPC ${hpc}</b>`;
    tip.style.left = (x + 14) + "px";
    tip.style.top = (y + 14) + "px";
    tip.style.display = "block";
  }

  function hideHover() {
    const tip = document.getElementById("hpcTooltip");
    if (tip) tip.style.display = "none";
    viewers.forEach((v) => { if (v._highlightEl) v._highlightEl.style.display = "none"; });
  }

  // Position the highlight box on every panel over the current hoverCell's
  // grid square, in that panel's own current pan/zoom (pixelFromPoint), so it
  // stays put under the cursor's panel and lands at the *same slide location*
  // in the other two even though each is a separate OSD instance.
  function paintHighlights() {
    if (!hoverCell || !currentHpcGrid) return;
    const g = currentHpcGrid;
    const u0 = hoverCell.col / g.ncols, u1 = (hoverCell.col + 1) / g.ncols;
    const w0 = (hoverCell.row / g.nrows) * currentAspect;
    const w1 = ((hoverCell.row + 1) / g.nrows) * currentAspect;
    viewers.forEach((v) => {
      const el = v._highlightEl;
      if (!el) return;
      const p0 = v.viewport.pixelFromPoint(new OpenSeadragon.Point(u0, w0), true);
      const p1 = v.viewport.pixelFromPoint(new OpenSeadragon.Point(u1, w1), true);
      el.style.left = Math.min(p0.x, p1.x) + "px";
      el.style.top = Math.min(p0.y, p1.y) + "px";
      el.style.width = Math.max(1, Math.abs(p1.x - p0.x)) + "px";
      el.style.height = Math.max(1, Math.abs(p1.y - p0.y)) + "px";
      el.style.display = "block";
    });
  }

  // Propagate viewport center+zoom from whichever viewer the user drives to the
  // others. All layers share the same normalized extent, so this keeps the three
  // panels aligned. A reentrancy guard stops the echo.
  //
  // OSD's "zoom"/"pan" viewer events fire exactly ONCE per zoomTo/panTo call —
  // synchronously, at the moment the gesture starts — not once per animation
  // frame as the value springs toward its target. So the *current* (mid-
  // animation) zoom/center via getZoom(true)/getCenter(true) is still the OLD
  // value at that instant; reading it (as this used to) propagates a no-op
  // while the driven viewer keeps animating on its own with no further events
  // to catch it. Reading the animation TARGET (no "current" flag) instead
  // gets where the gesture is headed, and animating the other two viewers to
  // that same target (not snapping instantly) keeps all three moving in
  // visual lockstep.
  function syncViewers() {
    viewers.forEach((v) => {
      const push = () => {
        if (syncing) return;
        syncing = true;
        const c = v.viewport.getCenter();
        const z = v.viewport.getZoom();
        viewers.forEach((o) => {
          if (o === v) return;
          o.viewport.zoomTo(z);
          o.viewport.panTo(c);
        });
        syncing = false;
        paintHighlights();
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

  // HPC composition -> labelled horizontal bars, colours matching the map overlay.
  function renderHpcChart(container, composition, topN = 12) {
    container.innerHTML = "";
    if (!composition || !composition.length) {
      container.innerHTML = '<p class="empty">No HPC assignment for this slide.</p>';
      return;
    }
    const sorted = composition.slice().sort((a, b) => b.n_tiles - a.n_tiles);

    // Risk-associated HPCs are clinically the point of this list -- always
    // surface them even if a small presence would otherwise fold into
    // "other" below the topN-by-tile-count cutoff.
    const isRisky = (r) => {
      const a = ATLAS[r.hpc];
      return !!(a && (a.risk === "high" || a.risk === "low"));
    };
    const naturalTop = sorted.slice(0, topN);
    const forcedExtra = sorted.filter((r) => isRisky(r) && !naturalTop.includes(r));
    const shown = naturalTop.concat(forcedExtra).sort((a, b) => b.n_tiles - a.n_tiles);
    const shownSet = new Set(shown);
    const rest = sorted.filter((r) => !shownSet.has(r));
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
      const a = ATLAS[r.hpc];
      const name = a && a.name ? ` — ${a.name}` : "";
      row.title = `HPC ${r.hpc}${name}: ${r.n_tiles.toLocaleString()} tiles (${(r.frac * 100).toFixed(1)}%)`;
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
      if (a && (a.risk === "high" || a.risk === "low")) {
        const badge = document.createElement("div");
        badge.className = "hpc-risk-line";
        badge.innerHTML = riskBadge(a.risk).trim();
        container.appendChild(badge);
      }
      if (a && a.description) {
        const desc = document.createElement("div");
        desc.className = "hpc-desc";
        desc.textContent = truncate(a.description, 130);
        container.appendChild(desc);
      }
    });
  }

  function truncate(s, n) {
    return s.length > n ? s.slice(0, n).replace(/\s+\S*$/, "") + "…" : s;
  }
})();
