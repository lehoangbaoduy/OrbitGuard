/* OrbitGuard replay stage - label plates: merge nearby satellite labels into one plate (no stacking) and size them. */
(function (root) {
  "use strict";
  const OG = root.OG;
  const CHAR_WIDTH_PX = 7.8; // monospace 12 px
  const PLATE_PAD_PX = 10;
  const PLATE_HEIGHT_PX = 18;

  const dist = (a, b) => Math.hypot(a[0] - b[0], a[1] - b[1], a[2] - b[2]);

  /* items: {i, q:[x,y,z], st (severity), isSel, c, html}. Most severe first, then the selected one; an item joins the
     first plate whose lead is within `threshold` km and that still has room, else it opens a new plate. */
  function mergeLabels(items, threshold, maxPerPlate) {
    const ordered = items.map((item, n) => ({ item, n })).sort((a, b) => b.item.st - a.item.st || Number(b.item.isSel) - Number(a.item.isSel) || a.n - b.n).map((e) => e.item);
    const groups = [];
    for (const item of ordered) {
      const group = groups.find((g) => dist(g.q, item.q) < threshold && g.m.length < maxPerPlate);
      if (group) group.m.push(item);
      else groups.push({ q: item.q, m: [item] });
    }
    return groups;
  }

  const plateWidth = (html) => Math.ceil(html.replace(/<[^>]+>/g, "").length * CHAR_WIDTH_PX) + PLATE_PAD_PX;

  /* A Plotly scene annotation for one plate: real text on a dark plate with a coloured edge and a short leader. */
  function annotationFor(group, n, mono) {
    const lead = group.m[0], anySel = group.m.some((m) => m.isSel);
    const text = group.m.map((m) => m.html).join('<span style="color:#595d6c">  ·  </span>');
    return {
      x: group.q[0], y: group.q[1], z: group.q[2], text, width: plateWidth(text), height: PLATE_HEIGHT_PX, align: "center", valign: "middle",
      showarrow: true, arrowhead: 0, arrowwidth: 1, arrowcolor: lead.c, ax: (n % 2 ? 1 : -1) * 18, ay: -(26 + (n % 3) * 10),
      bgcolor: "rgba(14,16,25,0.9)", bordercolor: anySel ? OG.ICE : lead.c, borderwidth: anySel ? 1.5 : 1, borderpad: 4,
      font: { family: mono, size: 12, color: lead.c },
    };
  }

  Object.assign(OG, { mergeLabels, plateWidth, annotationFor });
})(globalThis);
