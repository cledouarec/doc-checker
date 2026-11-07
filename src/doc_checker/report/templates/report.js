"use strict";
(() => {
  const form = document.getElementById("filters");
  const findings = Array.from(document.querySelectorAll("article.finding"));
  const texts = findings.map((finding) => finding.textContent.toLowerCase());
  const sections = Array.from(document.querySelectorAll("section.severity"));
  const shown = document.getElementById("shown");
  function apply() {
    const severities = new Set(
      Array.from(
        form.querySelectorAll("input[name=severity]:checked"),
        (box) => box.value,
      ),
    );
    const category = form.elements.category.value;
    const rule = form.elements.rule.value;
    const query = form.elements.q.value.trim().toLowerCase();
    let count = 0;
    findings.forEach((finding, index) => {
      const visible =
        severities.has(finding.dataset.severity) &&
        (!category || finding.dataset.category === category) &&
        (!rule || finding.dataset.rule === rule) &&
        (!query || texts[index].includes(query));
      finding.hidden = !visible;
      count += visible ? 1 : 0;
    });
    sections.forEach((section) => {
      section.hidden = !section.querySelector("article.finding:not([hidden])");
    });
    shown.value = String(count);
  }
  form.addEventListener("input", apply);
  form.addEventListener("submit", (event) => event.preventDefault());
  apply();
})();
