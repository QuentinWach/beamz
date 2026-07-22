(() => {
  const typesetMath = () => {
    const content = document.querySelector(".md-content");
    if (!content || !window.MathJax?.typesetPromise) return;

    window.MathJax.typesetPromise([content]).catch(() => {});
  };

  if (typeof document$ !== "undefined") {
    document$.subscribe(typesetMath);
  } else {
    window.addEventListener("DOMContentLoaded", typesetMath, { once: true });
  }
})();
