// Mermaid 渲染初始化：配合 mkdocs-material 的 document$ 事件
(() => {
  if (typeof mermaid === "undefined") return;
  mermaid.initialize({
    startOnLoad: false,
    theme: "base",
    securityLevel: "strict",
    flowchart: { curve: "basis", htmlLabels: true },
    themeVariables: {
      primaryColor: "#e8eaf6",
      primaryTextColor: "#1a237e",
      primaryBorderColor: "#5c6bc0",
      lineColor: "#5c6bc0"
    }
  });
  document$.subscribe(() => {
    mermaid.run({ querySelector: ".mermaid" });
  });
})();
