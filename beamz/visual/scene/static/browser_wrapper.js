const sceneSpec = window.__ZVIEW_SCENE__;
const titleNode = document.getElementById("zview-title");
if (titleNode) {
  titleNode.textContent = sceneSpec.title || "ZView";
}
document.title = sceneSpec.title || "ZView";

mountZView({
  el: document.getElementById("zview-root"),
  sceneSpec,
});
