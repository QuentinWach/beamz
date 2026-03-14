function syncTrait(model, name, value) {
  if (model.get(name) === value) {
    return;
  }
  model.set(name, value);
  model.save_changes();
}

export default {
  render({ model, el }) {
    const viewer = mountZView({
      el,
      sceneSpec: model.get("scene_json") || {},
      onHover: (value) => syncTrait(model, "hovered_object_id", value),
      onSelect: (value) => syncTrait(model, "selected_object_id", value),
    });

    const onChange = () => viewer.setScene(model.get("scene_json") || {});
    model.on("change:scene_json", onChange);

    return () => {
      if (typeof model.off === "function") {
        model.off("change:scene_json", onChange);
      }
      viewer.destroy();
    };
  },
};
