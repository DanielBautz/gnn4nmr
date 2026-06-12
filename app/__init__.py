"""Flask app factory for the GNN4NMR web app."""
import os

from .services import pathsetup  # noqa: F401  (must run before scripts imports)

from flask import Flask, render_template


def create_app():
    import torch

    # Leave CPU headroom for request threads while training runs.
    torch.set_num_threads(max(1, (os.cpu_count() or 4) - 2))

    app = Flask(__name__)

    from . import config
    from .api import api
    from .state import STATE
    from .services import dataset_service, features

    app.register_blueprint(api)

    @app.get("/")
    def index():
        return render_template("index.html")

    # Load the default dataset eagerly so the first page load is instant.
    default_path = os.path.join(config.DATA_DIR, config.DEFAULT_DATASET_FILE)
    if os.path.exists(default_path):
        entry = dataset_service.load_dataset(STATE, config.DEFAULT_DATASET_FILE)
        if entry["nx_graphs"]:
            sample = entry["dataset"][0]
            features.verify_against_sample(sample)
            node_map = dataset_service.get_node_map(entry, 0)
            for ntype in sample.node_types:
                mapped = len(node_map["by_type"][ntype])
                width = int(sample[ntype].x.shape[0])
                if mapped != width:
                    raise RuntimeError(
                        f"Node map mismatch for '{ntype}': {mapped} mapped vs "
                        f"{width} tensor rows. Index translation would be wrong."
                    )
        app.logger.info(
            "Loaded dataset %s (%d graphs, device=%s)",
            config.DEFAULT_DATASET_FILE,
            len(entry["nx_graphs"]),
            STATE.device,
        )
    else:
        app.logger.warning("Default dataset not found at %s", default_path)

    return app
