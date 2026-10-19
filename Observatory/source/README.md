# Viewer source

Run `python Observatory/source/build.py` to rebuild `web/app.js`, the inline template and the standalone frozen replay. No packages or build service are required. The three JavaScript source files are combined in order; interface CSS and panel behaviour live in `web/interface.css` and `web/interface.js`.

Serve `Observatory/web` with a local HTTP server. `trace.json` is the unchanged, independently audited public capture. `verification/validate-graph.cjs` validates the actual published graph; `validate-interactions.cjs` runs the application handlers with a mock DOM/WebGL (not a GPU test).
