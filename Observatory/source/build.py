"""Build a standalone frozen replay from this fork's renderer sources."""
from pathlib import Path
import re
root=Path(__file__).resolve().parent
dist=root/'dist' if (root/'dist').exists() else root.parent/'web'
app=(root/'app-source.js').read_text(encoding='utf-8').replace('/* @renderer */',(root/'renderer-functions.js').read_text(encoding='utf-8')).replace('/* @interaction */',(root/'interaction-functions.js').read_text(encoding='utf-8'))
(dist/'app.js').write_text(app,encoding='utf-8')
html=re.sub(r'\?v=[a-zA-Z0-9-]+','',(dist/'index.html').read_text(encoding='utf-8'))
html=html.replace('<link rel="stylesheet" href="style.css">','<style>'+(dist/'style.css').read_text(encoding='utf-8')+'</style>')
html=html.replace('<link rel="stylesheet" href="interface.css">','<style>'+(dist/'interface.css').read_text(encoding='utf-8')+'</style>')
for name in ['production','topology','filaments','app','interface']:
    script=(dist/(name+'.js')).read_text(encoding='utf-8')
    html=html.replace('<script src="'+name+'.js"></script>','<script>'+script+'</script>')
template=html.replace('<script>'+ (dist/'production.js').read_text(encoding='utf-8'),'<script>window.ATOMIC_TRACE=__ATOMIC_TRACE__;</script><script>'+(dist/'production.js').read_text(encoding='utf-8'),1)
(root/'template.html').write_text(template,encoding='utf-8')
trace=(dist/'trace.json').read_text(encoding='utf-8').replace('<','\\u003c')
html=html.replace('<script>'+ (dist/'production.js').read_text(encoding='utf-8'),'<script>window.ATOMIC_TRACE='+trace+';</script><script>'+(dist/'production.js').read_text(encoding='utf-8'),1)
(dist/'Atomic-Observatory-offline.html').write_text(html,encoding='utf-8')
print('Built dist/Atomic-Observatory-offline.html')
