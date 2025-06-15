import os
import re

# Define the code snippets to be inserted
js_snippet = """
window.addEventListener('resize', onWindowResize, false);

function onWindowResize() {
    
    const canvas = document.querySelector('canvas');
    const widthAttr = canvas.getAttribute('width');
    const heightAttr = canvas.getAttribute('height');
    const factor = widthAttr/heightAttr
    
    const width = Math.round(0.9*factor*window.innerHeight);
    const height = 0.9*window.innerHeight;   
	    
    camera.aspect = width / height;
    camera.updateProjectionMatrix();
    renderer.setSize(width, height);

    // ---- GUI WIDTH FIX ----
    const canvasWidth = renderer.domElement.clientWidth;
    const customContainer = document.getElementById('canvas_container_' + sceneID);

    if (customContainer) {
        try {
            customContainer.getElementsByClassName('c')[0].style.width = (canvasWidth - 20) + 'px';
            customContainer.getElementsByClassName('dg main')[0].style.width = canvasWidth + 'px';
            customContainer.getElementsByClassName('slider')[0].style.width = (canvasWidth - 100) + 'px';
        } catch (e) {
            console.warn('GUI resize skipped:', e);
        }
    }
}
"""

css_snippet = """
<style>
    html, body {
        margin: 0;
        padding: 0;
        overflow: hidden;
        width: 100%;
        height: 100%;
    }

    canvas {
        display: block;
        width: 100%;
        height: 100%;
    }
</style>
"""

modified_files = []
skipped_files = []

# Process all matching HTML files
for root, dirs, files in os.walk("."):
    for file in files:
        #if re.match(r"part_\d+_\d+\.html", file):
        if re.match(r"intro_\d+.html", file):
            filepath = os.path.join(root, file)
            with open(filepath, 'r', encoding='utf-8') as f:
                content = f.read()

            # Check if already modified
            #if "onWindowResize" in content and "html, body" in content and "canvas" in content:
            #    skipped_files.append(filepath)
            #    continue

            # Insert JS snippet before the LAST </script>
            matches = list(re.finditer(r"</script>", content, flags=re.IGNORECASE))
            if matches:
                last_match = matches[-1]
                insert_pos = last_match.start()
                content = content[:insert_pos] + js_snippet + "\n" + content[insert_pos:]

            # Insert CSS snippet before the first </body> or </html> (whichever comes first)
            content = re.sub(
                r"(</body>|</html>)",
                css_snippet + r"\n\1",
                content,
                count=1,
                flags=re.IGNORECASE | re.DOTALL
            )

            # Write the modified content back to the file
            with open(filepath, 'w', encoding='utf-8') as f:
                f.write(content)

            modified_files.append(filepath)

print(modified_files)


