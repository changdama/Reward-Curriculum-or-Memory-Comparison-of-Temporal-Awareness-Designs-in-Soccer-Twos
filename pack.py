import zipfile
import os

with zipfile.ZipFile('Group12_AGENT.zip', 'w', zipfile.ZIP_DEFLATED) as zf:
    for root, dirs, files in os.walk('Group12_AGENT'):
        # 跳过 __pycache__
        dirs[:] = [d for d in dirs if d != '__pycache__']
        for file in files:
            if file.endswith('.pyc'):
                continue
            filepath = os.path.join(root, file)
            arcname = filepath.replace(os.sep, '/')
            zf.write(filepath, arcname)
            print('Added:', arcname)
print('Done!')
