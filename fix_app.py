import re, sys
path = "app.py"
s = open(path).read()
if "STRETCH" in s:
    print("app.py already patched"); sys.exit(0)
s = s.replace('width="stretch"', "**STRETCH")
helper = '''

# Streamlit >= 1.50 uses width="stretch"; older versions use use_container_width=True.
def _version_tuple(v):
    parts = []
    for p in v.split(".")[:2]:
        digits = "".join(ch for ch in p if ch.isdigit())
        parts.append(int(digits) if digits else 0)
    return tuple(parts)


STRETCH = (
    {"width": "stretch"}
    if _version_tuple(st.__version__) >= (1, 50)
    else {"use_container_width": True}
)
'''
marker = 'st.set_page_config(page_title="Tennis Analytics", page_icon="🎾", layout="wide")\n'
assert marker in s, "could not find set_page_config line"
s = s.replace(marker, marker + helper)
open(path, "w").write(s)
print("patched app.py")
