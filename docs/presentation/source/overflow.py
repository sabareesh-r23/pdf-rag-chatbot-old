"""Rough text-fit check: wrap each paragraph with real Calibri/Cambria metrics and compare to box height."""
import sys
from pptx import Presentation
from pptx.util import Emu
from PIL import ImageFont
D = "/Applications/Microsoft PowerPoint.app/Contents/Resources/DFonts/"
FONTS = {("Calibri", False): D + "Calibri.ttf", ("Calibri", True): D + "Calibrib.ttf",
         ("Cambria", False): D + "Cambria.ttc", ("Cambria", True): D + "Cambriab.ttf",
         ("Courier New", False): D + "Calibri.ttf", ("Courier New", True): D + "Calibrib.ttf"}
cache = {}
def font(face, bold, size):
    key = (face if (face, bold) in FONTS else "Calibri", bold, size)
    if key not in cache:
        cache[key] = ImageFont.truetype(FONTS[(key[0], bold)], int(round(size * 4)))  # 4x for precision
    return cache[key]
prs = Presentation(sys.argv[1])
W, H = prs.slide_width, prs.slide_height
for n, slide in enumerate(prs.slides, 1):
    for sh in slide.shapes:
        if sh.left is not None and (sh.left + sh.width > W + 9525 or sh.top + sh.height > H + 9525 or sh.left < 0):
            print(f"slide {n}: OUT OF BOUNDS {sh.name}")
        if not sh.has_text_frame or not sh.text_frame.text.strip():
            continue
        box_w = Emu(sh.width).pt; box_h = Emu(sh.height).pt
        if box_w <= 0: continue
        total = 0
        for p in sh.text_frame.paragraphs:
            runs = [r for r in p.runs if r.text]
            size = next((r.font.size.pt for r in runs if r.font.size), 18)
            bold = any(r.font.bold for r in runs)
            face = next((r.font.name for r in runs if r.font.name), "Calibri")
            face = "Cambria" if face and "Cambria" in face else ("Courier New" if face == "Courier New" else "Calibri")
            f = font(face, bold, size)
            avail = box_w - (18 if p.level is not None and any(True for _ in [0]) and p._p.pPr is not None and p._p.pPr.find('{http://schemas.openxmlformats.org/drawingml/2006/main}buChar') is not None else 0)
            for line in "".join(r.text for r in runs).split("\n") or [""]:
                words, cur, lines = line.split(" "), "", 1
                for w in words:
                    trial = (cur + " " + w).strip()
                    if f.getlength(trial) / 4 > avail and cur:
                        lines += 1; cur = w
                    else:
                        cur = trial
                total += lines * size * 1.2
            sa = p.space_after.pt if p.space_after is not None else 0
            total += sa
        if total > box_h * 1.05:
            print(f"slide {n}: overflow {total:.0f}pt > {box_h:.0f}pt  [{sh.name}] {sh.text_frame.text[:60]!r}")
print("check done")
