"""Build the poster PDF (100 x 80 cm) from poster.html and the figure files, with headless Chrome.

Each figure is taken as figures/FigN.svg if it exists, else figures/FigN.png. Writes sc2bulk_poster.pdf, the
web page's poster image (sc2bulk_poster.png) and its link-preview card (og.png) next to this file."""
import html as html_lib
import json
import re
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
CHROME = "google-chrome"
CM_TO_PX = 96 / 2.54                   # CSS pixels per cm
IMAGE_WIDTH = 2400                     # sc2bulk_poster.png, the poster shown on the web page
CARD = (1200, 630)                     # og.png, the top of the poster as a link-preview card


def figure(n):
    for suffix in ("svg", "png"):
        path = HERE / "figures" / f"Fig{n}.{suffix}"
        if path.exists():
            return path.relative_to(HERE).as_posix()
    sys.exit(f"figures/Fig{n}.svg or .png is missing")


SAFE_CM = 1.5                          # print-safe zone: no text or figure closer than this to any edge
CHECK = """<script>window.onload = () => {
  const cm = 96 / 2.54, main = document.querySelector("main"), box = main.getBoundingClientRect();
  const floor = box.bottom - parseFloat(getComputedStyle(main).paddingBottom);
  const slack = [...main.querySelectorAll("section")].map(s => (floor - s.lastElementChild.getBoundingClientRect().bottom) / cm);
  let edge = Infinity;
  document.querySelectorAll("h1, h2, li, p, img, .refs, .qr, .qr-label").forEach(e => {
    const r = e.getBoundingClientRect();
    edge = Math.min(edge, r.left, r.top, 100 * cm - r.right, 80 * cm - r.bottom);
  });
  document.title = JSON.stringify({slack, edge: edge / cm});
};</script>"""


def check(built, common):
    """Fails if a column runs past its area or anything sits inside the print-safe zone."""
    dom = subprocess.run([*common, "--virtual-time-budget=3000", "--window-size=3780,3024", "--dump-dom", built.as_uri()],
                         check=True, capture_output=True, text=True).stdout
    result = json.loads(html_lib.unescape(re.search(r"<title>(.*?)</title>", dom).group(1)))
    print(f"column slack (cm): {', '.join(f'{s:.2f}' for s in result['slack'])}; nearest edge: {result['edge']:.2f} cm")
    assert min(result["slack"]) >= -0.01, "a column runs past its area"   # a bottom-pinned section ends exactly at 0
    assert result["edge"] >= SAFE_CM - 0.05, f"content is within {SAFE_CM} cm of an edge"


def main():
    html = (HERE / "poster.html").read_text().replace("{fig1}", figure(1)).replace("{fig2}", figure(2)).replace("{fig3}", figure(3))
    built = HERE / "build.html"
    common = [CHROME, "--headless", "--disable-gpu", "--no-sandbox", "--allow-file-access-from-files"]
    built.write_text(html.replace("</body>", CHECK + "</body>"))
    check(built, common)
    built.write_text(html)
    subprocess.run([*common, "--no-pdf-header-footer", f"--print-to-pdf={HERE / 'sc2bulk_poster.pdf'}", built.as_uri()], check=True)
    width, height = round(100 * CM_TO_PX), round(80 * CM_TO_PX)
    subprocess.run([*common, f"--force-device-scale-factor={IMAGE_WIDTH / width}", f"--window-size={width},{height}",
                    f"--screenshot={HERE / 'sc2bulk_poster.png'}", "--hide-scrollbars", built.as_uri()], check=True)
    built.write_text(f'<body style="margin:0"><img src="sc2bulk_poster.png" style="width:{CARD[0]}px;display:block"></body>')   # Chrome scales no lower than 0.5
    subprocess.run([*common, f"--window-size={CARD[0]},{CARD[1]}", f"--screenshot={HERE / 'og.png'}", "--hide-scrollbars",
                    built.as_uri()], check=True)
    built.unlink()


if __name__ == "__main__":
    main()
