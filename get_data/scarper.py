# pip install playwright
# playwright install chromiu

import time
import asyncio
from playwright.async_api import async_playwright


url = "https://invenio.bundesarchiv.de/invenio/login.xhtml"
csv_file = ""


async def run_scraper():
    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=False)
        page = await browser.new_page()

        await page.goto(url)

        input("\n [Aktion erforderlich] gehe zu Bestand. Dann drück ENTER... ")

        # locate 'Digitalisat anzeige'
        digitalisat_link = page.locator("a.detail-digitalisatLink")

        if await digitalisat_link.count() > 0:
            print("Öffne Digitalisat-Viewer...")

            # pop up
            async with page.expect_popup() as popup_info:
                
                await digitalisat_link.first.click()

            viewer_page = await popup_info.value

            # wait for Viewer
            await viewer_page.wait_for_load_state("networkidle")

            print(f"Viewer geöffnet: {await viewer_page.title()}")

            viewer_frame_outer = viewer_page.frame(name="digitalisatFrame")

            if viewer_frame_outer is None:
                print("Fehler: Äußerer Frame 'digitalisatFrame' wurde nicht gefunden")
                # abbruch?
            else:
                await viewer_frame_outer.wait_for_load_state("domcontentloaded")

                iframe_element = viewer_frame_outer.locator("#digitalisatContainer")
    
                await iframe_element.wait_for(state="attached")

                viewer_frame_inner = await iframe_element.content_frame()

                if viewer_frame_inner is None:
                    print("Fehler: Innerer iFrame 'digitalisatContainer' wurde nicht gefunden")
                    # abbruch?
                else:
                    await viewer_frame_inner.wait_for_load_state("networkidle")

                    inner_frame_html = await viewer_frame_inner.content()

                    with open("viewer_debug.html", "w", encoding="utf-8") as f:
                        f.write(inner_frame_html)

                    print("HTML des Viewers gespeichert.")

        else:
            print("Kein Digitalisat auf dieser Seite gefunden.")
       

        await browser.close()


if __name__ == "__main__":
    asyncio.run(run_scraper())