import asyncio
import os
from playwright.async_api import async_playwright

async def run():
    # create Folder for Downloads
    download_dir = os.path.join(os.getcwd(), "downloads")
    os.makedirs(download_dir, exist_ok=True)

    async with async_playwright() as playwright:
        # start Browser
        browser = await playwright.chromium.launch(headless=False) #slow_mo=50 ?
        context = await browser.new_context(
            accept_downloads=True
        )
        page = await context.new_page()
        #context = await browser.new_context(
        #    viewport={'width': 1920, 'height': 1080},
        #    user_agent='Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36',
        #    locale='de-DE',
        #    timezone_id='Europe/Berlin'
        #) ????


        # intital navigation and login
        await page.goto("https://invenio.bundesarchiv.de/invenio/login.xhtml")
        await page.get_by_role("button", name="Suche ohne Anmeldung").click()
        await page.get_by_role("link", name="Schließen").click()
        await page.get_by_text("Film und Filmbegleitmaterial").click()
        await page.get_by_text("R 9346-I Zulassungskarten").click()
        await page.get_by_text("nicht klassifiziert Zulassungskarten deutscher Filmprüfstellen.- Filmprüfstelle").click()
        await page.get_by_role("treeitem", name="nicht klassifiziert").click()

        # go to Bestand R 9346-I Zulassungskarten (nicht klassifziert)
        #page.locator("[id=\"masterLayoutForm:tektonik:tree:0_5\"] > .ui-treenode-content > .ui-tree-toggler").click()
        #page.get_by_text("R 9346-I Zulassungskarten").click()
        #page.get_by_role("treeitem", name="nicht klassifiziert").click()

        # Besser wäre hier wait_for_selector auf das Listen-Element
        await page.wait_for_timeout(2000)

        # collect all Document-links
        document_buttons = await page.get_by_label("Anzeige in neuem Fenster:").all()
        print(f"{len(document_buttons)} Dokumente auf dieser Seite gefunden.")

        # iteratre over documents
        for index, button in enumerate(document_buttons):
            print(f"Verarbeite Dokument {index + 1}/{len(document_buttons)}...")

            #P opup abfangen
            async with page.expect_popup() as popup_info:
                await button.click()
            popup = await popup_info.value
            
            # 4. Verschachtelte Iframes auflösen
            # Nutze frame_locator, das wartet automatisch, bis der Frame existiert
            target_frame = popup.frame_locator('frame[name="digitalisatFrame"]').frame_locator('#digitalisatContainer')

            # Klick auf den ersten Download-Button im Iframe
            await target_frame.get_by_role("button", name="Download").click()

            # 5. Download abfangen und physisch speichern
            async with popup.expect_download() as download_info:
                await target_frame.get_by_role("button", name="Download starten").click()
            
            download = await download_info.value
            
            # Dateinamen generieren (z.B. vom Archiv-Server übernommen)
            file_name = download.suggested_filename
            file_path = os.path.join(download_dir, f"{index}_{file_name}")
            
            await download.save_as(file_path)
            print(f"Gespeichert: {file_path}")

            # Popup schließen, um RAM zu sparen
            await popup.close()

        print("Alle sichtbaren Dokumente heruntergeladen.")
        await context.close()
        await browser.close()

if __name__ == "__main__":
    asyncio.run(run())



