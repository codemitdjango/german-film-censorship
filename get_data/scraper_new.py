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

        # go to Bestand R R 9346-I Zulassungskarten (nicht klassifziert)
        await page.get_by_text("R 9346-I Zulassungskarten").click()
        await page.get_by_text("nicht klassifiziert Zulassungskarten deutscher Filmprüfstellen.- Filmprüfstelle").click()
        await page.get_by_role("treeitem", name="nicht klassifiziert").click()

        # Besser wäre hier wait_for_selector auf das Listen-Element
        await page.wait_for_timeout(2000)

        total_pages = 363

        for current_page in range(1, total_pages + 1):
            print(f"\n--- Verarbeite Seite {current_page} von {total_pages} ---")

            try: 
                if current_page > 1:
                    # Navigiere zur nächsten Seite via Dropdown
                    dropdown_locator = page.locator("[id=\"masterLayoutForm:tabPanel:tabSearchNavi:selectPageList\"]")
                    await dropdown_locator.select_option(str(current_page))
                    
                    # Zwingend warten, bis die neuen Dokumente in den DOM geladen wurden
                    await page.wait_for_timeout(4000)

                # collect all Document buttons on current page
                document_buttons = await page.get_by_label("Anzeige in neuem Fenster:").all()
                print(f"{len(document_buttons)} Dokumente auf Seite {current_page} gefunden.")  

                # iteratre over documents
                for index, button in enumerate(document_buttons):
                    print(f"Lade Dokument {index + 1}/{len(document_buttons)} (Seite {current_page})...")

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

            except Exception as error:
                print(f"Fehler bei Dokument {index + 1} auf Seite {current_page}: {error}")
                # Sicherstellen, dass das Popup im Fehlerfall geschlossen wird, sonst crasht der Browser irgendwann durch OOM (Out of Memory)
                if 'popup' in locals() and not popup.is_closed():
                    await popup.close()


        print("Alle sichtbaren Dokumente heruntergeladen.")
        await context.close()
        await browser.close()

if __name__ == "__main__":
    asyncio.run(run())



