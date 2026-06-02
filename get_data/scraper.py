import asyncio
import os
from playwright.async_api import async_playwright

async def login_and_base_navigation(page):
    print("Führe initialen Login durch...")
    await page.goto("https://invenio.bundesarchiv.de/invenio/login.xhtml")
    await page.get_by_role("button", name="Suche ohne Anmeldung").click()
    await page.get_by_role("link", name="Schließen").click()
    await page.get_by_text("Film und Filmbegleitmaterial").click()
    # Kurzes Warten, damit der Baum im DOM sicher entfaltet ist
    await page.wait_for_timeout(1000)

async def download_collection_pages(page, download_dir, total_pages, collection_name):
    collection_dir = os.path.join(download_dir, collection_name)
    os.makedirs(collection_dir, exist_ok=True)

    print(f"\nStarte Download für Bestand: {collection_name} in {collection_dir}")

    for current_page in range(1, total_pages + 1):
        print(f"\n--- Verarbeite Seite {current_page} von {total_pages} ---")

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

            try:
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

async def run():
    base_download_dir = os.path.join(os.getcwd(), "downloads")
    os.makedirs(base_download_dir, exist_ok=True)

    collections_to_scrape = [
        {
            "name": "R_9346-I_Zulassungskarten",
            "navigation_steps": [
                {"type": "text", "value": "R 9346-I Zulassungskarten"},
                {"type": "text", "value": "nicht klassifiziert"},
                {"type": "role", "role": "treeitem", "name": "nicht klassifiziert"}
            ],
            "total_pages": 363
        },
        {
            "name": "",
            "navigation_steps": [
                {}
            ]
        },
        {
            "name": "",
            "navigation_steps": [
                {}
            ]
        }
    ]

    async with async_playwright() as playwright:
        browser = await playwright.chromium.launch(headless=False)
        context = await browser.new_context(accept_downloads=True)
        page = await context.new_page()

        await login_and_base_navigation(page)

        for collection in collections_to_scrape:
            for step in collection["navigation_steps"]:
                if step["type"] == "text":
                    await page.get_by_text(step["value"]).click()
                elif step["type"] == "role":
                    await page.get_by_role(step["role"], name=step["name"]).click()
                
                # Kurze Pause zwischen Klicks im Baum, da das Archiv oft langsam nachlädt
                await page.wait_for_timeout(1000)

            # Warten auf die finale Tabelle
            await page.wait_for_timeout(2000)

            await download_collection_pages(
                 page=page,
                 download_dir=base_download_dir,
                 total_pages=collection["total_pages"],
                 collection_name=collection["name"]
            )

            print(f"Bestand {collection['name']} vollständig verarbeitet.")
            # Optional: Hier müsstest du Logik einbauen, um im Baum wieder nach oben zu navigieren, 
            # falls die nächsten Bestände in komplett anderen Zweigen liegen. Bei Bedarf einfach 
            # die Seite neu laden und login_and_base_navigation erneut aufrufen.

        print("Alle sichtbaren Dokumente heruntergeladen.")
        await context.close()
        await browser.close()
        

if __name__ == "__main__":
    asyncio.run(run())



