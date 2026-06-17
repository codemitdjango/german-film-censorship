import asyncio
import os
import json
from datetime import datetime
from playwright.async_api import async_playwright

# Hilfsfunktion: Wartet, bis blockierende Lade-Overlays verschwinden
async def wait_for_overlays(page_or_popup):
    try:
        # Fängt das allgemeine Lade-Modal ab
        loading_modal = page_or_popup.locator("#loading_modal")
        if await loading_modal.is_visible():
            await loading_modal.wait_for(state="hidden", timeout=20000)
        
        # Fängt das Session-Timeout-Modal ab, falls es auftaucht
        session_modal = page_or_popup.locator("[id*='sessionExpiresMessageDialog_modal']")
        if await session_modal.is_visible():
            await session_modal.wait_for(state="hidden", timeout=5000)
    except Exception:
        pass  # Wenn es bereits weg ist oder nicht existiert, ignorieren

# Führt den Login und die Grundnavigation durch
async def login_and_base_navigation(page):
    print(f"[{datetime.now().strftime('%H:%M:%S')}] Starte Login-Prozess...")
    await page.goto("https://invenio.bundesarchiv.de/invenio/login.xhtml", wait_until="networkidle")
    
    # Falls wir bereits eingeloggt sind oder die Schließen-Schaltfläche da ist
    try:
        await page.get_by_role("button", name="Suche ohne Anmeldung").click(timeout=5000)
        await page.get_by_role("link", name="Schließen").click(timeout=5000)
    except Exception:
        pass

    await page.get_by_text("Film und Filmbegleitmaterial").click()
    await page.wait_for_timeout(1000)

# Überprüft den Session-Status und stellt ihn bei Bedarf wieder her
async def ensure_session(page, collection):
    try:
        # Falls die Page oder der Context bereits im Schließen begriffen ist, abfangen
        if page.is_closed():
            return False
            
        if "login.xhtml" in page.url or await page.locator("[id='sessionExpiresMessageDialog']").is_visible():
            print(f"\n[{datetime.now().strftime('%H:%M:%S')}] Session abgelaufen. Erneuere Verbindung...")
            await login_and_base_navigation(page)
            
            for step in collection["navigation_steps"]:
                if step["type"] == "text":
                    await page.get_by_text(step["value"]).click()
                elif step["type"] == "exact_text":
                    await page.get_by_text(step["value"], exact=True).click()
                elif step["type"] == "role":
                    await page.get_by_role(step["role"], name=step["name"]).click()
                await page.wait_for_timeout(1500)
            await page.wait_for_timeout(2000)
            return True
    except Exception as e:
        # Verhindert, dass ein "Target closed" Fehler das Hauptskript killt
        print(f"Session-Check ignoriert (Verbindung im Umbruch): {e}")
    return False

# Iteriert durch die Seiten und lädt Dokumente herunter
async def download_collection_pages(page, download_dir, start_page, end_pages, collection, log_file_path):
    collection_name = collection["name"]
    collection_dir = os.path.join(download_dir, collection_name)
    os.makedirs(collection_dir, exist_ok=True)

    print(f"\nStarte Download für {collection_name} -> {collection_dir}")

    for current_page in range(start_page, end_pages + 1):
        print(f"\n--- Verarbeite Seite {current_page} von {end_pages} ---")

        # Session-Check vor Seitenwechsel
        await ensure_session(page, collection)

        if current_page > 1:
            dropdown_locator = page.locator("[id=\"masterLayoutForm:tabPanel:tabSearchNavi:selectPageList\"]")
            await dropdown_locator.select_option(str(current_page))
            await page.wait_for_timeout(4000)
            await wait_for_overlays(page)

        # Dokumenten-Links ermitteln
        document_items = await page.get_by_role("listitem").filter(has=page.get_by_label("Anzeige in neuem Fenster:")).all()
        print(f"Gefunden: {len(document_items)} Dokumente auf Seite {current_page}.")

        for index, item in enumerate(document_items):
            print(f"Dokument {index + 1}/{len(document_items)} (Seite {current_page})...")
            popup = None
            try:
                # Vor jedem Dokument prüfen, ob Session noch lebt
                await ensure_session(page, collection)
                await wait_for_overlays(page)

                # Metadaten extrahieren
                title_element = item.locator("xpath=preceding-sibling::*[1]")
                title_text = await title_element.inner_text()
                meta_text = await item.inner_text()
                meta_lines = [line.strip() for line in meta_text.splitlines() if line.strip()]

                document_data = {"Titel_und_Signatur": title_text.strip()}
                for i in range(0, len(meta_lines) - 1, 2):
                    key = meta_lines[i]
                    if key in ["Link kopieren", "Digitalisat anzeigen"]:
                        continue
                    document_data[key] = meta_lines[i + 1]

                # Popup öffnen
                button = item.get_by_label("Anzeige in neuem Fenster:")
                async with page.expect_popup() as popup_info:
                    await button.click(force=True)  # force verhindert Intercept-Fehler
                popup = await popup_info.value
                
                # Warten bis Popup stabil geladen ist
                await popup.wait_for_load_state("networkidle")
                await wait_for_overlays(popup)

                # Iframes anvisieren
                target_frame = popup.frame_locator('frame[name="digitalisatFrame"]').frame_locator('#digitalisatContainer')

                # Download-Menü öffnen
                btn_download_menu = target_frame.get_by_role("button", name="Download")
                await btn_download_menu.wait_for(state="visible", timeout=20000)
                await btn_download_menu.click(force=True)
                
                # Download auslösen
                async with popup.expect_download() as download_info:
                    await target_frame.get_by_role("button", name="Download starten").click(force=True)
                
                download = await download_info.value
                
                # Pfade generieren & Speichern
                file_name = download.suggested_filename
                base_name = os.path.splitext(file_name)[0]
                doc_folder = os.path.join(collection_dir, base_name)
                os.makedirs(doc_folder, exist_ok=True)

                file_path = os.path.join(doc_folder, file_name)
                json_file_path = os.path.join(doc_folder, f"{base_name}.json")
                
                await download.save_as(file_path)
                with open(json_file_path, "w", encoding="utf-8") as json_file:
                    json.dump(document_data, json_file, ensure_ascii=False, indent=4)

                print(f"Erfolgreich gespeichert: {base_name}")
                await popup.close()

            except Exception as error:
                print(f"FEHLER bei Dokument {index + 1} auf Seite {current_page}: {error}")
                timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                with open(log_file_path, "a", encoding="utf-8") as log_file:
                    log_file.write(f"[{timestamp}] ERROR: Collection '{collection_name}', Page {current_page}, Document Index {index + 1}. Details: {error}\n")
                
                if popup and not popup.is_closed():
                    await popup.close()

# Worker für einen Seiten-Chunk
async def process_chunk(browser, collection, start_page, end_page, base_dir, log_file_path, semaphore):
    async with semaphore:
        context = await browser.new_context(accept_downloads=True)
        page = await context.new_page()
        page.set_default_timeout(45000)  # Höheres Timeout für träge Server-Antworten
        
        try:
            await login_and_base_navigation(page)
            
            navigation_failed = False
            for step in collection["navigation_steps"]:
                try:
                    if step["type"] == "text":
                        await page.get_by_text(step["value"]).click()
                    elif step["type"] == "exact_text":
                        await page.get_by_text(step["value"], exact=True).click()
                    elif step["type"] == "role":
                        await page.get_by_role(step["role"], name=step["name"]).click()
                    await page.wait_for_timeout(2000)
                    await wait_for_overlays(page)
                except Exception as error:
                    print(f"NAVIGATION ERROR Worker [{start_page}-{end_page}]: {error}")
                    navigation_failed = True
                    break 

            if not navigation_failed:
                await page.wait_for_timeout(2000)
                await download_collection_pages(
                    page=page,
                    download_dir=base_dir,
                    start_page=start_page,
                    end_pages=end_page,
                    collection=collection,
                    log_file_path=log_file_path
                )
        finally:
            try:
                if not page.is_closed():
                    await context.close()
                print(f"[Worker {start_page}-{end_page}] Context sauber geschlossen.")
            except Exception:
                pass

async def run():
    base_download_dir = r"S:\Zulassungskarten_Data"
    os.makedirs(base_download_dir, exist_ok=True)
    log_file_path = os.path.join(base_download_dir, "error_log.txt")

    collections_to_scrape = [
        {
            "name": "R_9346-I_Zulassungskarten",
            "navigation_steps": [
                {"type": "text", "value": "R 9346-I Zulassungskarten"},
                {"type": "role", "role": "treeitem", "name": "nicht klassifiziert"}
            ],
            "start_page":  255,
            "total_pages": 363
        }
    ]

    # Reduziert auf 2 für höhere Stabilität bei Iframe-Downloads, falls der Bundesarchiv-Server blockiert
    MAX_CONCURRENT_WORKERS = 2 
    CHUNK_SIZE = 50  # Kleinere Chunks verringern das Risiko massiver Datenverluste bei Timeouts
    semaphore = asyncio.Semaphore(MAX_CONCURRENT_WORKERS)

    async with async_playwright() as playwright:
        browser = await playwright.chromium.launch(headless=False)
        
        for collection in collections_to_scrape:
            total_pages = collection["total_pages"]
            start_page_overall = collection.get("start_page", 1)
            
            tasks = []
            for chunk_start in range(start_page_overall, total_pages + 1, CHUNK_SIZE):
                chunk_end = min(chunk_start + CHUNK_SIZE - 1, total_pages)
                
                task = asyncio.create_task(
                    process_chunk(
                        browser=browser,
                        collection=collection,
                        start_page=chunk_start,
                        end_page=chunk_end,
                        base_dir=base_download_dir,
                        log_file_path=log_file_path,
                        semaphore=semaphore
                    )
                )
                tasks.append(task)
            
            print(f"Starte {len(tasks)} parallele Worker für {collection['name']}...")
            await asyncio.gather(*tasks)

        print("\nAlle konfigurierten Downloads abgeschlossen.")
        await browser.close()

if __name__ == "__main__":
    asyncio.run(run())