import asyncio
import os
import json
from datetime import datetime
from playwright.async_api import async_playwright
from pathlib import Path

# Handles login and navigates to the root film node
async def login_and_base_navigation(page):
    print("Performing initial login...")

    await page.goto("https://invenio.bundesarchiv.de/invenio/login.xhtml")
    await page.get_by_role("button", name="Suche ohne Anmeldung").click()
    await page.get_by_role("link", name="Schließen").click()
    await page.get_by_text("Film und Filmbegleitmaterial").click()

# Iterates through all pages of a specific collection and downloads documents
async def download_collection_pages(page, download_dir, total_pages, collection_name, log_file_path):
    # create specific subfolder for the current collection
    collection_dir = os.path.join(download_dir, collection_name)
    os.makedirs(collection_dir, exist_ok=True)

    print(f"\nStarting download for collection: {collection_name} into {collection_dir}")

    # iterate over all pages
    for current_page in range(1, total_pages + 1):
        print(f"\n--- Processing page {current_page} of {total_pages} ---")

        # hanlde pagination via dropdown (skrip for page 1)
        if current_page > 1:
            dropdown_locator = page.locator("[id=\"masterLayoutForm:tabPanel:tabSearchNavi:selectPageList\"]")
            await dropdown_locator.select_option(str(current_page))
            
            # wait
            await page.wait_for_timeout(4000)


        document_items = await page.get_by_role("listitem").filter(has=page.get_by_label("Anzeige in neuem Fenster:")).all()
        print(f"Found {len(document_items)} documents on page {current_page}.")


        # collect all Document buttons on the current page
        #document_buttons = await page.get_by_label("Anzeige in neuem Fenster:").all()
        #print(f"Found {len(document_buttons)} documents on page {current_page}.")

        # iteratre over documents
        for index, item in enumerate(document_items):
            print(f"Downloading document {index + 1}/{len(document_items)} (Page {current_page})...")

            try:
                # get Meta Data
                title_element = item.locator("xpath=preceding-sibling::*[1]")
                title_text = await title_element.inner_text()
                meta_text = await item.inner_text()
                meta_lines = [line.strip() for line in meta_text.splitlines() if line.strip()]

                document_data = {
                    "Titel_und_Signatur": title_text.strip()
                }
                
                # form JSON
                for i in range(0, len(meta_lines) -1, 2):
                    key = meta_lines[i]

                    # skip unnescessary data
                    if key in ["Link kopieren", "Digitalisat anzeigen"]:
                        continue

                    value = meta_lines[i + 1]
                    document_data[key] = value

                #full_raw_text = f"{title_text}\n{meta_text}"
                #cleaned_metadata = "\n".join([line.strip() for line in full_raw_text.splitlines() if line.strip()])

                # all Buttons 
                button = item.get_by_label("Anzeige in neuem Fenster:")

                # Intercept popup window
                async with page.expect_popup() as popup_info:
                    await button.click()
                popup = await popup_info.value
                
                # resolve nested iframes to locate the actual content
                target_frame = popup.frame_locator('frame[name="digitalisatFrame"]').frame_locator('#digitalisatContainer')

                # click download button
                btn_download_menu = target_frame.get_by_role("button", name="Download")
                await btn_download_menu.wait_for(state="visible", timeout=15000)
                await btn_download_menu.click()

                # intercept the actual file download
                async with popup.expect_download() as download_info:
                    await target_frame.get_by_role("button", name="Download starten").click()
                
                download = await download_info.value
                
                # generate unique filename and save to collection folder
                file_name = download.suggested_filename
                file_path = os.path.join(collection_dir, f"{file_name}")
                
                await download.save_as(file_path)
                print(f"Saved: {file_path}")

                base_name = os.path.splitext(file_name)[0]
                json_file_path = os.path.join(collection_dir, f"{base_name}.json")
                
                with open (json_file_path, "w", encoding="utf-8") as json_file:
                    json.dump(document_data, json_file, ensure_ascii=False, indent=4)
                print(f"Saved Metadata: {json_file_path}")

                # close popup
                await popup.close()

            except Exception as error:
                print(f"Error downloading document {index + 1} on page {current_page}: {error}")

                # log failed downloads
                timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                error_message = f"[{timestamp}] ERROR: Collection '{collection_name}', Page {current_page}, Document Index {index + 1}. Details: {error}\n"
                with open (log_file_path, "a", encoding="utf-8") as log_file:
                    log_file.write(error_message)

                # close popup
                if 'popup' in locals() and not popup.is_closed():
                    await popup.close()

# main & session management
async def run():
    # setup base directory for all downloads
    base_download_dir = str(Path(__file__).parent.parent / "data" / "01_raw")
    os.makedirs(base_download_dir, exist_ok=True)

    log_file_path = os.path.join(base_download_dir, "error_log.txt")

    with open(log_file_path, "a", encoding="utf-8") as log_file:
        log_file.write(f"\n--- Start new Scraping-Job at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} ---\n")

    # configurations of target collections
    collections_to_scrape = [
        {
            "name": "R_9346-I_Zulassungskarten",
            "navigation_steps": [
                {"type": "text", "value": "R 9346-I Zulassungskarten"},
                {"type": "role", "role": "treeitem", "name": "nicht klassifiziert"}
            ],
            "total_pages": 363
        },
        {
            "name": "R_9346-II_Zulassungskarten",
            "navigation_steps": [
                {"type": "text", "value": "R 9346-II Zulassungskarten"},
                {"type": "exact_text", "value": "Zulassungskarten deutscher Filmprüfstellen"},
                {"type": "text", "value": "1 Filmprüfstelle München"}
            ],
            "total_pages": 4
        },
        {
            "name": "R_9346-IV_Zulassungskarten",
            "navigation_steps": [
                {"type": "text", "value": "R 9346-IV Zulassungskarten"},
                {"type": "exact_text", "value": "Zulassungskarten deutscher Filmprüfstellen"},
                {"type": "text", "value": "Polizeipräsident von Berlin"}
            ],
            "total_pages": 13
        }
    ]

    async with async_playwright() as playwright:
        browser = await playwright.chromium.launch(headless=False)
        
        # process each collection in an isolated browser context
        for collection in collections_to_scrape:
            print(f"Starting isolated session for collection: {collection['name']}")

            # open fresh context 
            context = await browser.new_context(accept_downloads=True)
            page = await context.new_page()

            await login_and_base_navigation(page)

            # go through the specific tree branch for this collection
            navigation_failed = False
            for step in collection["navigation_steps"]:
                try:
                    if step["type"] == "text":
                        await page.get_by_text(step["value"]).click()
                    elif step["type"] == "exact_text":
                        await page.get_by_text(step["value"], exact=True).click()
                    elif step["type"] == "role":
                        await page.get_by_role(step["role"], name=step["name"]).click()
                
                    # wait
                    await page.wait_for_timeout(1500)

                except Exception as error:
                    print(f"NAVIGATION ERROR in collection {collection['name']}. Step: {step}. Details: {error}")
                    navigation_failed = True
                    break 

            if navigation_failed:
                print(f"Skipping download for {collection['name']} due to navigation errors.")
                await context.close()
                continue

            # wait
            await page.wait_for_timeout(2000)

            # start download routine
            await download_collection_pages(
                 page=page,
                 download_dir=base_download_dir,
                 total_pages=collection["total_pages"],
                 collection_name=collection["name"],
                 log_file_path=log_file_path
            )

            # close context to clear session
            await context.close()
            print(f"Collection {collection['name']} completed. Session terminated.")

        print("\nAll configured collections have been successfully downloaded.")
        await browser.close()    

if __name__ == "__main__":
    asyncio.run(run())