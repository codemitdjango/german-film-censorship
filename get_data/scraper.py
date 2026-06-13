import asyncio
import os
import json
from datetime import datetime
from playwright.async_api import async_playwright
from pathlib import Path

# perform login and navigate to the root directory for films
async def login_and_base_navigation(page):
    print("Performing initial login...")

    await page.goto("https://invenio.bundesarchiv.de/invenio/login.xhtml")
    await page.get_by_role("button", name="Suche ohne Anmeldung").click()
    await page.get_by_role("link", name="Schließen").click()
    await page.get_by_text("Film und Filmbegleitmaterial").click()

# iterate through all pages of a collection and trigger downloads
async def download_collection_pages(page, download_dir, start_page, total_pages, collection_name, log_file_path):

    # setup collection-specific directory
    collection_dir = os.path.join(download_dir, collection_name)
    os.makedirs(collection_dir, exist_ok=True)

    print(f"\nStarting download for collection: {collection_name} into {collection_dir}")

    # iteratre through each page
    for current_page in range(start_page, total_pages + 1):
        print(f"\n--- Processing page {current_page} of {total_pages} ---")

        # hanlde pagination via dropdown (skip for page 1)
        if current_page > 1:
            dropdown_locator = page.locator("[id=\"masterLayoutForm:tabPanel:tabSearchNavi:selectPageList\"]")
            await dropdown_locator.select_option(str(current_page))
            
            # wait
            await page.wait_for_timeout(4000)

        # identify all document items on the current page
        document_items = await page.get_by_role("listitem").filter(has=page.get_by_label("Anzeige in neuem Fenster:")).all()
        print(f"Found {len(document_items)} documents on page {current_page}.")

        # iterate over each document
        for index, item in enumerate(document_items):
            print(f"Downloading document {index + 1}/{len(document_items)} (Page {current_page})...")

            try:
                # extract metadata
                title_element = item.locator("xpath=preceding-sibling::*[1]")
                title_text = await title_element.inner_text()
                meta_text = await item.inner_text()
                meta_lines = [line.strip() for line in meta_text.splitlines() if line.strip()]

                document_data = {
                    "Titel_und_Signatur": title_text.strip()
                }
                
                # parse metadata lines into JSON structure
                for i in range(0, len(meta_lines) -1, 2):
                    key = meta_lines[i]
                    if key in ["Link kopieren", "Digitalisat anzeigen"]:
                        continue
                    value = meta_lines[i + 1]
                    document_data[key] = value


                # open viewer popup
                button = item.get_by_label("Anzeige in neuem Fenster:")
                async with page.expect_popup() as popup_info:
                    await button.click()
                popup = await popup_info.value
                
                # drill down into nestes iframes
                target_frame = popup.frame_locator('frame[name="digitalisatFrame"]').frame_locator('#digitalisatContainer')

                # locate and click download button
                btn_download_menu = target_frame.get_by_role("button", name="Download")
                await btn_download_menu.wait_for(state="visible", timeout=15000)
                await btn_download_menu.click()

                # execute file download
                async with popup.expect_download() as download_info:
                    await target_frame.get_by_role("button", name="Download starten").click()
                
                download = await download_info.value
                
                # define file paths and save assets
                file_name = download.suggested_filename
                base_name = os.path.splitext(file_name)[0]
                doc_folder = os.path.join(collection_dir, base_name)
                os.makedirs(doc_folder, exist_ok=True)

                file_path = os.path.join(doc_folder, file_name)
                json_file_path = os.path.join(doc_folder, f"{base_name}.json")
                
                await download.save_as(file_path)
                print(f"Saved: {file_path}")
                
                with open (json_file_path, "w", encoding="utf-8") as json_file:
                    json.dump(document_data, json_file, ensure_ascii=False, indent=4)
                print(f"Saved Metadata: {json_file_path}")

                # clean up
                await popup.close()

            except Exception as error:
                print(f"Error downloading document {index + 1} on page {current_page}: {error}")

                # llog error to file
                timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                error_message = f"[{timestamp}] ERROR: Collection '{collection_name}', Page {current_page}, Document Index {index + 1}. Details: {error}\n"
                with open (log_file_path, "a", encoding="utf-8") as log_file:
                    log_file.write(error_message)

                # close popup on failure
                if 'popup' in locals() and not popup.is_closed():
                    await popup.close()

# main workflow execution
async def run():
    # setup base directories
    base_download_dir = r"S:Zulassungskarten_Data"
    os.makedirs(base_download_dir, exist_ok=True)
    log_file_path = os.path.join(base_download_dir, "error_log.txt")

    with open(log_file_path, "a", encoding="utf-8") as log_file:
        log_file.write(f"--- Start new Scraping-Job at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} ---\n")

    # define collections to scrape
    collections_to_scrape = [
        {
            "name": "R_9346-I_Zulassungskarten",
            "navigation_steps": [
                {"type": "text", "value": "R 9346-I Zulassungskarten"},
                {"type": "role", "role": "treeitem", "name": "nicht klassifiziert"}
            ],
            "start_page": 13,
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

    # initialize browser session
    async with async_playwright() as playwright:
        browser = await playwright.chromium.launch(headless=False)
        
        # iteratre over each collection
        for collection in collections_to_scrape:
            print(f"Starting isolated session for collection: {collection['name']}")

            # open new browser context
            context = await browser.new_context(accept_downloads=True)
            page = await context.new_page()

            await login_and_base_navigation(page)

            # follow navigation steps
            navigation_failed = False
            for step in collection["navigation_steps"]:
                try:
                    if step["type"] == "text":
                        await page.get_by_text(step["value"]).click()
                    elif step["type"] == "exact_text":
                        await page.get_by_text(step["value"], exact=True).click()
                    elif step["type"] == "role":
                        await page.get_by_role(step["role"], name=step["name"]).click()

                    await page.wait_for_timeout(1500)
                except Exception as error:
                    print(f"NAVIGATION ERROR in collection {collection['name']}. Step: {step}. Details: {error}")
                    navigation_failed = True
                    break 

            if navigation_failed:
                print(f"Skipping download for {collection['name']} due to navigation errors.")
                await context.close()
                continue

            await page.wait_for_timeout(2000)

            # start downloading process
            await download_collection_pages(
                 page=page,
                 download_dir=base_download_dir,
                 start_page=collection.get("start_page", 1) # otherwise 1
                 total_pages=collection["total_pages"],
                 collection_name=collection["name"],
                 log_file_path=log_file_path
            )

            # cclean up session
            await context.close()
            print(f"Collection {collection['name']} completed. Session terminated.")

        print("\nAll configured collections have been successfully downloaded.")
        await browser.close()    

if __name__ == "__main__":
    asyncio.run(run())