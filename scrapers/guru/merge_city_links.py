import os
def merge_city_links(folder_path="scrapers/guru/links", output_file="scrapers/guru/links/all_links.txt"):
    all_links = set()  #set automatically removes duplicates
    
    files = [f for f in os.listdir(folder_path) if f.endswith("_links.txt")]
    print(f"Found {len(files)} city files: {files}")

    for file_name in files:
        file_path = os.path.join(folder_path, file_name)
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                links = f.read().splitlines()
                #filter empty lines and add to the set
                valid_links = [link.strip() for link in links if link.strip()]
                all_links.update(valid_links)
                print(f"Loaded {len(valid_links)} links from {file_name}")
        except Exception as e:
            print(f"❌ Error reading {file_name}: {e}")

    with open(output_file, "w", encoding="utf-8") as f:
        for link in sorted(all_links):
            f.write(link + "\n")

    print(f"\n--- MERGE COMPLETE ---")
    print(f"total unique restaurants for scraping: {len(all_links)}")

merge_city_links()