from bing_image_downloader import downloader

fruits = ["apple fruit", "banana fruit", "orange fruit"]

for fruit in fruits:
    downloader.download(
        fruit,
        limit=30,
        output_dir="data/train",
        adult_filter_off=True,
        force_replace=False,
        timeout=60
    )
