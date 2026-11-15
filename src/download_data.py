from urllib.request import urlretrieve
from .config import CYCLE_BASE, COMPONENTS, RAW_DIR


def download_all() -> None:
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    for filename in COMPONENTS.values():
        url = f"{CYCLE_BASE}/{filename}.XPT"
        destination = RAW_DIR / f"{filename}.XPT"
        if destination.exists() and destination.stat().st_size > 0:
            print(f"Already downloaded: {destination.name}")
            continue
        print(f"Downloading {url}")
        urlretrieve(url, destination)
        print(f"Saved {destination}")


if __name__ == "__main__":
    download_all()
