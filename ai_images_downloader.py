import gdown
import time
import os

def resume_download():
    max_retries = 5
    retry_delay = 300  # 5 minutes
    
    for attempt in range(max_retries):
        try:
            print(f"Attempt {attempt + 1}/{max_retries}")
            gdown.download_folder(
                'https://drive.google.com/drive/folders/1u2xu7bSrWxrbUxk-dT-UvEJq8IjdmNTP',
                output='data/raw/real/',
                remaining_ok=True,
                quiet=False
            )
            print("Download completed successfully!")
            break
        except Exception as e:
            print(f"Attempt {attempt + 1} failed: {e}")
            if attempt < max_retries - 1:
                print(f"Waiting {retry_delay} seconds before retry...")
                time.sleep(retry_delay)
            else:
                print("All retry attempts failed")

resume_download()