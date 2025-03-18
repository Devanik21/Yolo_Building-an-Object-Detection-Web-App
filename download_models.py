import os
import requests
import sys

def download_file(url, filename):
    """Download a file from the given URL"""
    if os.path.exists(filename):
        print(f"{filename} already exists.")
        return
    
    try:
        print(f"Downloading {filename}...")
        response = requests.get(url, stream=True)
        total_size = int(response.headers.get('content-length', 0))
        
        with open(filename, 'wb') as file:
            if total_size == 0:
                file.write(response.content)
            else:
                downloaded = 0
                total_size_mb = total_size / (1024 * 1024)
                for data in response.iter_content(chunk_size=4096):
                    downloaded += len(data)
                    file.write(data)
                    done = int(50 * downloaded / total_size)
                    sys.stdout.write(f"\r[{'=' * done}{' ' * (50-done)}] {downloaded/(1024*1024):.2f}/{total_size_mb:.2f} MB")
                    sys.stdout.flush()
        print("\nDownload complete!")
    except Exception as e:
        print(f"Error downloading {filename}: {e}")
        if os.path.exists(filename):
            os.remove(filename)

def main():
    """Download all required model files"""
    print("Downloading model files...")
    
    # Create models directory if it doesn't exist
    os.makedirs("models", exist_ok=True)
    
    # YOLO files
    download_file("https://pjreddie.com/media/files/yolov3.weights", "yolov3.weights")
    download_file("https://raw.githubusercontent.com/pjreddie/darknet/master/cfg/yolov3.cfg", "yolov3.cfg")
    
    # SSD MobileNet files
    download_file("https://github.com/chuanqi305/MobileNet-SSD/raw/master/mobilenet_iter_73000.caffemodel", "ssd_mobilenet.caffemodel")
    download_file("https://raw.githubusercontent.com/chuanqi305/MobileNet-SSD/master/deploy.prototxt", "ssd_mobilenet.prototxt")
    
    # Class names
    download_file("https://raw.githubusercontent.com/pjreddie/darknet/master/data/coco.names", "coco.names")
    
    print("All model files have been downloaded!")

if __name__ == "__main__":
    main()
