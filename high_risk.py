
import urllib.request
import tempfile

def download_file(url):
    """Download file from URL"""
    temp_file = tempfile.NamedTemporaryFile(delete=False)
    urllib.request.urlretrieve(url, temp_file.name)
    return temp_file.name

# Example usage (commented out for safety)
# download_file("http://example.com/file.exe")