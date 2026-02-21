import urllib.request
import zipfile

url = "https://archive.ics.uci.edu/ml/machine-learning-databases/00228/smsspamcollection.zip"
urllib.request.urlretrieve(url, "spam.zip")

with zipfile.ZipFile("spam.zip", "r") as z:
    z.extractall("data")

print("Dataset downloaded!")
