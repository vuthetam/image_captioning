import os
import importlib
import subprocess
import zipfile
from pathlib import Path

import nltk

# Cài và chọn Java 11.
java_home = Path("/kaggle/working/java11")

if not (java_home / "bin/java").is_file():
    java_home.mkdir(parents=True, exist_ok=True)
    archive = Path("/kaggle/working/java11.tar.gz")

    subprocess.run([
        "curl", "-fL", "--retry", "3",
        "https://api.adoptium.net/v3/binary/latest/11/ga/linux/x64/jre/hotspot/normal/eclipse",
        "-o", str(archive),
    ], check=True)

    subprocess.run([
        "tar", "-xzf", str(archive),
        "-C", str(java_home), "--strip-components=1",
    ], check=True)

os.environ["JAVA_HOME"] = str(java_home)
os.environ["PATH"] = f"{java_home / 'bin'}:{os.environ['PATH']}"

subprocess.run(["java", "-version"], check=True)

# Dữ liệu NLTK cho CHAIR.
for resource in [
    "punkt_tab",
    "averaged_perceptron_tagger_eng",
    "wordnet",
]:
    nltk.download(
        resource,
        download_dir="/root/nltk_data",
        quiet=True,
        raise_on_error=True,
    )

# Tài nguyên Stanford CoreNLP cho SPICE.
spice_module = importlib.import_module("pycocoevalcap.spice.spice")
spice_lib = Path(spice_module.__file__).parent / "lib"
spice_lib.mkdir(parents=True, exist_ok=True)

base_url = (
    "https://repo.maven.apache.org/maven2/"
    "edu/stanford/nlp/stanford-corenlp/3.6.0"
)

for filename in [
    "stanford-corenlp-3.6.0.jar",
    "stanford-corenlp-3.6.0-models.jar",
]:
    jar_path = spice_lib / filename

    if not jar_path.is_file() or not zipfile.is_zipfile(jar_path):
        subprocess.run([
            "curl", "-fL", "--retry", "3",
            f"{base_url}/{filename}", "-o", str(jar_path),
        ], check=True)

    assert zipfile.is_zipfile(jar_path), f"Invalid JAR: {jar_path}"

print("Đã chuẩn bị Java 11 và tài nguyên cho CHAIR, SPICE.")
