import zipfile
import io

fname = "Argon ramp 2 low pressures 20.5 mW in.spz"

with zipfile.ZipFile(fname) as z:
    print("Files in zip:", z.namelist())
    for name in z.namelist():
        with z.open(name) as f:
            data = f.read(500)
            try:
                text = data.decode('utf-8', errors='replace')
                print(f"\n--- {name} (first 500 bytes as text) ---")
                print(text)
            except:
                print(f"\n--- {name} (binary) ---")
                print(data[:100])
