import os

folder = "/Users/ceabyfernandez/bachelorsthesis/statistics_doublecompressed/qcn_restored/9070"

for filename in os.listdir(folder):
    # Skip directories
    if os.path.isdir(os.path.join(folder, filename)):
        continue

    # Split name and extension
    name, ext = os.path.splitext(filename)
    
    # Keep only the first 4 characters
    new_name = name[:4] + ext
    
    # Rename the file
    os.rename(
        os.path.join(folder, filename),
        os.path.join(folder, new_name)
    )

print("Renaming complete.")