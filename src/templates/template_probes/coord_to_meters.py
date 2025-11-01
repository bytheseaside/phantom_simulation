import csv

# Input and output filenames
INPUT_FILE = "probes_mm.csv"
OUTPUT_FILE = "probes.csv"

def mm_to_m(mm_value):
    return float(mm_value) / 1000.0

with open(INPUT_FILE, newline='') as infile, open(OUTPUT_FILE, 'w', newline='') as outfile:
    reader = csv.DictReader(infile)
    fieldnames = reader.fieldnames
    writer = csv.DictWriter(outfile, fieldnames=fieldnames)
    writer.writeheader()

    for row in reader:
        # Convert x, y, z columns from mm → m
        for axis in ['x', 'y', 'z']:
            if row[axis].strip() != "":
                row[axis] = f"{mm_to_m(row[axis]):.6f}"
        writer.writerow(row)

print(f"✅ Converted coordinates written to {OUTPUT_FILE}")
