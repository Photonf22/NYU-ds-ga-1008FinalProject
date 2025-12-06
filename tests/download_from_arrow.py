import os
from glob import glob

from datasets import Dataset, Image

def main():
    arrow_files = sorted(glob("*.arrow"))
    os.makedirs("extracted_images", exist_ok=True)

    for arrow_path in arrow_files:
        print(f"Processing {arrow_path}")
        ds = Dataset.from_file(arrow_path)  # memory-mapped Arrow file [web:15][web:24]

        # Get column names; assume first is image, second is label
        col_names = list(ds.features.keys())
        img_col, label_col = col_names[0], col_names[1]

        # Ensure image column is decoded as PIL.Image
        ds = ds.cast_column(img_col, Image())  # [web:1][web:9]

        for idx, example in enumerate(ds):
            img = example[img_col]          # PIL.Image
            label = str(example[label_col]) # ensure string

            # sanitize label for filename if needed
            safe_label = "".join(
                c if c.isalnum() or c in ("-", "_") else "_"
                for c in label
            )

            filename = f"{safe_label}_{idx:06d}.jpg"
            out_path = os.path.join("extracted_images", filename)

            img.save(out_path, format="JPEG")

if __name__ == "__main__":
    main()

