import os
import random
import matplotlib.pyplot as plt
import matplotlib.patches as patches

from dataset import PotholeDataset

DATA_ROOT = "/dtu/datasets1/02516/potholes"

def save_random_samples(n=10, out_dir="samples"):
    # lag datasettet
    ds = PotholeDataset(DATA_ROOT)
    total = len(ds)
    print(f"Datasettet har {total} bilder.")

    # lag output-mappen hvis den ikke finnes
    os.makedirs(out_dir, exist_ok=True)

    # trekk tilfeldige indekser
    idxs = random.sample(range(total), n)

    for i, idx in enumerate(idxs):
        img, boxes = ds[idx]

        fig, ax = plt.subplots(1)
        ax.imshow(img)

        for box in boxes:
            cls, xmin, ymin, xmax, ymax = box
            w = xmax - xmin
            h = ymax - ymin

            rect = patches.Rectangle(
                (xmin, ymin),
                w,
                h,
                linewidth=2,
                edgecolor="r",
                facecolor="none",
            )
            ax.add_patch(rect)
            ax.text(xmin, ymin - 2, cls, color="r")

        plt.axis("off")
        fname = f"sample_{i}.png"
        save_path = os.path.join(out_dir, fname)
        plt.savefig(save_path, bbox_inches="tight")
        plt.close()

        print(f"Lagret: {save_path}")

    print("\nFerdig! Bildene ligger i mappen:", os.path.abspath(out_dir))


if __name__ == "__main__":
    save_random_samples(n=10)
