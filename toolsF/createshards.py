import os, tarfile, random, io, glob
from PIL import Image

random.seed(0)

def make_shards(img_root, out_dir, prefix, shard_size=2000):
    os.makedirs(out_dir, exist_ok=True)
    # Raccogli tutti i file immagine (stile ImageFolder)
    all_imgs = []
    for cls in sorted(os.listdir(img_root)):
        d = os.path.join(img_root, cls)
        if not os.path.isdir(d): 
            continue
        for f in os.listdir(d):
            if f.lower().endswith((".jpg",".jpeg",".png")):
                all_imgs.append((os.path.join(d,f), cls))
    random.shuffle(all_imgs)
    print(f"[{prefix}] images:", len(all_imgs))
    # Scrivi shard .tar
    for si in range(0, len(all_imgs), shard_size):
        chunk = all_imgs[si:si+shard_size]
        tar_path = os.path.join(out_dir, f"{prefix}-{si//shard_size:05d}.tar")
        with tarfile.open(tar_path, "w") as tar:
            for fp, cls in chunk:
                base = os.path.splitext(os.path.basename(fp))[0]
                # add image
                with open(fp, "rb") as fh:
                    data = fh.read()
                info = tarfile.TarInfo(f"{base}.jpg")
                info.size = len(data)
                tar.addfile(info, io.BytesIO(data))
                # add label (text)
                lbytes = (cls+"\n").encode()
                linfo = tarfile.TarInfo(f"{base}.cls")
                linfo.size = len(lbytes)
                tar.addfile(linfo, io.BytesIO(lbytes))
        print("wrote", tar_path)

IMAGENET = "/data/dataset/pytorch_only/imagenet/"             # <-- change me (contains train/ and val/)
OUT      = "/data/dataset/pytorch_only/imagenet-shards/"      # <-- change me (HDD is fine)

make_shards(os.path.join(IMAGENET,"train"), os.path.join(OUT,"train"), "train", shard_size=2000)
make_shards(os.path.join(IMAGENET,"val"),   os.path.join(OUT,"val"),   "val",   shard_size=2000)
print("Done.")

