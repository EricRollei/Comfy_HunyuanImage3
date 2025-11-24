import math

def get_res(mp, ratio_w, ratio_h):
    target_pixels = mp * 1_000_000
    ratio = ratio_w / ratio_h
    
    # w * h = target
    # w / h = ratio  => w = h * ratio
    # h * ratio * h = target => h^2 = target / ratio => h = sqrt(target/ratio)
    
    h = math.sqrt(target_pixels / ratio)
    w = h * ratio
    
    # Round to nearest 64 (based on image_resolution_align in config.json)
    h = round(h / 64) * 64
    w = round(w / 64) * 64
    
    actual_mp = (w * h) / 1_000_000
    return w, h, actual_mp

ratios = [
    (9, 16, "9:16"),
    (5, 8, "5:8"),
    (2, 3, "2:3"),
    (3, 4, "3:4"),
    (4, 5, "4:5"),
    (1, 1, "1:1"),
    (5, 4, "5:4"),
    (4, 3, "4:3"),
    (3, 2, "3:2"),
    (8, 5, "8:5"),
    (16, 9, "16:9"),
]

mps = [1, 2, 3]

print("        resolutions = [")
for mp in mps:
    print(f"            # {mp}MP Class")
    for rw, rh, label in ratios:
        w, h, actual = get_res(mp, rw, rh)
        print(f'            ({w}, {h}, "{label} ({actual:.1f}MP)"),')
    print("")
print("        ]")
