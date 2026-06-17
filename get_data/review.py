import os
import re
from itertools import groupby

directory = 'S:/Zulassungskarten_Data/R_9346-I_Zulassungskarten'
pattern = re.compile(r'_(\d+)$')

existing_numbers = set()

for filename in os.listdir(directory):
    if os.path.isdir(os.path.join(directory, filename)):
        match = pattern.search(filename)
        if match:
            existing_numbers.add(int(match.group(1)))

if not existing_numbers:
    print("Keine passenden Ordner gefunden.")
else:
    min_num = min(existing_numbers)
    max_num = max(existing_numbers)
    
    expected_numbers = set(range(min_num, max_num + 1))
    missing_numbers = sorted(expected_numbers - existing_numbers)
    
    if missing_numbers:
        ranges = []
        for _, g in groupby(enumerate(missing_numbers), lambda ix: ix[0] - ix[1]):
            group = list(map(lambda x: x[1], g))
            if len(group) == 1:
                ranges.append(str(group[0]))
            else:
                ranges.append(f"{group[0]}-{group[-1]}")

        formatted_missing = ", ".join(ranges)

        print(f"Es fehlen {len(missing_numbers)} Ordner.")
        print(f"Fehlende Nummern: {formatted_missing}")
    else:
        print("Die Nummerierung ist vollständig durchgehend.")