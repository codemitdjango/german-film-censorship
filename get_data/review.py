import os
import re

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
        print(f"Es fehlen {len(missing_numbers)} Ordner.")
        print(f"Fehlende Nummern: {missing_numbers}")
    else:
        print("Die Nummerierung ist vollständig durchgehend.")