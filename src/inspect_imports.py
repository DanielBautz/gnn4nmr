#!/usr/bin/env python3
"""
Dieses Skript inspiziert die Import-Anweisungen in main.py und gibt sie aus.
"""

import os
import re
import sys

def inspect_imports(file_path):
    """Untersucht die Import-Anweisungen in einer Datei und gibt sie aus."""
    print(f"Untersuche Importe in {file_path}")
    
    if not os.path.exists(file_path):
        print(f"Fehler: Datei {file_path} existiert nicht!")
        return
    
    with open(file_path, 'r') as f:
        content = f.read()
    
    # Suche nach Import-Anweisungen
    import_pattern = r'^\s*(import|from)\s+([^\s]+).*$'
    imports = []
    
    for line in content.split('\n'):
        match = re.match(import_pattern, line)
        if match:
            imports.append(line.strip())
    
    print("Gefundene Import-Anweisungen:")
    for i, imp in enumerate(imports, 1):
        print(f"{i}. {imp}")
    
    # Zeige die ersten 15 Zeilen des Skripts
    print("\nDie ersten 15 Zeilen der Datei:")
    for i, line in enumerate(content.split('\n')[:15], 1):
        print(f"{i}: {line}")

if __name__ == "__main__":
    # Standard: Untersuche main.py im selben Verzeichnis
    main_script = os.path.join(os.path.dirname(__file__), "main.py")
    
    # Wenn ein Argument übergeben wurde, verwende dieses als Pfad
    if len(sys.argv) > 1:
        main_script = sys.argv[1]
    
    inspect_imports(main_script)