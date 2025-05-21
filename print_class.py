import struct
import sys

def find_classification_blocks(filename):
    """Finds blocks between marker values and returns relevant lines."""
    results = []
    collecting = False
    block = []

    with open(filename, 'r') as file:
        prev = None
        for line in file:
            split = line.strip().split()
            prev_split = prev.strip().split() if prev else []

            if len(prev_split) > 6 and len(split) > 6:
                if prev_split[6] == "00000789" and split[6] == "00000102":
                    if collecting:
                        results.append(block)
                        block = []
                        collecting = False
                    else:
                        collecting = True

            if collecting:
                block.append(line.strip())

            prev = line

    if block:
        results.append(block)
    
    return results


def extract_7th_column(lines):
    return [line.strip().split()[6] for line in lines if len(line.strip().split()) > 6]


def hex_to_int(hex_str):
    try:
        return int(hex_str, 16)
    except:
        return None


def main():
    if len(sys.argv) != 3:
        print("Usage: python get_classification.py <type> <format>")
        print("type: V or NV")
        print("format: int")
        return

    typ = sys.argv[1]
    fmt = sys.argv[2]

    fileName = "veer/tempFiles/logV.txt" if typ == "V" else "veer/tempFiles/logNV.txt"
    blocks = find_classification_blocks(fileName)

    if not blocks:
        print("No classification block found.")
        return

    print("Classification results:")

    for block in blocks:
        for line in block:
            if "lw" in line or "flw" in line:  # looks for the load
                val = extract_7th_column([line])[0]
                if fmt == "int":
                    result = hex_to_int(val)
                else:
                    print(f"Unsupported format: {fmt}")
                    return
                print(result)
                break  # only need first load per block


if __name__ == "__main__":
    main()
