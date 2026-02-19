import csv

nmi_ids = {
    1: "Cross-Context Consistency section",
    2: "Mediation Bias section",
    4: "Scaling Failure section",
    5: "Mediation Bias section",
    7: "Detectability section",
    13: "Perturbation Validation section",
    15: "Cross-Context Consistency section",
    22: "Pseudotime Directionality section",
    23: "Cross-Species Ortholog Transfer section",
    24: "Uncertainty Calibration section",
    26: "Batch and Donor Leakage section",
}

input_path = r"D:\openclaw\mechinterp-bio\biodyn-work\tracking\tracking.csv"
output_path = r"D:\openclaw\mechinterp-bio\biodyn-work\tracking\tracking.csv"

with open(input_path, 'r', encoding='utf-8') as f:
    content = f.read()

lines = content.strip().split('\n')
# Parse manually to preserve structure
new_lines = []
for i, line in enumerate(lines):
    # Remove trailing commas
    stripped = line.rstrip(',')
    if i == 0:
        new_lines.append(stripped + ',Used in NMI paper')
    else:
        # Extract project ID
        proj_id = stripped.split(',')[0]
        try:
            pid = int(proj_id)
            if pid in nmi_ids:
                new_lines.append(stripped + ',Yes - ' + nmi_ids[pid])
            else:
                new_lines.append(stripped + ',')
        except ValueError:
            new_lines.append(stripped + ',')

with open(output_path, 'w', encoding='utf-8', newline='') as f:
    f.write('\n'.join(new_lines) + '\n')

print("Done! Updated tracking.csv with NMI paper column.")
for pid, section in sorted(nmi_ids.items()):
    print(f"  ID {pid}: {section}")
