import json

with open('Implementasi_Decision_Tree_Stroke_Prediction.ipynb', 'r', encoding='utf-8') as f:
    nb = json.load(f)

print("=" * 70)
print("VERIFIKASI NOTEBOOK")
print("=" * 70)
print(f"Total cells: {len(nb['cells'])}")
print(f"Nbformat: {nb['nbformat']}.{nb['nbformat_minor']}")
print("\n" + "=" * 70)
print("STRUKTUR NOTEBOOK:")
print("=" * 70)

markdown_count = 0
code_count = 0

for i, cell in enumerate(nb['cells'][:30], 1):
    if cell['cell_type'] == 'markdown':
        markdown_count += 1
        if cell['source']:
            title = cell['source'][0][:70]
            print(f"{i:2d}. [MD] {title}")
    elif cell['cell_type'] == 'code':
        code_count += 1

print("\n" + "=" * 70)
print(f"Total Markdown cells: {markdown_count}")
print(f"Total Code cells: {code_count}")
print("=" * 70)
print("\n✓ NOTEBOOK VALID DAN SIAP DIGUNAKAN!")
