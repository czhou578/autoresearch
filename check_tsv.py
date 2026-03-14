with open('results.tsv', 'r') as f:
    for i, line in enumerate(f):
        print(f"Line {i+1}: {repr(line)}")
