file = '../results/test.en-ru'

with open(file) as infile, open('test.en-ru', 'w') as outfile:
    for line in infile:
        if not line.strip(): continue  # skip the empty line
        outfile.write(line)  # non-empty line. Write it to output