
def filter_lines(input_filename, output_filename):
    with open(input_filename, 'r') as infile, open(output_filename, 'w') as outfile:
        for line in infile:
            if len(line.split()) >= 7:
                outfile.write(line + '\n')

input_filename = 'cisco_docs.txt'
output_filename = 'filtered_cisco_docs.txt'


if __name__ == '__main__':
    filter_lines(input_filename, output_filename)
