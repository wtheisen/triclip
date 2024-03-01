with open('./video_captions.csv', 'r') as f:
    lines = [line.rstrip() for line in f]

fixed_lines = ["id,video_path,caption"]

for line in lines[1:]:
    split_line = line.split('.mp4,')

    if split_line[1][0] != '"':
        split_line[1] = '"' + split_line[1]

    if split_line[1][-1] != '"':
        split_line[1] = split_line[1] + '"'

    fixed_lines.append(split_line[0] + '.mp4,' + split_line[1])

with open('fixed_video_captions.csv', 'w') as f:
    for line in fixed_lines:
        f.write(f"{line}\n")

