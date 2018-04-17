"""''''''
The aim of the class utils is to provide
all the supporting functions for handle the files
''''''"""


def read_file(path):
    # open the file
    with open(path) as file:
        # populate the list
        lines = [line.strip() for line in file]
    file.close()
    return lines


def write_csv(songs_array, path):
    file = open(path, "w")
    for s in songs_array:
        file.write(s.to_string_a() + "\n")
    file.close()


def write_arff(songs_array, path):
    # header composition
    s = "@RELATION mfcc_features\n"
    s += "@ATTRIBUTE AVG_ENT NUMERIC\n@ATTRIBUTE STD_DEV_ENT NUMERIC\n@ATTRIBUTE MAX_ENT NUMERIC\n" \
         "@ATTRIBUTE MIN_ENT NUMERIC\n@ATTRIBUTE MAX_DIFF_ENT NUMERIC\n"
    for i in range(0, 26):
        s += "@ATTRIBUTE MEAN_MFCC_" + str(i) + " NUMERIC\n"
    # for i in range(0, 26):
    #    s += "@ATTRIBUTE MEAN_DELTA_MFCC_" + str(i) + " NUMERIC\n"
    # for i in range(0, 26):
    #    s += "@ATTRIBUTE MEAN_DELTA2_MFCC_" + str(i) + " NUMERIC\n"
    for i in range(0, 26):
        s += "@ATTRIBUTE STD_MFCC_" + str(i) + " NUMERIC\n"
    for i in range(0, 26):
        s += "@ATTRIBUTE STD_DELTA_MFCC_" + str(i) + " NUMERIC\n"
    for i in range(0, 26):
        s += "@ATTRIBUTE STD_DELTA2_MFCC_" + str(i) + " NUMERIC\n"
    s += "@ATTRIBUTE class {blues,classical,country,disco,hiphop,jazz,metal,pop,reggae,rock}"
    s += "\n"
    s += "\n"
    s += "@DATA\n"
    # data composition
    for song in songs_array:
        s += song.to_string() + "\n"
    file = open(path, "w")
    file.write(s)
    file.close()


def get_label_from_line(line):
    a=line.split("\t")
    return a[1]



def get_filename_from_line(line):
    return line.split("\t")[0]
