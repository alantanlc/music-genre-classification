import random

# Load entropy file
filename = 'entropy-raw.txt'
with open(filename, 'r') as f:
	content = f.readlines()
content = [x.strip() for x in content]
f.close()

# Genres
genres = ['blues','classical','country','disco','hiphop','jazz','metal','pop','reggae','rock']

# Generate random data set
n_rand = 10
rand_data_set = []
for g in genres:
	# Extract all items of current genre
	genre_data_set = []
	for i in content:
		if i.find(g) != -1:
			genre_data_set.append(i)

	# Pick n-random items from list
	r = random.sample(genre_data_set, n_rand)

	# Append to data_set
	rand_data_set = rand_data_set + r

# Generate arff for each genre using same random data set
for g in genres:
	# Extract all items of current genre
	genre_data_set = []
	for i in content:
		if i.find(g) != -1:
			genre_data_set.append(i)

	# Remove data with current genre from random data set
	rand = []
	for i in rand_data_set:
		if i.find(g) == -1:
			# Modify label
			x = i.split(",")
			x[5] = "not" + g
			y = ",".join(x)

			# Append to rand list
			rand.append(y)

	combined_data_set = genre_data_set + rand

	# Write dataset to file
	filename = 'entropy-' + g + '.arff'
	classes = [g, 'not' + g]
	with open(filename, 'w') as f:

		# Write header
		f.write(
			"@RELATION music_genre\n@ATTRIBUTE AVG_ENT NUMERIC\n@ATTRIBUTE STD_DEV_ENT NUMERIC\n@ATTRIBUTE MAX_ENT NUMERIC\n@ATTRIBUTE MIN_ENT NUMERIC\n@ATTRIBUTE MAX_DIFF_ENT NUMERIC\n@ATTRIBUTE class {" + ",".join(classes) + "}\n\n@DATA\n")

		for item in combined_data_set:
			f.write("{}\n".format(item))
	f.close()