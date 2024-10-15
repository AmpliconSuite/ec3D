import sys

if __name__ == "__main__":
	bins = dict()
	res = -1
	fp = open(sys.argv[1], 'r')
	for line in fp:
		s = line.strip().split('\t')
		if res < 0:
			res = int(s[2]) - int(s[1])
		for i in range(3, len(s)):
			bins[int(s[i])] = [s[0], int(s[1])]
	fp.close()
	print("Loaded ecDNA matrix annotations.")

	clusters = dict()
	fp = open(sys.argv[2], 'r')
	for line in fp:
		s = line.strip().split('\t')
		if s[0] == "bin":
			continue
		try:
			clusters[int(s[1])].append(int(s[0]))
		except:
			clusters[int(s[1])] = [int(s[0])]
	fp.close()
	print("Loaded clusters of bins.")

	si = set([])
	fp = open(sys.argv[3], 'r')
	for line in fp:
		s = line.strip().split('\t')
		if s[0] == "bin1":
			continue
		si.add((int(s[0]), int(s[1])))
	fp.close()
	print("Loaded significant interactions.")

	clusters_merged = dict()
	connections = dict()
	for c in clusters.keys():
		s, e = -1, -1
		for bin in clusters[c]:
			if s == -1:
				s = bin
				e = bin
			elif bin > e + 2:
				try:
					clusters_merged[c].append((s, e))
				except:
					clusters_merged[c] = [(s, e)]
				s = bin
				e = bin
			else:
				e = bin
		try:
			clusters_merged[c].append((s, e))
		except:
			clusters_merged[c] = [(s, e)]

	for (i, j) in si:
		for c in clusters_merged.keys():
			for (s1, e1) in clusters_merged[c]:
				for (s2, e2) in clusters_merged[c]:
					if s1 <= i <= e1 and s2 <= j <= e2 and (s1 != s2 or e1 != e2):
						i1 = clusters_merged[c].index((s1, e1))
						i2 = clusters_merged[c].index((s2, e2))
						try:
							connections[c].append((i1, i2))
						except:
							connections[c] = [(i1, i2)]

	fp_w = open(sys.argv[4], 'w')
	fp_w.write("cluster\tinterval\tstart\tend\tconnections\n")
	j = 0
	for c in sorted(clusters_merged.keys()):
		for i in range(len(clusters_merged[c])):
			fp_w.write("%d\tI%d\t%d\t%d" %(c, j + i, clusters_merged[c][i][0], clusters_merged[c][i][1]))
			ci_list = []
			if c in connections:
				for ci in range(len(connections[c])):
					if connections[c][ci][0] == i and connections[c][ci][1] not in ci_list:
						ci_list.append(connections[c][ci][1])
					if connections[c][ci][1] == i and connections[c][ci][0] not in ci_list:
						ci_list.append(connections[c][ci][0])
			ci_list = sorted(ci_list)
			if len(ci_list) > 0:
				fp_w.write("\t")
				for ci_item in ci_list[:-1]:
					fp_w.write("I%d," %(ci_item + j))
				fp_w.write("I%d\n" %(ci_list[-1] + j))
			else:
				fp_w.write("\n")
		j += len(clusters_merged[c])

	fp_w.close()
	print("Wrote intervals of clusters to %s." %sys.argv[4])
	print("Finished.")


