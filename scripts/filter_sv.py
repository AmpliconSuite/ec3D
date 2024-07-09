import argparse


if __name__ == "__main__":
	parser = argparse.ArgumentParser(description = ".")
	parser.add_argument("--ecdna_cycle", help = "Input ecDNA intervals, in *.bed (chr, start, end, orientation) format.", required = True)
	parser.add_argument("--sv_list", help = "List of SVs from AmpliconClassifier output.", required = True)
	parser.add_argument("--bp_match_cutoff", help = "Breakpoint matching cutoff, default = 100.", type = int, default = 100)
	parser.add_argument("--output_fn", help = "Optional, output filename.")
	args = parser.parse_args()

	"""
	Read in ecDNA cycle
	"""
	intrvls = dict()
	bp_list = []
	fp = open(args.ecdna_cycle, 'r')
	for line in fp:
		s = line.strip().split('\t')
		try:
			intrvls[s[4]].append(s)
		except:
			intrvls[s[4]] = [s]
	fp.close()
	for cycle_id in intrvls.keys():
		if intrvls[cycle_id][0][5] == 'True':
			for i in range(-1, len(intrvls[cycle_id]) - 1):
				if intrvls[cycle_id][i][3] == '+' and intrvls[cycle_id][i + 1][3] == '+':
					bp_list.append([intrvls[cycle_id][i][0], intrvls[cycle_id][i][2], '+', intrvls[cycle_id][i + 1][0], intrvls[cycle_id][i + 1][1], '-'])
				elif intrvls[cycle_id][i][3] == '+' and intrvls[cycle_id][i + 1][3] == '-':
					bp_list.append([intrvls[cycle_id][i][0], intrvls[cycle_id][i][2], '+', intrvls[cycle_id][i + 1][0], intrvls[cycle_id][i + 1][2], '+'])
				elif intrvls[cycle_id][i][3] == '-' and intrvls[cycle_id][i + 1][3] == '+':
					bp_list.append([intrvls[cycle_id][i][0], intrvls[cycle_id][i][1], '-', intrvls[cycle_id][i + 1][0], intrvls[cycle_id][i + 1][1], '-'])
				else:
					bp_list.append([intrvls[cycle_id][i][0], intrvls[cycle_id][i][1], '-', intrvls[cycle_id][i + 1][0], intrvls[cycle_id][i + 1][2], '+'])
		else:
			if len(intrvls[cycle_id]) == 1:
				continue
			for i in range(len(intrvls[cycle_id]) - 1):
				if intrvls[cycle_id][i][3] == '+' and intrvls[cycle_id][i + 1][3] == '+':
					bp_list.append([intrvls[cycle_id][i][0], intrvls[cycle_id][i][2], '+', intrvls[cycle_id][i + 1][0], intrvls[cycle_id][i + 1][1], '-'])
				elif intrvls[cycle_id][i][3] == '+' and intrvls[cycle_id][i + 1][3] == '-':
					bp_list.append([intrvls[cycle_id][i][0], intrvls[cycle_id][i][2], '+', intrvls[cycle_id][i + 1][0], intrvls[cycle_id][i + 1][2], '+'])
				elif intrvls[cycle_id][i][3] == '-' and intrvls[cycle_id][i + 1][3] == '+':
					bp_list.append([intrvls[cycle_id][i][0], intrvls[cycle_id][i][1], '-', intrvls[cycle_id][i + 1][0], intrvls[cycle_id][i + 1][1], '-'])
				else:
					bp_list.append([intrvls[cycle_id][i][0], intrvls[cycle_id][i][1], '-', intrvls[cycle_id][i + 1][0], intrvls[cycle_id][i + 1][2], '+'])
	
	"""
	Read in and remove SV breakpoints overlapping with 
	"""
	fp = open(args.sv_list, 'r')
	if args.output_fn:
		fp_w = open(args.output_fn, 'w')
	else:
		fp_w = open(args.sv_list.replace(".tsv", "_filtered.tsv"), 'w')
	i = 0
	for line in fp:
		if i == 0:
			fp_w.write("%s" %line)
		else:
			s = line.strip().split('\t')
			ort = [s[7][0], s[7][1]]
			bp_match = False
			for bp in bp_list:
				if s[0] == bp[0] and ort[0] == bp[2] and s[2] == bp[3] and ort[1] == bp[5] and abs(int(s[1]) - int(bp[1])) <= args.bp_match_cutoff and abs(int(s[3]) - int(bp[4])) <= args.bp_match_cutoff:
					print (s)
					bp_match = True
					break
				if s[0] == bp[3] and ort[0] == bp[5] and s[2] == bp[0] and ort[1] == bp[2] and abs(int(s[1]) - int(bp[4])) <= args.bp_match_cutoff and abs(int(s[1]) - int(bp[1])) <= args.bp_match_cutoff:
					print (s)
					bp_match = True
					break
			if not bp_match:
				fp_w.write("%s" %line)
		i += 1
	fp.close()
	fp_w.close()

