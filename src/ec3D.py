import argparse
import subprocess


if __name__ == "__main__":
	parser = argparse.ArgumentParser(description = "Compute the 3D coordinates from Hi-C.")
	parser.add_argument("--cool", help = "Input whole genome Hi-C map, in *.cool format.", required = True)
	parser.add_argument("--ecdna_cycle", help = "Input ecDNA intervals, in *.bed (chr, start, end, orientation) format.", required = True)
	parser.add_argument("--output_prefix", help = "Prefix of output files.", required = True)
	parser.add_argument("--resolution", help = "Bin size.", required = True)
	parser.add_argument("--ref", help = "One of {hg19, hg38, GRCh38, mm10}.", required = True)
	args = parser.parse_args()

	extract_matrix_command = "python3 extract_matrix.py --cool " + args.cool
	extract_matrix_command += " --ecdna_cycle " + args.ecdna_cycle
	extract_matrix_command += " --output_prefix " + args.output_prefix
	extract_matrix_command += " --resolution " + args.resolution
	print (extract_matrix_command)
	p = subprocess.Popen(extract_matrix_command, shell = True)
	p_status = p.wait()

	reconstruct_command = "python3 spatial_structure.py --matrix " + args.output_prefix + "_collapsed_matrix.npy"
	reconstruct_command += (" --annotation " + args.output_prefix + "_annotations.bed")
	reconstruct_command += (" --output_prefix " + args.output_prefix)
	print (reconstruct_command)
	p = subprocess.Popen(reconstruct_command, shell = True)
	p_status = p.wait()

	dup_flag = False
	fp = open(args.output_prefix + "_annotations.bed", 'r')
	for line in fp:
		s = line.strip().split()
		if len(s) > 4:
			dup_flag = True
	fp.close()
	if dup_flag:
		fp = open(args.output_prefix + "_hyperparameters.txt", 'r')
		params = dict()
		for line in fp:
			s = line.strip().split('\t')
			params[s[0]] = s[1]
		fp.close()
		expand_matrix_command = "python3 expand_matrix.py --raw_matrix " + args.output_prefix + "_raw_collapsed_matrix.npy"
		expand_matrix_command += (" --annotation " + args.output_prefix + "_annotations.bed")
		expand_matrix_command += (" --structure " + args.output_prefix + "_coordinates.txt")
		expand_matrix_command += (" --alpha " + params['alpha'])
		expand_matrix_command += (" --beta " + params['beta'])
		expand_matrix_command += (" --output_prefix " + args.output_prefix)
		print (expand_matrix_command)
		p = subprocess.Popen(expand_matrix_command, shell = True)
		p_status = p.wait()
	else:
		expand_matrix_command = "cp " + args.output_prefix + "_collapsed_matrix.npy " + args.output_prefix + "_expanded_matrix.npy"
		print (expand_matrix_command)
		p = subprocess.Popen(expand_matrix_command, shell = True)
		p_status = p.wait()

	si_command = "python3 significant_interactions.py --matrix " + args.output_prefix + "_expanded_matrix.txt"
	si_command += (" --output_prefix " + args.output_prefix)
	print (si_command)
	p = subprocess.Popen(si_command, shell = True)
	p_status = p.wait()

	plt_command_1 = "python3 plot_interactions.py --ecdna_cycle " + args.ecdna_cycle
	plt_command_1 += (" --resolution " + args.resolution)
	plt_command_1 += (" --matrix " + args.output_prefix + "_expanded_matrix.txt")
	plt_command_1 += (" --annotation " + args.output_prefix + "_annotations.bed")
	plt_command_1 += (" --interactions " + args.output_prefix + "_significant_interactions.csv")
	plt_command_1 += (" --output_prefix " + args.output_prefix)
	print (plt_command_1)
	p = subprocess.Popen(plt_command_1, shell = True)
	p_status = p.wait()
	
	plt_command_2 = "python3 plot_structure.py --structure " + args.output_prefix + "_coordinates.txt"
	plt_command_2 += (" --interactions " + args.output_prefix + "_significant_interactions.csv")
	plt_command_2 += (" --clusters " + args.output_prefix + "_clustered_bins.csv")
	plt_command_2 += (" --annotation " + args.output_prefix + "_annotations.bed --ref " + args.ref)
	plt_command_2 += (" --output_prefix " + args.output_prefix)
	print (plt_command_2)
	p = subprocess.Popen(plt_command_2, shell = True)
	p_status = p.wait()

	print ("Finished.")


