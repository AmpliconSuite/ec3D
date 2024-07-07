import argparse
import subprocess


if __name__ == "__main__":
	parser = argparse.ArgumentParser(description = "Compute the 3D coordinates from Hi-C.")
	parser.add_argument("--cool", help = "Input whole genome Hi-C map, in *.cool format.", required = True)
	parser.add_argument("--ecdna_cycle", help = "Input ecDNA intervals, in *.bed (chr, start, end, orientation) format.", required = True)
	parser.add_argument("--output_prefix", help = "Prefix of output files.", required = True)
	parser.add_argument("--resolution", help = "Bin size.", type = int, required = True)

	extract_matrix_command = "python3 extract_matrix.py --cool " + args.cool
	extract_matrix_command += " --ecdna_cycle " + args.ecdna_cycle
	extract_matrix_command += " --output_prefix " + args.output_prefix
	extract_matrix_command += " --resolution " + args.resolution
	print (extract_matrix_command)
	p = subprocess.Popen(extract_matrix_command, shell = True)
	p_status = p.wait()

	3d_struct_command = "python3 spatial_structure.py --cool " + args.cool
	3d_struct_command += " --ecdna_cycle " + args.ecdna_cycle
	3d_struct_command += " --output_prefix " + args.output_prefix
	3d_struct_command += " --resolution " + args.resolution
	print (extract_matrix_command)
	p = subprocess.Popen(extract_matrix_command, shell = True)
	p_status = p.wait()

	fp = open()
	if 
		extract_matrix_command = "python3 extract_matrix.py --cool " + args.cool
		extract_matrix_command += " --ecdna_cycle " + args.ecdna_cycle
		extract_matrix_command += " --output_prefix " + args.output_prefix
		extract_matrix_command += " --resolution " + args.resolution
		print (extract_matrix_command)
		p = subprocess.Popen(extract_matrix_command, shell = True)
		p_status = p.wait()
	else:
		sdd

