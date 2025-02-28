import os
from datetime import datetime


fapath = r'../fasta/barcode96.fa'   # reference fasta

inputpath = r'/data/biolab-nvme-pool1/lijy/guppy_basecalled/sustechseqs_230608/pass/'   # guppybasecalled fast5 files
outputpath = r'/data/biolab-nvme-pool1/lijy/mapping_res/paf_sustechseqs_sustag96/'  # path to save paf files

filelist = os.listdir(inputpath)
os.makedirs(outputpath, exist_ok=True)

minimap2_path = r'/home/lijy/workspace/minimap2-2.24/minimap2'  # minimap2 dir

file_raw_list = list(
    filter(lambda filename: filename[-len('.fastq'):] == '.fastq', filelist))

CUSTOM = True

k_mer_size = 12 # -k[15]
minimizer_window_size = 8 # -w[10]
minimal_chaining_score = 25 # -m[40]
n_discard_chains_minimizers_le = 3 # -n [3]
minimal_peak_DP_alignment_score_to_output  = 40 # -s [40]

threads = 12


if CUSTOM:
    f = open(os.path.join(outputpath,'config.txt'), 'w')
    f.write('k_mer_size: %d\n' % k_mer_size)
    f.write('minimizer_window_size: %d\n' % minimizer_window_size)
    f.write('minimal_chaining_score: %d\n' % minimal_chaining_score)
    f.write('n_discard_chains_minimizers_le: %d\n' % n_discard_chains_minimizers_le)
    f.write('minimal_peak_DP_alignment_score_to_output: %d\n' % minimal_peak_DP_alignment_score_to_output)
    f.close()


print('Start minimap2 mapping... ', flush=True)
print('fasta file: ', fapath)
print('inputpath: ', inputpath)
print('outputpath: ', outputpath)
print('Minimap2 version: ', minimap2_path, flush=True)
print('mapping params: CUSTOM? ', CUSTOM, flush=True)

start_time = datetime.now()
print('start time: ', start_time, flush=True)
print('\n\n')

for i in range(len(file_raw_list)):
    curfile = file_raw_list[i]
    if CUSTOM :
        command = f"{minimap2_path} -c -k{k_mer_size} -w{minimizer_window_size} -n{n_discard_chains_minimizers_le} -m{minimal_chaining_score} " \
                f"-s{minimal_peak_DP_alignment_score_to_output} -t{threads} {fapath} {inputpath+curfile} > {outputpath+curfile}.paf"
    else:
        command = f'{minimap2_path} -cx map-ont {fapath} {inputpath+curfile} > {outputpath+curfile}.paf'
    print('processing %s' % curfile)
    os.system(command)

	# # break for check
    # break

end_time = datetime.now()
duration = end_time - start_time
print('end time: ', end_time, flush=True)
print('duration time: ', duration, flush=True)
