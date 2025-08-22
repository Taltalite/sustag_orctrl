import os
import pandas as pd
import pickle
import tqdm

head = ['Query_sequence_name', 'Query_sequence_length', 'Query_start', 'Query_end', 'Relative_strand',
        'Target_sequence_name', 'Target_sequence_length', 'Target_start_on_original_strand',
        'Target_end_on_original_strand', 'Number_of_residue_matches', 'Alignment_block_length', 'Mapping_quality',
        'NM', 'ms', 'AS', 'nn', 'tp', 'cm','s1','s2','de','rl','cg']


def dict_gen(input_paf_paths, output_file):
    
    reads_hash_dic = {}

    # depend on minimap2 results to filter reads
    mapping_qulity_threshold = 5

    for inputpath in input_paf_paths:
        print('processing %s' % inputpath)
        filelist = os.listdir(inputpath)
        file_raw_list = list(
            filter(lambda filename: filename[-4:] == '.paf', filelist))
        file_raw_list = list(
            filter(lambda filename: filename[-4:] == '.paf', filelist))
        with tqdm.trange(len(file_raw_list), desc='reading pafs') as tbar:

            for i in range(len(file_raw_list)):
                curfile = file_raw_list[i]
                data = pd.read_csv(os.path.join(inputpath, curfile), sep='\t', on_bad_lines='skip', names=head)
                mapping_q = data['Mapping_quality'].to_list()
                query_s = data['Query_start'].to_list()
                query_e = data['Query_end'].to_list()
                query_len = data['Query_sequence_length'].to_list()
                query_name = data['Query_sequence_name'].to_list()
                target_name = data['Target_sequence_name'].to_list()
                relative_strand = data['Relative_strand'].to_list()

                for j in range(len(mapping_q)):
                    if mapping_q[j] >= mapping_qulity_threshold:
                        if query_name[j] not in reads_hash_dic:
                            reads_hash_dic[query_name[j]] = {
                            'Q_s':query_s[j], 'Q_e':query_e[j], 'Q_l':query_len[j],
                            'T_n':target_name[j], 'M_q':mapping_q[j], 'R_s':relative_strand[j]
                            }
                        elif query_name[j] in reads_hash_dic and reads_hash_dic[query_name[j]]['M_q'] < mapping_q[j]:
                            reads_hash_dic[query_name[j]]['M_q'] = mapping_q[j]
                tbar.update()


    print('writing to file: %s' % output_file)

    cnt=0
    for _ in reads_hash_dic:
        cnt += 1
    print('reads cnt:', cnt)

    df = open(output_file, 'wb')
    pickle.dump(reads_hash_dic, df)
    df.close()
    
if __name__ == "__main__":
    
    # input paf files path
    input_paf_paths = [
        r'/data/biolab-nvme-pool1/lijy/sim_readuntil_paf/CNNLSTMselnet_sustag384_t0.20finetuned_0_5m/'
        ]

    # output dictionary file
    output_file = r'./CNNLSTMselnet_sustag384_t0.20finetuned_0_5m.P'
    
    dict_gen(input_paf_paths, output_file)