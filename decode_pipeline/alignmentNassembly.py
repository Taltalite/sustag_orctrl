import subprocess, copy, tqdm, os
from scipy.spatial import distance
from Bio import SeqIO

def read_file(path):
    with open(path) as handle:
        record = SeqIO.parse(handle,"fasta")
        all_record = [(str(lin.id), str(lin.seq)) for lin in record]
    return all_record

def assembly(fasta_path, result_path):
    p = subprocess.Popen(('/home/zhaoxy/workspace/muscle -align {} -output {}').format(fasta_path, result_path),
                        shell = True,
                        stdout = subprocess.PIPE,
                        stderr = subprocess.STDOUT)
    for line in p.stdout.readlines():
        print('',end = '')

def batch_assembly(fasta_path, sigma):
    batch_size = 50
    batch_output = []
    all_record = read_file(fasta_path)
    temp_file_dir = r'./temp_file/'
    if not os.path.exists(temp_file_dir):
        os.makedirs(temp_file_dir)
    with tqdm.trange((len(all_record)//batch_size), desc='assemblying in batches', mininterval=3.0) as pbar:
        for i in range(len(all_record)//batch_size +1):
            temp_batch_path = r'./temp_file/temp_batch.fasta'
            temp_batch_output_path = r'./temp_file/temp_batch_output.fasta'
            with open(temp_batch_path, 'w')as f:
                for item in all_record[batch_size*i : batch_size*(i+1)]:
                    # if not 223 <= len(item[1]) <= 263:
                    #     continue
                    f.write(f'>{item[0]}\n')
                    f.write(f'{item[1]}\n')
            # raise('')
            assembly(temp_batch_path, temp_batch_output_path)
            consensus = get_consensus2(temp_batch_output_path, sigma)
            batch_output.append(consensus)
            pbar.update()
    batches_path = r'./temp_file/temp_batches.fasta'
    batches_output_path = r'./temp_file/temp_batches_output.fasta'
    with open(batches_path, 'w')as f:
        for idx, item in enumerate(batch_output):
            f.write(f'>batch_consensus_{idx}\n')
            f.write(f'{item}\n')
    assembly(batches_path, batches_output_path)
    consensus = get_consensus2(batches_output_path, sigma)
    return consensus

def get_consensus2(result_path, sigma):
    assembly_seqs = [str(i.seq) for i in SeqIO.parse(result_path, "fasta")]
    seq_len = max([len(item) for item in assembly_seqs])
    ratio_list = []
    ratio_div = []
    for idx in range(len(assembly_seqs)):
        assembly_seqs[idx] = assembly_seqs[idx]+'-'*(seq_len-len(assembly_seqs[idx])) if len(assembly_seqs[idx]) < seq_len else assembly_seqs[idx]
    for m in range(min([len(item) for item in assembly_seqs])):
        temp = []
        for n in range(len(assembly_seqs)):
            temp.append(assembly_seqs[n][m])
        lis = str(temp)
        if sigma == 4:
            ratio_list.append([lis.count('A'),lis.count('C'),lis.count('G'),lis.count('T')])
        if sigma == 8:
            ratio_list.append([lis.count('A') + 0.5*(lis.count('M')+lis.count('R')),
                           lis.count('C') + 0.5*(lis.count('M')+lis.count('Y')),
                           lis.count('G') + 0.5*(lis.count('K')+lis.count('R')),
                           lis.count('T') + 0.5*(lis.count('K')+lis.count('Y'))])
        ratio_div.append(lis.count('-'))

    minus_num = seq_len - 243
    t = copy.deepcopy(ratio_div)
    max_index = []
    for _ in range(minus_num):
        number = max(t)
        index = t.index(number)
        t[index] = 0
        max_index.append(index)

    consensus = []
    for idx , ratio in enumerate(ratio_list):

        if idx in max_index:
            continue

        res_A = distance.jensenshannon([1,0,0,0],ratio)
        res_C = distance.jensenshannon([0,1,0,0],ratio)
        res_G = distance.jensenshannon([0,0,1,0],ratio)
        res_T = distance.jensenshannon([0,0,0,1],ratio)
        res_M = distance.jensenshannon([1,1,0,0],ratio)
        res_K = distance.jensenshannon([0,0,1,1],ratio)
        res_R = distance.jensenshannon([1,0,1,0],ratio)
        res_Y = distance.jensenshannon([0,1,0,1],ratio)

        res = [res_A,res_C,res_G,res_T,res_M,res_K,res_R,res_Y]
        min_index = res[:sigma].index(min(res[:sigma]))
        if min_index == 0:
            consensus.append('A')
        if min_index == 1:
            consensus.append('C')
        if min_index == 2:
            consensus.append('G')
        if min_index == 3:
            consensus.append('T')
        if min_index == 4:
            consensus.append('M')
        if min_index == 5:
            consensus.append('K')
        if min_index == 6:
            consensus.append('R')
        if min_index == 7:
            consensus.append('Y')
    consensus_str = ''.join(m for m in consensus)
    return consensus_str

def alignment(ref_path, out_consensus, in_grouping_res_dir):
    
    print('in_grouping_res_dir', in_grouping_res_dir)
    print('out_consensus', out_consensus)
    print('\n\n')
    
    refs = [(str(seq.id), str(seq.seq)) for seq in SeqIO.parse(ref_path, "fasta")]

    assembly_results = []
    correct_count = 0

    for idx in range(len(refs)):
        if idx >= 55:
            break 
        print(f'Process {idx} working...')
        seq_name = f'{in_grouping_res_dir}/grouping_res_{idx}.fasta'
        assembly_result = batch_assembly(fasta_path = seq_name, sigma=4)

        assembly_results.append(assembly_result)
        correct_count += 1 if assembly_result == refs[idx][1] else 0
        print(f'Process {idx} done.')
        # raise('')
        if not assembly_result == refs[idx][1]:
            print(f'Process {idx} not correct.')
            print(f'Normal strand assembly result is \n{assembly_result}')
            print(f'Reference is \n{refs[idx][1]}')
            count = 0
            for i, bit in enumerate(assembly_result):
                try:
                    count += 1 if assembly_result[i] != refs[idx][1][i] else 0
                except IndexError:
                    print('assembly_result length did not match refs')
                    print('assembly_result length: ', len(assembly_result))
                    break
            print(f'error counts = {count}\n')

    with open(out_consensus,'w')as f:
        for idx , result in enumerate(assembly_results):
            f.write(f'>index_{idx}\n{result}\n')
    

if __name__ == '__main__':
    ref_path = r'/data/nas-shared/zhaoxy/CompositeHedges/all_seqs20230512/zxy_sustech_seqs.fasta'
    out_consensus = r'./text29_42_assembly_results_CNNLSTMselnet_sustag384_t0.20finetuned_0_5m.fasta'
    in_grouping_res_dir = r'./grouping_res_CNNLSTMselnet_sustag384_t0.20finetuned_0_5m/'
    
    alignment(ref_path, out_consensus, in_grouping_res_dir)