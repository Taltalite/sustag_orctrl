import os
import retrive_readuntil_fastq
import mapping
import readsdic_gen
import seq_grouping
import alignmentNassembly
import decode
import datetime

# start from 1
st_bcd = 29
end_bcd = 42

# DATA_MINS = [int(5), int(10), int(30), int(60), int(90), int(120), int(180), int(240), int(300), int(360), int(420), int(480), int(540), int(600)]
# DATA_MINS = [int(20), int(40), int(50), int(75), int(105), int(135), int(150)]
# DATA_MINS = [int(5), int(10), int(15), int(30), int(40), int(50), int(60), int(80), int(100), int(120), int(150), int(180), int(210)]
DATA_MINS = [int(60)]

for DATA_MIN in DATA_MINS:
    MODEL_NAME = 'ORCovL_e80_200kft'

    print(f"Model applied: {MODEL_NAME} \nBarcode target range: {st_bcd}-{end_bcd} \nRetrieved time: {DATA_MIN} min")

    input_fastq_dir = r"/data/biolab-nvme-pool1/lijy/guppy_basecalled/sustechseqs_230608/pass/" 
    in_model_pred_pickle_file = f"/home/lijy/workspace/sustag/sustechseqs_test/{MODEL_NAME}_pred_dict.P"
    retrived_fastq_dir = f"/data/biolab-nvme-pool1/lijy/sim_readuntil_fastq/{MODEL_NAME}_{DATA_MIN}m/"  

    # reference .fasta file
    # ref_fapath = r'/home/lijy/workspace/fasta/barcodeljy_384.fasta'
    ref_fapath = r'/home/lijy/workspace/fasta/sustag384.fasta'
    ref_sustseqs_path = r'./zxy_sustech_seqs.fasta'
    # output .paf files path
    out_paf_path = f"/data/biolab-nvme-pool1/lijy/sim_readuntil_paf/{MODEL_NAME}_{DATA_MIN}m/" 

    out_pickle_file = f"./pickle_files/{MODEL_NAME}_{DATA_MIN}m.P"

    out_grouping_res_dir = f"./grouping_res/grouping_res_{MODEL_NAME}_{DATA_MIN}m/"

    out_consensus = f'./asm_res/text{st_bcd}_{end_bcd}_assembly_results_{MODEL_NAME}_{DATA_MIN}m.fasta'

    out_text_file = f'./decode_res/sustech_text{st_bcd}_{end_bcd}_{MODEL_NAME}_{DATA_MIN}m.txt'

    out_pic_file = r'./sustech_logo_decode.jpg'


    target_pred = [i for i in range(st_bcd, end_bcd+1)]  # start from 1 (sustag384)
    # time_range = (0, 1*60*60)  # 0 - 1h
    time_range = (0, DATA_MIN*60)  # 0 - 5min

    current_datetime = datetime.datetime.now()
    print("Pipeline starts from: ", current_datetime)

    retrive_readuntil_fastq.filter_fastq_files(input_fastq_dir, retrived_fastq_dir, in_model_pred_pickle_file, target_pred, time_range) 

    os.makedirs(out_paf_path, exist_ok=True)
    mapping.mapping(ref_fapath, retrived_fastq_dir, out_paf_path)

    readsdic_gen.dict_gen([out_paf_path], out_pickle_file)

    seq_grouping.grouping(retrived_fastq_dir, out_pickle_file, ref_fapath, out_grouping_res_dir)

    alignmentNassembly.alignment(ref_sustseqs_path, out_consensus, out_grouping_res_dir)


    decode.decode_main(out_consensus, out_pic_file, out_text_file)

    current_datetime = datetime.datetime.now()
    print("Pipeline ends at: ", current_datetime)


