conda activate dvd

python generate_tsv.py \
--net res101 \
--dataset vg \
--out visdial_val_features.csv \
--cuda \
--image_dir /home/luchenyu/dvd/attention_map_vis/VisualDialog_val2018/ \
--load_dir data/pretrained_model/ \
