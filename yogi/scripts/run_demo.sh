# load videos into database
yogi clipset create-clips --strobed 0 /videos/human-* human
yogi clipset create-clips --strobed 0 /videos/mouse-* mouse

# split videos into frames
yogi clipset split human
yogi clipset split mouse

# human
yogi model import /home/dbutler/.yogi/json/human-right-hand-manual.json
yogi model import /home/dbutler/.yogi/json/human-right-hand-automatic.json

CUDA_VISIBLE_DEVICES=0 yogi label clipset human human-right-hand-manual
CUDA_VISIBLE_DEVICES=0 yogi label clipset human human-right-hand-automatic 

# human manual pipeline
CLIPSET=human
MODEL=human-right-hand-manual
OUTPUT_DIR=/output/$CLIPSET/$MODEL
mkdir -p $OUTPUT_DIR
yogi video for-clips --conf-threshold 0.05 --output-dir $OUTPUT_DIR $CLIPSET $MODEL

# human automatic pipeline
CLIPSET=human
MODEL=human-right-hand-automatic
OUTPUT_DIR=/output/$CLIPSET/$MODEL
mkdir -p $OUTPUT_DIR
yogi video for-clips --conf-threshold 0.05 --output-dir $OUTPUT_DIR $CLIPSET $MODEL

# mouse
yogi model import /home/dbutler/.yogi/json/mouse-left-hand.json
CUDA_VISIBLE_DEVICES=0 yogi label clipset mouse mouse-left-hand

CLIPSET=mouse
MODEL=mouse-left-hand
OUTPUT_DIR=/output/$CLIPSET/$MODEL
mkdir -p $OUTPUT_DIR
yogi video for-clips --conf-threshold 0.65 --output-dir $OUTPUT_DIR $CLIPSET $MODEL

# compress output
cd /output;
for V in */*/*;
do
  ffmpeg -i $V $(dirname $V)/small-$(basename $V);
  rm $V;
done
