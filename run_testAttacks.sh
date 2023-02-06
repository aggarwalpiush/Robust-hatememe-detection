set -e

SELF=$(dirname "$(realpath $0)")
DATA_DIR="$SELF"
#DATASET_LIST=("harmeme" "mami" "fb")
DATASET_LIST=("harmeme")
#OCR_LIST=("googleOCR" "easyOCR")
OCR_LIST=("googleOCR")
MODEL_LIST=("roberta_resnet")
#MODEL_LIST=("villa" "uniter" "visual_bert")
for DATASET in "${DATASET_LIST[@]}"; do
    for OCR in "${OCR_LIST[@]}"; do
        MEME_ATTACK_DIR="$DATA_DIR/datasets/$DATASET/files_new"
        cp "$MEME_ATTACK_DIR/test_${OCR}_withlove.json" "$MEME_ATTACK_DIR/test.json"
	for MODEL in "${MODEL_LIST[@]}"; do
	  echo "$MODEL<=======>$OCR<===========>$DATASET"
	  echo "$MODEL<=======>$OCR<===========>$DATASET"
	  echo "$MODEL<=======>$OCR<===========>$DATASET"
          TESTING_DIR="$SELF/models/$MODEL"
          cd $TESTING_DIR;
	  if [ $MODEL == "villa" ]; then
		echo "inside villa"
		python "testing_piush.py" --dataset $DATASET --ocr $OCR --tr "bert-base-cased"  --num_pos 6
	  fi
	  if [ $MODEL != "villa" ]; then
         	 python "testing_piush.py" $DATASET $OCR
 	  fi
          cd $SELF;
	  echo "<=======>END<===========>"
          echo "<=======>END<===========>"
          echo "<=======>END<===========>"

        done;
    done;
done;

