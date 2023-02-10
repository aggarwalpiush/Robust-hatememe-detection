#set -e 
SELF=$(dirname "$(realpath $0)")
DATASET="fb"
OCR="easyOCR"
MEME_ROOT_DIR="$SELF/../datasets/$DATASET"
DATA_DIR="$SELF"
UNITER_DIR="$DATA_DIR/features/$DATASET"
PRETRAIN="$SELF/pretrain"
UNITER_CKPT="$SELF/saved/$DATASET"
SRC="$SELF"
if [ ! -d $UNITER_DIR ]; then
    mkdir -p $UNITER_DIR
fi
ATTACK_LIST=("newsprint" "s&p" "s&p_0.4" "s&p_text_0.2" "spread_1" "spread_3" "blur_text_5" "test_imgs" "orig_ocr")
#ATTACK_LIST=("s&p")
for ATTACK in "${ATTACK_LIST[@]}"; do
    if [ ! -d "$UNITER_DIR/$ATTACK/$OCR/MEME_NPZ" ]; then
        mkdir -p "$UNITER_DIR/$ATTACK/$OCR/MEME_NPZ"
        docker run --gpus all --ipc=host --rm \
            --mount src="$MEME_ROOT_DIR/$ATTACK",dst=/img,type=bind,readonly \
            --mount src="$UNITER_DIR/$ATTACK/$OCR/MEME_NPZ",dst=/output,type=bind \
            -w /src chenrocks/butd-caffe:nlvr2 \
            bash -c "python tools/generate_npz.py --gpu 0"
    fi


    OUT_DIR="$UNITER_DIR/$ATTACK/$OCR/MEME_NPZ_DB"
    echo "converting image features ..."
    if [ ! -d $OUT_DIR ]; then
        mkdir -p $OUT_DIR

        IMG_NPY="$UNITER_DIR/$ATTACK/$OCR/MEME_NPZ"
        NAME=$(basename $IMG_NPY)
        docker run --ipc=host --rm  \
            --mount src="$SRC",dst=/src,type=bind \
            --mount src="$OUT_DIR",dst=/img_db,type=bind \
            --mount src="$IMG_NPY",dst=/$NAME,type=bind,readonly \
            -w /src dsfhe49854/uniter \
            python scripts/convert_imgdir.py --img_dir /$NAME --output /img_db
        echo "done"
    fi;




    OUT_DIR="$UNITER_DIR/$ATTACK/$OCR/txt_db"
    if [ ! -d "$OUT_DIR/nlvr2_test.db" ]; then
        echo "preprocessing ${ATTACK} annotations..."
        mkdir -p $OUT_DIR
        # add here script from json to jsonl if not available
        docker run --ipc=host --rm  \
            --mount src="$SRC",dst=/src,type=bind \
            --mount src="$OUT_DIR",dst=/txt_db,type=bind \
            --mount src="$MEME_ROOT_DIR/attack_files/$OCR",dst=/ann,type=bind,readonly \
            -w /src dsfhe49854/uniter \
            python prepro_without_web_anno.py --annotation /ann/$ATTACK.jsonl \
                            --output /txt_db/nlvr2_test.db
    fi;

done

for ATTACK in "${ATTACK_LIST[@]}"; do
    echo "$ATTACK"
    TXT_DB="$UNITER_DIR/$ATTACK/$OCR/txt_db"
    IMG_DIR="$UNITER_DIR/$ATTACK/$OCR/MEME_NPZ_DB"


    MODEL_LIST=("villa-base-1-advimagel2")


    for MODEL in "${MODEL_LIST[@]}"; do
	if [ ! -d $TXT_DB/$MODEL ]; then
            mkdir -p $TXT_DB/$MODEL
	fi

        if [ -d "$UNITER_CKPT/$MODEL" ]; then
            docker run --gpus all --ipc=host \
                -v "$SRC":/src \
                --mount src="$UNITER_CKPT",dst=/storage,type=bind \
                --mount src="$PRETRAIN",dst=/pretrain,type=bind,readonly \
                --mount src="$TXT_DB",dst=/txt,type=bind \
                --mount src="$IMG_DIR",dst=/img,type=bind,readonly \
                -w /src dsfhe49854/uniter \
                python3 test_meme_attacks.py \
                --config "config/final_test/$MODEL.json" \
		--output_dir "/txt/$MODEL"
        fi;
    done;
done
