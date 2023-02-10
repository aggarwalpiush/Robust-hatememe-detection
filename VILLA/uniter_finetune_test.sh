#set -e 
SELF=$(dirname "$(realpath $0)")
DATASET="harmeme"
OODDATASET="mami"
MEME_ROOT_DIR="$SELF/../datasets/$DATASET"
DATA_DIR="$SELF"
UNITER_DIR="$DATA_DIR/features/$OODDATASET"
PRETRAIN="$SELF/pretrain"
UNITER_CKPT="$SELF/saved/$DATASET"
SRC="$SELF"
if [ ! -d $UNITER_DIR ]; then
    mkdir -p $UNITER_DIR
fi
ATTACK="test_imgs"
OCR="googleOCR"
TXT_DB="$UNITER_DIR/$ATTACK/$OCR/txt_db"
IMG_DIR="$UNITER_DIR/$ATTACK/$OCR/MEME_NPZ_DB"


MODEL_LIST=("uniter-large-1")

for MODEL in "${MODEL_LIST[@]}"; do
    if [ ! -d $TXT_DB/OOD/$DATASET/$MODEL ]; then
        mkdir -p $TXT_DB/OOD/$DATASET/$MODEL
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
            --output_dir "/txt/OOD/$DATASET/$MODEL"
    fi;
done;
