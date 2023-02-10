set -e 
SELF=$(dirname "$(realpath $0)")
DATASET="mami"
MEME_ROOT_DIR="$SELF/../datasets/$DATASET"
DATA_DIR="$SELF"
UNITER_DIR="$DATA_DIR/features/$DATASET"

if [ ! -d $UNITER_DIR ]; then
    mkdir -p $UNITER_DIR
fi

if [ ! -d "$UNITER_DIR/MEME_NPZ" ]; then
    mkdir -p "$UNITER_DIR/MEME_NPZ"
    docker run --gpus all --ipc=host --rm \
        --mount src="$MEME_ROOT_DIR/img",dst=/img,type=bind,readonly \
        --mount src="$UNITER_DIR/MEME_NPZ",dst=/output,type=bind \
        -w /src chenrocks/butd-caffe:nlvr2 \
        bash -c "python tools/generate_npz.py --gpu 0"
fi


SRC="$SELF"
OUT_DIR="$UNITER_DIR/MEME_NPZ_DB"
echo "converting image features ..."
if [ ! -d $OUT_DIR ]; then
    mkdir -p $OUT_DIR

    IMG_NPY="$UNITER_DIR/MEME_NPZ"
    NAME=$(basename $IMG_NPY)
    docker run --ipc=host --rm  \
        --mount src="$SRC",dst=/src,type=bind \
        --mount src="$OUT_DIR",dst=/img_db,type=bind \
        --mount src="$IMG_NPY",dst=/$NAME,type=bind,readonly \
        -w /src dsfhe49854/uniter \
        python scripts/convert_imgdir.py --img_dir /$NAME --output /img_db
    echo "done"
fi;




OUT_DIR="$UNITER_DIR/txt_db"
for SPLIT in 'train' 'test' 'val'; do
    if [ ! -d "$OUT_DIR/nlvr2_${SPLIT}.db" ]; then
        echo "preprocessing ${SPLIT} annotations..."
        mkdir -p $OUT_DIR
        # add here script from json to jsonl if not available
        docker run --ipc=host --rm  \
            --mount src="$SRC",dst=/src,type=bind \
            --mount src="$OUT_DIR",dst=/txt_db,type=bind \
            --mount src="$MEME_ROOT_DIR/files",dst=/ann,type=bind,readonly \
            -w /src dsfhe49854/uniter \
            python prepro_without_web_anno.py --annotation /ann/$SPLIT.jsonl \
                            --output /txt_db/nlvr2_${SPLIT}.db
    fi;
done;

PRETRAIN="$SELF/pretrain"
UNITER_CKPT="$SELF/saved/$DATASET"

TXT_DB="$UNITER_DIR/txt_db"
IMG_DIR="$UNITER_DIR/MEME_NPZ_DB"

mkdir -p $UNITER_CKPT

MODEL_LIST=("villa-base-1")

for MODEL in "${MODEL_LIST[@]}"; do
    if [ ! -d "$UNITER_CKPT/$MODEL" ]; then
        echo " ********** [Train $MODEL] ********** "
        echo " ********** [Train $MODEL] ********** "
        echo " ********** [Train $MODEL] ********** "

        docker run --gpus all --ipc=host \
            -v "$SRC":/src \
            --mount src="$UNITER_CKPT",dst=/storage,type=bind \
            --mount src="$PRETRAIN",dst=/pretrain,type=bind,readonly \
            --mount src="$TXT_DB",dst=/txt,type=bind,readonly \
            --mount src="$IMG_DIR",dst=/img,type=bind,readonly \
            -w /src dsfhe49854/uniter \
            python3 train_meme_itm.py --config "config/final_train/$MODEL.json"
    fi;
<<comment
    if [ -d "$UNITER_CKPT/$MODEL" ] && [ ! -e "$UNITER_CKPT/$MODEL/test.csv" ]; then
        echo " ********** [Test $MODEL] ********** "
        echo " ********** [Test $MODEL] ********** "
        echo " ********** [Test $MODEL] ********** "

        docker run --gpus all --ipc=host \
            -v "$SRC":/src \
            --mount src="$UNITER_CKPT",dst=/storage,type=bind \
            --mount src="$PRETRAIN",dst=/pretrain,type=bind,readonly \
            --mount src="$TXT_DB",dst=/txt,type=bind,readonly \
            --mount src="$IMG_DIR",dst=/img,type=bind,readonly \
            -w /src dsfhe49854/uniter \
            python3 test_meme_itm.py --config "config/final_test/$MODEL.json"
    fi;
comment
done;

