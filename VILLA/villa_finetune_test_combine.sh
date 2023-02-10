#set -e
SELF=$(dirname "$(realpath $0)")
SRC="$SELF"
DATA_DIR="$SELF"
PRETRAIN="$SELF/pretrain"
DATASET=("mami" "harmeme" "fb")
#DATASET=("mami")
OCRS=("easyOCR" "googleOCR")
ATTACK_LIST=("newsprint" "s&p" "s&p_0.4" "s&p_text_0.2" "spread_1" "spread_3" "blur_text_5" "orig_ocr" "with_sp_5px" "without_sp_5px")
#ATTACK_LIST=("s&p_0.4")

for DS in "${DATASET[@]}"; do
  MEME_ROOT_DIR="$SELF/../datasets/$DS"
  UNITER_DIR="$DATA_DIR/features/$DS"
  UNITER_CKPT="$SELF/saved/$DS"
  if [ ! -d $UNITER_DIR ]; then
      mkdir -p $UNITER_DIR
  fi
  for OCR in "${OCRS[@]}"; do
    for ATTACK in "${ATTACK_LIST[@]}"; do
#	echo "trying to delete txt feature directory"
 #       if [ -d "$UNITER_DIR/$ATTACK/$OCR/MEME_NPZ" ]; then
  #          rm -rf "$UNITER_DIR/$ATTACK/$OCR/MEME_NPZ"
   #         echo "worked"
       # fi
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
    #    if [ -d "$UNITER_DIR/$ATTACK/$OCR/MEME_NPZ_DB" ]; then
     #       rm -rf "$UNITER_DIR/$ATTACK/$OCR/MEME_NPZ_DB"
      #      echo "worked"
       # fi
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
	#echo "trying to delete txt feature directory"
	#if [ -d "$OUT_DIR/nlvr2_test.db" ]; then
         #   rm -rf "$OUT_DIR/nlvr2_test.db"
	  #  echo "worked"
        #fi
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
  done
done

for DS in "${DATASET[@]}"; do
  MEME_ROOT_DIR="$SELF/../datasets/$DS"
  UNITER_DIR="$DATA_DIR/features/$DS"
  UNITER_CKPT="$SELF/saved/$DS"
  if [ ! -d $UNITER_DIR ]; then
      mkdir -p $UNITER_DIR
  fi
  for OCR in "${OCRS[@]}"; do
    for ATTACK in "${ATTACK_LIST[@]}"; do
      TXT_DB="$UNITER_DIR/$ATTACK/$OCR/txt_db"
      IMG_DIR="$UNITER_DIR/$ATTACK/$OCR/MEME_NPZ_DB"


      #MODEL_LIST=("villa-base-1-advimagel2" "villa-base-1-advtextimagel2gussiannoise_advstep100" "villa-base-1-advtextimagel2" "villa-base-1-advtextimagel2gussiannoise_advstep20" "villa-base-1-advtextimagel2gussiannoise" "villa-base-1-advtextimagel2gussiannoise_advstep6" "villa-base-1-advtextimagel2gussiannoise_advstep10" "villa-base-1-advtextl2")

      MODEL_LIST=("villa-base-1-advtextimagel2normalnoise_advstep2" "villa-base-1-advtextimagel2normalnoise_advstep4" "villa-base-1-advtextimagel2normalnoise_advstep6" "villa-base-1-advtextimagel2normalnoise_advstep10" "villa-base-1-advtextimagel2guassiannoise_advstep1" "villa-base-1-advtextimagel2guassiannoise_advstep2" "villa-base-1-advtextimagel2guassiannoise_advstep4" "villa-base-1-advtextimagel2guassiannoise_advstep5")


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
                  --mount src="$SRC",dst=/result,type=bind \
                  -w /src dsfhe49854/uniter \
                  python3 test_meme_attacks.py \
                  --config "config/final_test/$MODEL.json" \
                  --output_dir "/txt/$MODEL" \
                  --result_file "/result/attack_results_adv_steps.tsv" \
                  --dataset "$DS" \
                  --model "$MODEL" \
                  --ocr "$OCR" \
                  --attack "$ATTACK" 
          fi;
      done;
    done
  done
done
