DATASET="nytimes"
RUN_NAME="checkpoint_transformer_conv_wide_16head_weighted_$DATASET"

python train.py --user-dir . --save-dir "../save/$RUN_NAME" --tensorboard-logdir "../save/$RUN_NAME/log" \
    --task caption --dataset $DATASET --valid-subset test-small --arch transformer2_conv_mm \
    --use-image 1 --use-text 1 --weighted-roberta 1 \
    --encoder-embed-dim 1024  --encoder-attention-heads 16  \
    --decoder-layers 4 --decoder-attention-heads 16 --decoder-ffn-embed-dim 4096 --decoder-embed-dim 1024 \
    --adaptive-softmax-cutoff [5000,20000] --adaptive-softmax-factor 1.0 \
    --batch-size 16 --t-total 437600 --num-workers 2 --max-update 437600 --validate-interval 2 \
    --criterion adaptive_loss --optimizer bert_adam --lr 0.0001 --warmup 0.05 --schedule warmup_linear --adam-betas "(0.9,0.98)" --adam-eps 0.000001 --weight-decay 0.00001 --max-grad-norm 0.1 \
    --ddp-backend=no_c10d --skip-invalid-size-inputs-valid-test --fp16 --fp16-no-flatten-grads
