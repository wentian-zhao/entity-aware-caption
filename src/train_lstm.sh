python train.py --user-dir . --task caption --save-dir ../checkpoint \
  --arch lstm --batch-size 48 \
  --optimizer adam --lr 1e-4 \
  --criterion adaptive_loss --adaptive-softmax-cutoff 5000,20000 \
  --ddp-backend=no_c10d