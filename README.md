# human motion TAE

motion data reconstruction by Transformer based Autoencoder

train data downloaded from https://github.com/Mathux/ACTOR/blob/master/DATASETS.md

# Environment

- Windows
- CUDA 12.8
- RTX 4060 Ti (Ada Lovelace)

### pip venv

```
pip install -r requirements.txt
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128
```

# final model structure

```
MotionBERT(
  (embedding): Linear(in_features=72, out_features=256, bias=True)
  (positional_encoding): PositionalEncoding(
    (dropout): Dropout(p=0.05, inplace=False)
  )
  (encoder_layers): ModuleList(
    (0-7): 8 x EncoderLayer(
      (self_attn): MultiHeadSelfAttention(
        (wq): Linear(in_features=256, out_features=256, bias=True)
        (wk): Linear(in_features=256, out_features=256, bias=True)
        (wv): Linear(in_features=256, out_features=256, bias=True)
        (fc_out): Linear(in_features=256, out_features=256, bias=True)
        (dropout): Dropout(p=0.05, inplace=False)
      )
      (feed_forward): FeedForward(
        (sequential): Sequential(
          (0): Linear(in_features=256, out_features=1024, bias=True)
          (1): GELU(approximate='none')
          (2): Dropout(p=0.05, inplace=False)
          (3): Linear(in_features=1024, out_features=256, bias=True)
        )
      )
      (norm1): LayerNorm((256,), eps=1e-05, elementwise_affine=True)
      (norm2): LayerNorm((256,), eps=1e-05, elementwise_affine=True)
      (dropout1): Dropout(p=0.05, inplace=False)
      (dropout2): Dropout(p=0.05, inplace=False)
    )
  )
  (output_layer): Linear(in_features=256, out_features=72, bias=True)
)
```
