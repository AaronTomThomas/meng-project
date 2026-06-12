# PEFT evaluation collated results

## Grouped thesis table

| Model | Dataset | Method | Params | n | Seeds | Layers | Block | Chunks | LR | Val imp | Test imp | Test PPL reduction |
|---|---|---|---:|---:|---|---|---:|---|---:|---:|---:|---:|
| GPT-2 small | OpenWebText-10k | AKAZA-FreeZ b4 | 76076 | 1 | 0 | 1-11 | 256 | 8192/1024/1024 | 0.0003 | 0.135711 | 0.133887 | 12.531% |
| GPT-2 small | OpenWebText-10k | LoRA attn_c_proj r4 a16 | 67584 | 1 | 0 | 1-11 | 256 | 8192/1024/1024 | 0.0003 | 0.124530 | 0.123908 | 11.654% |
| GPT-2 small | OpenWebText-10k | LoReFT r4 drop0.0 | 67628 | 1 | 0 | 1-11 | 256 | 8192/1024/1024 | 0.0003 | 0.121354 | 0.120051 | 11.313% |
| GPT-2 small | PTB | AKAZA-FreeZ b1 | 25355 | 2 | 0,1 | 1-11 | 256 | 4439/350/397 | 0.0003 | 1.606634 ± 0.013525 | 1.603304 ± 0.012950 | 79.876 ± 0.261% |
| GPT-2 small | PTB | AKAZA-FreeZ b4 | 76076 | 2 | 0,1 | 1-11 | 256 | 4439/350/397 | 0.0003 | 1.750765 ± 0.000059 | 1.734042 ± 0.002858 | 82.343 ± 0.050% |
| GPT-2 small | PTB | LoRA attn_c_proj r4 a16 | 67584 | 2 | 0,1 | 1-11 | 256 | 4439/350/397 | 0.0003 | 1.695823 ± 0.000877 | 1.680879 ± 0.004104 | 81.379 ± 0.076% |
| GPT-2 small | PTB | LoReFT r4 drop0.0 | 67628 | 2 | 0,1 | 1-11 | 256 | 4439/350/397 | 0.0003 | 1.787449 ± 0.001301 | 1.771963 ± 0.000581 | 83.000 ± 0.010% |
| GPT-2 small | WikiText-2 | AKAZA-FreeZ b1 | 25355 | 2 | 0,1 | 1-11 | 256 | 1609/143/188 | 0.0003 | 0.369531 ± 0.000783 | 0.373423 ± 0.000065 | 31.163 ± 0.004% |
| GPT-2 small | WikiText-2 | AKAZA-FreeZ b4 | 76076 | 2 | 0,1 | 1-11 | 256 | 1609/143/188 | 0.0003 | 0.389647 ± 0.000628 | 0.393246 ± 0.000037 | 32.514 ± 0.003% |
| GPT-2 small | WikiText-2 | LoRA attn_c_proj r4 a16 | 67584 | 2 | 0,1 | 1-11 | 256 | 1609/143/188 | 0.0003 | 0.355458 ± 0.001080 | 0.365772 ± 0.003723 | 30.634 ± 0.258% |
| GPT-2 small | WikiText-2 | LoReFT r4 drop0.0 | 67628 | 2 | 0,1 | 1-11 | 256 | 1609/143/188 | 0.0003 | 0.377185 ± 0.000440 | 0.378225 ± 0.001148 | 31.492 ± 0.079% |
| Pythia-160M | OpenWebText-10k | AKAZA-FreeZ b4 | 76076 | 1 | 0 | 1-11 | 256 | 8192/1024/1024 | 0.0003 | -0.219879 | -0.228775 | -25.706% |
| Pythia-160M | OpenWebText-10k | LoRA attn_dense r4 a16 | 67584 | 1 | 0 | 1-11 | 256 | 8192/1024/1024 | 0.0003 | -0.215408 | -0.222504 | -24.920% |
| Pythia-160M | OpenWebText-10k | LoReFT r4 drop0.0 | 67628 | 1 | 0 | 1-11 | 256 | 8192/1024/1024 | 0.0003 | -0.240791 | -0.246719 | -27.982% |
| Pythia-160M | PTB | AKAZA-FreeZ b4 | 76076 | 2 | 0,1 | 1-11 | 256 | 4456/352/398 | 0.0003 | 1.198375 ± 0.008475 | 1.177842 ± 0.010355 | 69.205 ± 0.319% |
| Pythia-160M | PTB | LoRA attn_dense r4 a16 | 67584 | 2 | 0,1 | 1-11 | 256 | 4456/352/398 | 0.0003 | 1.047852 ± 0.016008 | 1.040137 ± 0.013179 | 64.658 ± 0.466% |
| Pythia-160M | PTB | LoReFT r4 drop0.0 | 67628 | 2 | 0,1 | 1-11 | 256 | 4456/352/398 | 0.0003 | 1.243497 ± 0.006309 | 1.223800 ± 0.004518 | 70.589 ± 0.133% |
| Pythia-160M | WikiText-2 | AKAZA-FreeZ b4 | 76076 | 2 | 0,1 | 1-11 | 256 | 8192/986/1127 | 0.0003 | 0.470824 ± 0.002025 | 0.467444 ± 0.001170 | 37.340 ± 0.073% |
| Pythia-160M | WikiText-2 | LoRA attn_dense r4 a16 | 67584 | 2 | 0,1 | 1-11 | 256 | 8192/986/1127 | 0.0003 | 0.414439 ± 0.007233 | 0.414948 ± 0.000216 | 33.963 ± 0.014% |
| Pythia-160M | WikiText-2 | LoReFT r4 drop0.0 | 67628 | 2 | 0,1 | 1-11 | 256 | 8192/986/1127 | 0.0003 | 0.477701 ± 0.012065 | 0.473257 ± 0.014863 | 37.700 ± 0.926% |
| Pythia-1B | WikiText-2 | AKAZA-FreeZ b1 | 98320 | 1 | 0 | 0-15 | 256 | 512/128/128 | 3e-05 | 0.702721 | 0.673325 | 48.999% |
| Pythia-1B | WikiText-2 | LoRA attn_dense r4 a4 | 262144 | 1 | 0 | 0-15 | 256 | 512/128/128 | 3e-05 | 0.706604 | 0.673752 | 49.021% |
| Pythia-1B | WikiText-2 | LoReFT r4 drop0.05 | 262208 | 1 | 0 | 0-15 | 256 | 512/128/128 | 3e-05 | 0.728226 | 0.691055 | 49.895% |

## Per-run table

| Model | Dataset | Run | Method | Seed | Params | Layers | Block | Chunks | LR | Best epoch | Val imp | Test imp | Test PPL reduction |
|---|---|---|---|---:|---:|---|---:|---|---:|---:|---:|---:|---:|
| GPT-2 small | OpenWebText-10k | `owt10k_gpt2_akaza_b4_seed0` | AKAZA-FreeZ b4 | 0 | 76076 | 1-11 | 256 | 8192/1024/1024 | 0.0003 | 6 | 0.135711 | 0.133887 | 12.531% |
| GPT-2 small | OpenWebText-10k | `owt10k_gpt2_lora_attn_c_proj_r4_a16_seed0` | LoRA attn_c_proj r4 a16 | 0 | 67584 | 1-11 | 256 | 8192/1024/1024 | 0.0003 | 4 | 0.124530 | 0.123908 | 11.654% |
| GPT-2 small | OpenWebText-10k | `owt10k_gpt2_loreft_r4_seed0` | LoReFT r4 drop0.0 | 0 | 67628 | 1-11 | 256 | 8192/1024/1024 | 0.0003 | 3 | 0.121354 | 0.120051 | 11.313% |
| GPT-2 small | PTB | `ptb_gpt2_akaza_b1_seed0` | AKAZA-FreeZ b1 | 0 | 25355 | 1-11 | 256 | 4439/350/397 | 0.0003 | 50 | 1.616198 | 1.612460 | 80.060% |
| GPT-2 small | PTB | `ptb_gpt2_akaza_b1_seed1` | AKAZA-FreeZ b1 | 1 | 25355 | 1-11 | 256 | 4439/350/397 | 0.0003 | 50 | 1.597071 | 1.594147 | 79.692% |
| GPT-2 small | PTB | `ptb_gpt2_akaza_b4_seed0` | AKAZA-FreeZ b4 | 0 | 76076 | 1-11 | 256 | 4439/350/397 | 0.0003 | 48 | 1.750723 | 1.732021 | 82.307% |
| GPT-2 small | PTB | `ptb_gpt2_akaza_b4_seed1` | AKAZA-FreeZ b4 | 1 | 76076 | 1-11 | 256 | 4439/350/397 | 0.0003 | 49 | 1.750807 | 1.736064 | 82.379% |
| GPT-2 small | PTB | `ptb_gpt2_lora_attn_c_proj_r4_a16_seed0` | LoRA attn_c_proj r4 a16 | 0 | 67584 | 1-11 | 256 | 4439/350/397 | 0.0003 | 48 | 1.696444 | 1.683781 | 81.433% |
| GPT-2 small | PTB | `ptb_gpt2_lora_attn_c_proj_r4_a16_seed1` | LoRA attn_c_proj r4 a16 | 1 | 67584 | 1-11 | 256 | 4439/350/397 | 0.0003 | 48 | 1.695203 | 1.677978 | 81.325% |
| GPT-2 small | PTB | `ptb_gpt2_loreft_r4_seed0` | LoReFT r4 drop0.0 | 0 | 67628 | 1-11 | 256 | 4439/350/397 | 0.0003 | 44 | 1.788369 | 1.772374 | 83.007% |
| GPT-2 small | PTB | `ptb_gpt2_loreft_r4_seed1` | LoReFT r4 drop0.0 | 1 | 67628 | 1-11 | 256 | 4439/350/397 | 0.0003 | 47 | 1.786529 | 1.771552 | 82.993% |
| GPT-2 small | WikiText-2 | `wt2_gpt2_akaza_b1_seed0` | AKAZA-FreeZ b1 | 0 | 25355 | 1-11 | 256 | 1609/143/188 | 0.0003 | 46 | 0.368977 | 0.373377 | 31.159% |
| GPT-2 small | WikiText-2 | `wt2_gpt2_akaza_b1_seed1` | AKAZA-FreeZ b1 | 1 | 25355 | 1-11 | 256 | 1609/143/188 | 0.0003 | 49 | 0.370085 | 0.373469 | 31.166% |
| GPT-2 small | WikiText-2 | `wt2_gpt2_akaza_b4_seed0` | AKAZA-FreeZ b4 | 0 | 76076 | 1-11 | 256 | 1609/143/188 | 0.0003 | 16 | 0.390091 | 0.393220 | 32.512% |
| GPT-2 small | WikiText-2 | `wt2_gpt2_akaza_b4_seed1` | AKAZA-FreeZ b4 | 1 | 76076 | 1-11 | 256 | 1609/143/188 | 0.0003 | 18 | 0.389203 | 0.393273 | 32.516% |
| GPT-2 small | WikiText-2 | `wt2_gpt2_lora_attn_c_proj_r4_a16_seed0` | LoRA attn_c_proj r4 a16 | 0 | 67584 | 1-11 | 256 | 1609/143/188 | 0.0003 | 14 | 0.354694 | 0.363139 | 30.451% |
| GPT-2 small | WikiText-2 | `wt2_gpt2_lora_attn_c_proj_r4_a16_seed1` | LoRA attn_c_proj r4 a16 | 1 | 67584 | 1-11 | 256 | 1609/143/188 | 0.0003 | 13 | 0.356221 | 0.368404 | 30.816% |
| GPT-2 small | WikiText-2 | `wt2_gpt2_loreft_r4_drop0_seed0` | LoReFT r4 drop0.0 | 0 | 67628 | 1-11 | 256 | 1609/143/188 | 0.0003 | 9 | 0.376874 | 0.379037 | 31.548% |
| GPT-2 small | WikiText-2 | `wt2_gpt2_loreft_r4_drop0_seed1` | LoReFT r4 drop0.0 | 1 | 67628 | 1-11 | 256 | 1609/143/188 | 0.0003 | 9 | 0.377496 | 0.377414 | 31.437% |
| Pythia-160M | OpenWebText-10k | `owt10k_pythia160m_akaza_b4_seed0` | AKAZA-FreeZ b4 | 0 | 76076 | 1-11 | 256 | 8192/1024/1024 | 0.0003 | 6 | -0.219879 | -0.228775 | -25.706% |
| Pythia-160M | OpenWebText-10k | `owt10k_pythia160m_lora_attn_dense_r4_a16_seed0` | LoRA attn_dense r4 a16 | 0 | 67584 | 1-11 | 256 | 8192/1024/1024 | 0.0003 | 2 | -0.215408 | -0.222504 | -24.920% |
| Pythia-160M | OpenWebText-10k | `owt10k_pythia160m_loreft_r4_drop0_seed0` | LoReFT r4 drop0.0 | 0 | 67628 | 1-11 | 256 | 8192/1024/1024 | 0.0003 | 3 | -0.240791 | -0.246719 | -27.982% |
| Pythia-160M | PTB | `ptb_pythia160m_akaza_b4_seed0` | AKAZA-FreeZ b4 | 0 | 76076 | 1-11 | 256 | 4456/352/398 | 0.0003 | 28 | 1.192383 | 1.170521 | 68.979% |
| Pythia-160M | PTB | `ptb_pythia160m_akaza_b4_seed1` | AKAZA-FreeZ b4 | 1 | 76076 | 1-11 | 256 | 4456/352/398 | 0.0003 | 34 | 1.204368 | 1.185164 | 69.430% |
| Pythia-160M | PTB | `ptb_pythia160m_lora_attn_dense_r4_a16_seed0` | LoRA attn_dense r4 a16 | 0 | 67584 | 1-11 | 256 | 4456/352/398 | 0.0003 | 24 | 1.059171 | 1.049456 | 64.987% |
| Pythia-160M | PTB | `ptb_pythia160m_lora_attn_dense_r4_a16_seed1` | LoRA attn_dense r4 a16 | 1 | 67584 | 1-11 | 256 | 4456/352/398 | 0.0003 | 13 | 1.036532 | 1.030818 | 64.329% |
| Pythia-160M | PTB | `ptb_pythia160m_loreft_r4_drop0_seed0` | LoReFT r4 drop0.0 | 0 | 67628 | 1-11 | 256 | 4456/352/398 | 0.0003 | 33 | 1.239036 | 1.220605 | 70.495% |
| Pythia-160M | PTB | `ptb_pythia160m_loreft_r4_drop0_seed1` | LoReFT r4 drop0.0 | 1 | 67628 | 1-11 | 256 | 4456/352/398 | 0.0003 | 38 | 1.247958 | 1.226994 | 70.683% |
| Pythia-160M | WikiText-2 | `wt2_pythia160m_akaza_b4_seed0` | AKAZA-FreeZ b4 | 0 | 76076 | 1-11 | 256 | 8192/986/1127 | 0.0003 | 13 | 0.469392 | 0.466617 | 37.288% |
| Pythia-160M | WikiText-2 | `wt2_pythia160m_akaza_b4_seed1` | AKAZA-FreeZ b4 | 1 | 76076 | 1-11 | 256 | 8192/986/1127 | 0.0003 | 27 | 0.472256 | 0.468272 | 37.392% |
| Pythia-160M | WikiText-2 | `wt2_pythia160m_lora_attn_dense_r4_a16_seed0` | LoRA attn_dense r4 a16 | 0 | 67584 | 1-11 | 256 | 8192/986/1127 | 0.0003 | 7 | 0.409324 | 0.415101 | 33.973% |
| Pythia-160M | WikiText-2 | `wt2_pythia160m_lora_attn_dense_r4_a16_seed1` | LoRA attn_dense r4 a16 | 1 | 67584 | 1-11 | 256 | 8192/986/1127 | 0.0003 | 17 | 0.419553 | 0.414796 | 33.952% |
| Pythia-160M | WikiText-2 | `wt2_pythia160m_loreft_r4_drop0_seed0` | LoReFT r4 drop0.0 | 0 | 67628 | 1-11 | 256 | 8192/986/1127 | 0.0003 | 8 | 0.469170 | 0.462747 | 37.045% |
| Pythia-160M | WikiText-2 | `wt2_pythia160m_loreft_r4_drop0_seed1` | LoReFT r4 drop0.0 | 1 | 67628 | 1-11 | 256 | 8192/986/1127 | 0.0003 | 25 | 0.486233 | 0.483767 | 38.354% |
| Pythia-1B | WikiText-2 | `wt2_pythia1b_akaza_b1_all_layers_b256_seed0` | AKAZA-FreeZ b1 | 0 | 98320 | 0-15 | 256 | 512/128/128 | 3e-05 | 42 | 0.702721 | 0.673325 | 48.999% |
| Pythia-1B | WikiText-2 | `wt2_pythia1b_lora_attn_dense_r4_a4_all_layers_b256_seed0` | LoRA attn_dense r4 a4 | 0 | 262144 | 0-15 | 256 | 512/128/128 | 3e-05 | 16 | 0.706604 | 0.673752 | 49.021% |
| Pythia-1B | WikiText-2 | `wt2_pythia1b_loreft_r4_drop005_all_layers_b256_seed0` | LoReFT r4 drop0.05 | 0 | 262208 | 0-15 | 256 | 512/128/128 | 3e-05 | 10 | 0.728226 | 0.691055 | 49.895% |