# template for experiments

train() 함수 변경시 --> engine/core/trainer
실험할 모델 또는 모듈 추가 --> engine/models/

exp에 추가한 모듈 및 모델 반영시 --> exp/exp.py get_model 부분 수정

실험에 사용할 model, dataloader, optimier, lr_scheduler, eval_loader, evaluator --> exp.py 수정

# EXP_template


## train
```
python -m yolox.tools.train -n yolox-s -d 8 -b 64 --fp16 -o [--cache]

python tools/train.py -expn expname -d 8 -b 64 -f exps/ex1.py  --fp16 -o [--cache]

python tools/train.py -expn segtest0 -d 4 -b 8 -f exps/citytest.py --fp16 -o


```