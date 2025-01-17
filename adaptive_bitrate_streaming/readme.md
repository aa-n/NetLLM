## 0 文件说明
- abr_algorithm_comparison.csv为不同模型在不同数据集下的各个metric的测试结果
- baseline.py / baseline.ipynb是对baseline的测试以及处理代码
- data_analyze.ipynb是对NetLLM的分析代码

### 0.1 模型权重
- 根目录为：NetLLM
- 为减小提交文件大小，对于中间结果文件均不上传(**可通过命令随时复现**)
- 由于Github LFS限制，并未将模型权放置在代码当中，复现需要在根目录新建目录`downloaded_plms/llama/base`,在其中放置llama7b的权重`https://modelscope.cn/models/shakechen/Llama-2-7b/files`
  - 此外，由于使用transformers库加载，还需要转为hf格式，`https://blog.csdn.net/icestorm_rain/article/details/138862681?fromshare=blogdetail&sharetype=blogdetail&sharerId=138862681&sharerefer=PC&sharesource=m0_55964465&sharefrom=from_link`

## 1 硬件限制
27GB -- NVIDIA V100 / A100

## 2 推理命令
- 对于baseline(bba、mpc、genet)
  1. 创建虚拟环境
     `conda create -n abr_tf python=3.7`

  2. 安装依赖
     ```sh
     conda activate abr_tf
     pip install tensorflow-gpu==1.15
     pip install tensorboard==1.15.0
     pip install tensorboard-plugin-wit==1.8.0
     pip install tflearn==0.5.0
     pip install numba==0.53.1
     pip install gym==0.18.0
     pip install stable-baselines[mpi]==2.10.1
     pip install pandas==1.1.5
     pip install tqdm==4.62.2
     pip install protoc==3.19.0
     ```
  3. 推理
  ```sh
  python run_baseline.py --model genet --cuda-id 0
  python run_baseline.py --model mpc 
  python run_baseline.py --model bba 
  ```

- 对于NetLLM
```sh
python run_plm.py --trace oboe --test --plm-type llama --plm-size base --rank 128 --device cuda:0 --model-dir  data/ft_plms/try_llama2_7b
```

## 3 AMP优化
为进一步减少内存消耗，将模型推理时采用混合精度计算
```
from torch.cuda.amp import autocast

with autocast():
    bit_rate = model.sample(state, target_return, timestep)
```

